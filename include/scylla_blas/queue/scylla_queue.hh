#pragma once

#include <iostream>
#include <memory>

#include <fmt/format.h>
#include <session.hh>

#include <scylla_blas/utils/scylla_types.hh>
#include <scylla_blas/utils/utils.hh>

namespace scylla_blas {
class empty_container_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// We can freely modify this struct to contain all required info for task
struct task {
    int64_t data;
};

class scylla_queue {
    int64_t _id;
    std::shared_ptr<scmd::session> _session;
    bool multi_producer;
    bool multi_consumer;
    int64_t cnt_new;
    int64_t cnt_used;

    /* Can these be safely copy-constructed to a different process? */
    scmd::statement fetch_counters_stmt;

    scmd::prepared_query update_new_counter_prepared;
    scmd::prepared_query update_new_counter_trans_prepared;

    scmd::prepared_query update_used_counter_prepared;
    scmd::prepared_query update_used_counter_trans_prepared;

    scmd::prepared_query fetch_task_by_id_prepared;
    scmd::prepared_query insert_task_prepared;

public:
    static void init_meta(const std::shared_ptr<scmd::session>& session) {
        std::string init_meta = "CREATE TABLE blas.queue_meta ( "
                                "   id bigint PRIMARY KEY, "
                                "   multi_producer BOOLEAN, "
                                "   multi_consumer BOOLEAN, "
                                "   cnt_new BIGINT, "
                                "   cnt_used BIGINT"
                                ");";
        session->execute(init_meta);
    }

    static void deinit_meta(const std::shared_ptr<scmd::session>& session) {
        session->execute("DROP TABLE IF EXISTS blas.queue_meta;");
    }

    static bool queue_exists(const std::shared_ptr<scmd::session>& session, int64_t id) {
        auto result = session->execute(fmt::format("SELECT * FROM blas.queue_meta WHERE id = {};", id));
        return result.row_count() == 1;
    }

    static void create_queue(const std::shared_ptr<scmd::session>& session, int64_t id, bool multi_producer = false, bool multi_consumer = true) {
        session->execute(fmt::format(R"(
            CREATE TABLE blas.queue_{0} (
                id bigint PRIMARY KEY,
                value BLOB
            );)", id));
        session->execute<int64_t, bool, bool>(R"(
            INSERT INTO blas.queue_meta (id, multi_producer, multi_consumer, cnt_new, cnt_used) VALUES (?, ?, ?, 0, 0);"
            )", id, multi_producer, multi_consumer);
    }

    static void delete_queue(const std::shared_ptr<scmd::session>& session, int64_t id) {
        auto future_1 = session->execute_async(fmt::format("DROP TABLE IF EXISTS blas.queue_{0};", id));
        auto future_2 = session->execute_async("DELETE FROM blas.queue_meta WHERE id = ?;", id);
        future_1.wait();
        future_2.wait();
    }

    scylla_queue(const std::shared_ptr<scmd::session>& session, int64_t id) :
            _id(id),
            _session(session),
#define PREPARE(x, args...) x(_session->prepare(fmt::format(args)))
            PREPARE(update_new_counter_prepared,
                    "UPDATE blas.queue_meta SET cnt_new = ? WHERE id = {};", _id),
            PREPARE(update_new_counter_trans_prepared,
                    "UPDATE blas.queue_meta SET cnt_new = ? WHERE id = {} IF cnt_new = ?;", _id),
            fetch_counters_stmt(fmt::format(
                    "SELECT cnt_new, cnt_used FROM blas.queue_meta WHERE id = {};", _id)),
            PREPARE(update_used_counter_prepared,
                    "UPDATE blas.queue_meta SET cnt_used = ? WHERE id = {};", _id),
            // cnt_new >= ? will always be true, it is there to give us current value of cnt_new
            PREPARE(update_used_counter_trans_prepared,
                    "UPDATE blas.queue_meta SET cnt_used = ? WHERE id = {} IF cnt_used = ? AND cnt_new >= ?;", _id),
            PREPARE(fetch_task_by_id_prepared,
                    "SELECT value FROM blas.queue_{} WHERE id = ?;", _id),
            PREPARE(insert_task_prepared,
                    "INSERT INTO blas.queue_{} (id, value) VALUES (?, ?)", _id)
#undef PREPARE
    {
        auto result = session->execute("SELECT * FROM blas.queue_meta WHERE id = ?;", _id);
        if (result.row_count() != 1) {
            throw std::runtime_error(fmt::format("Tried to connect to non-existing queue (row count: {})", result.row_count()));
        }
        result.next_row();
        multi_producer = result.get_column<bool>("multi_producer");
        multi_consumer = result.get_column<bool>("multi_consumer");
        cnt_new = result.get_column<int64_t>("cnt_new");
        cnt_used = result.get_column<int64_t>("cnt_used");
    }

    int64_t produce(const struct task& task) {
        if(multi_producer) {
            return produce_multi(task);
        } else {
            return produce_simple(task);
        }
    }

    struct task consume() {
        if(multi_consumer) {
            return consume_multi();
        } else {
            return consume_simple();
        }
    }

private:
    void update_counters() {
        auto result = _session->execute(fetch_counters_stmt);
        if(!result.next_row()) {
            throw std::runtime_error("Queue deleted while working?");
        }
        cnt_new = result.get_column<int64_t>("cnt_new");
        cnt_used = result.get_column<int64_t>("cnt_used");
    }

    scmd::future insert_task(int64_t task_id, const struct task& task) {
        scmd::statement insert_task = insert_task_prepared.get_statement();
        insert_task.bind(task_id);
        // TODO: implement binding/retrieving bytes in driver and get rid of this ugliness.
        scmd_internal::throw_on_cass_error(cass_statement_bind_bytes(insert_task.get_statement(), 1,
                                                                     reinterpret_cast<const cass_byte_t *>(&task), sizeof task));
        return _session->execute_async(insert_task);
    }

    int64_t produce_simple(const struct task& task) {
        auto future_1 = insert_task(cnt_new, task);
        cnt_new++;
        auto future_2 = _session->execute_async(update_new_counter_prepared, cnt_new);

        future_1.wait();
        future_2.wait();

        return cnt_new - 1;
    }

    int64_t produce_multi(const struct task& task) {
        update_counters();
        while(true) {
            auto result = _session->execute(update_new_counter_trans_prepared, cnt_new + 1, cnt_new);
            if(!result.next_row()) {
                throw std::runtime_error("Queue deleted while working?");
            }
            if(result.get_column<bool>("[applied]")) {
                insert_task(cnt_new, task).wait();
                return cnt_new++;
            } else {
                cnt_new = result.get_column<int64_t>("cnt_new");
            }
        }
    }

    struct task task_from_value(const CassValue *v) {
        struct task ret;
        const cass_byte_t *out_data;
        size_t out_size;
        cass_value_get_bytes(v, &out_data, &out_size);
        if(out_size != sizeof(struct task)) {
            throw std::runtime_error("Invalid data in queue");
        }
        memcpy(&ret, out_data, out_size);
        return ret;
    }

    struct task fetch_task_loop(int64_t task_id) {
        // Now we need to fetch task data - it may not be there yet, but it should be rare.
        while(true) {
            auto task_result = _session->execute(fetch_task_by_id_prepared, task_id);
            if(!task_result.next_row()) {
                // Task was not inserted yet, we need to wait.
                // It shouldn't happen too often, requires a race condition.
                // Maybe some sleep here?
                continue;
            }
            // TODO: bytes in driver
            return task_from_value(task_result.get_column_raw("value"));
        }
    }

    struct task consume_simple() {
        // First we need to check if there is task to fetch
        // There is, if used counter is less than new counter
        update_counters();
        if (cnt_used >= cnt_new) {
            throw empty_container_error("No tasks to fetch right now");
        }

        // It is simple consume - so no other consumers, so we can automatically claim task.
        cnt_used++;
        auto future_2 = _session->execute_async(update_used_counter_prepared, cnt_used);
        future_2.wait();

        return fetch_task_loop(cnt_used - 1);
    }

    struct task consume_multi() {
        // First we need to check if there is task to fetch
        // There is, if used counter is less than new counter
        update_counters();

        while(true) {
            if (cnt_used >= cnt_new) {
                throw empty_container_error("No tasks to fetch right now");
            }
            auto result = _session->execute(update_used_counter_trans_prepared, cnt_used + 1, cnt_used, cnt_new);
            if(!result.next_row()) {
                throw std::runtime_error("Queue deleted while working?");
            }
            cnt_new = result.get_column<int64_t>("cnt_new");
            cnt_used = result.get_column<int64_t>("cnt_used");
            if(result.get_column<bool>("[applied]")) {
                // We claimed a task
                return fetch_task_loop(cnt_used++);
            }
        }
    }


};
}
