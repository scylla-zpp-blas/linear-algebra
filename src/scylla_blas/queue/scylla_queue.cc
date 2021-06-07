#include <scylla_blas/logging/logging.hh>
#include <scylla_blas/queue/scylla_queue.hh>

using task = scylla_blas::scylla_queue::task;
using response = scylla_blas::scylla_queue::response;
using scylla_queue = scylla_blas::scylla_queue;
using session_map_t = std::unordered_map<scmd::session *, std::unordered_set<scylla_queue *>>;

session_map_t scylla_queue::session_map = session_map_t{};

void scylla_blas::scylla_queue::init_meta(const std::shared_ptr<scmd::session> &session) {
    scmd::statement create_meta_table(R"(CREATE TABLE IF NOT EXISTS blas.queue_meta (
                                            queue_id bigint PRIMARY KEY,
                                            multi_producer BOOLEAN,
                                            multi_consumer BOOLEAN,
                                            cnt_new BIGINT,
                                            cnt_used BIGINT
                                        ))");
    create_meta_table.set_timeout(0);
    auto future_1 = session->execute_async(create_meta_table);

    scmd::statement create_table(R"(
            CREATE TABLE blas.queue_data (
                queue_id bigint,
                task_id bigint,
                is_finished BOOLEAN,
                value BLOB,
                response BLOB,
                PRIMARY KEY(queue_id, task_id)
            ))");
    create_table.set_timeout(0);
    auto future_2 = session->execute_async(create_table);

    future_1.wait();
    future_2.wait();
}

[[maybe_unused]] void scylla_blas::scylla_queue::deinit_meta(const std::shared_ptr<scmd::session> &session) {
    auto future_1 = session->execute_async("DROP TABLE IF EXISTS blas.queue_meta");
    auto future_2 = session->execute_async("DROP TABLE IF EXISTS blas.queue_data");
    future_1.wait();
    future_2.wait();
}

bool scylla_blas::scylla_queue::queue_exists(const std::shared_ptr<scmd::session> &session, int64_t id) {
    auto result = session->execute("SELECT * FROM blas.queue_meta WHERE queue_id = ?", id);
    return result.row_count() == 1;
}

void scylla_blas::scylla_queue::create_queue(const std::shared_ptr<scmd::session> &session,
                                             int64_t id, bool multi_producer, bool multi_consumer) {
    scmd::statement register_queue = scmd::statement(R"(
            INSERT INTO blas.queue_meta (queue_id, multi_producer, multi_consumer, cnt_new, cnt_used)
            VALUES (?, ?, ?, 0, 0))", 3);
    session->execute(register_queue, id, multi_producer, multi_consumer);
}

void scylla_blas::scylla_queue::delete_queue(const std::shared_ptr<scmd::session> &session, int64_t id) {
    auto future_1 = session->execute_async("DELETE FROM blas.queue_data WHERE queue_id = ?", id);
    auto future_2 = session->execute_async("DELETE FROM blas.queue_meta WHERE queue_id = ?", id);
    future_1.wait();
    future_2.wait();
}

scylla_blas::scylla_queue::scylla_queue(const std::shared_ptr<scmd::session> &session, int64_t id) :
        queue_id(id),
        _session(session)
{
    auto iter = session_map.find(_session.get());
    if (iter != session_map.end()) {
        scylla_queue *other = *iter->second.begin();
        copy_statements_from(other);
        iter->second.insert(this);
    } else {
        prepare_statements();
        session_map.insert({_session.get(), {this}});
    }

    auto result = session->execute("SELECT * FROM blas.queue_meta WHERE queue_id = ?", queue_id);
    if (result.row_count() != 1) {
        throw std::runtime_error(fmt::format("Tried to connect to non-existing queue (row count: {})", result.row_count()));
    }
    result.next_row();
    multi_producer = result.get_column<bool>("multi_producer");
    multi_consumer = result.get_column<bool>("multi_consumer");
    cnt_new = result.get_column<int64_t>("cnt_new");
    cnt_used = result.get_column<int64_t>("cnt_used");
}

scylla_blas::scylla_queue::scylla_queue(scylla_queue &&other) noexcept :
    queue_id(std::exchange(other.queue_id, -1)),
    _session(std::move(other._session)),
    multi_producer(other.multi_producer),
    multi_consumer(other.multi_consumer),
    cnt_new(other.cnt_new),
    cnt_used(other.cnt_used)
{
    copy_statements_from(&other);
    auto session_ptr = _session.get();
    if (session_ptr == nullptr) return;
    auto iter = session_map.find(session_ptr);
    iter->second.erase(&other);
    iter->second.insert(this);
}

scylla_blas::scylla_queue& scylla_blas::scylla_queue::operator=(scylla_queue &&other) noexcept {
    {
        auto session_ptr = _session.get();
        if (session_ptr != nullptr) {
            auto iter = session_map.find(session_ptr);
            if (iter->second.size() == 1) {
                session_map.erase(iter);
            } else {
                iter->second.erase(this);
            }
        }
    }
    queue_id = std::exchange(other.queue_id, -1);
    _session = std::move(other._session);
    multi_producer = other.multi_producer;
    multi_consumer = other.multi_consumer;
    cnt_new = other.cnt_new;
    cnt_used = other.cnt_used;
    copy_statements_from(&other);
    {
        auto session_ptr = _session.get();
        if (session_ptr != nullptr) {
            auto iter = session_map.find(session_ptr);
            iter->second.erase(&other);
            iter->second.insert(this);
        }
    }

    return *this;
}


scylla_blas::scylla_queue::~scylla_queue() {
    auto session_ptr = _session.get();
    if (session_ptr == nullptr) return;
    auto iter = session_map.find(session_ptr);
    if (iter->second.size() == 1) {
        session_map.erase(iter);
    } else {
        iter->second.erase(this);
    }
}

int64_t scylla_blas::scylla_queue::produce(const task &task) {
    if (multi_producer) {
        return produce_multi(task);
    } else {
        return produce_simple(task);
    }
}

int64_t scylla_blas::scylla_queue::produce(const std::vector<task> &tasks) {
    if (multi_producer) {
        return produce_many_multi(tasks);
    } else {
        return produce_many_simple(tasks);
    }
}

std::optional<std::pair<int64_t, task>> scylla_blas::scylla_queue::consume() {
    if (multi_consumer) {
        return consume_multi();
    } else {
        return consume_simple();
    }
}

void scylla_blas::scylla_queue::mark_as_finished(int64_t id) {
    response r = { .type = proto::R_NONE };
    mark_as_finished(id, r);
}

void scylla_blas::scylla_queue::mark_as_finished(int64_t id, const response &response) {
    scmd::statement stmt = mark_task_finished_prepared->get_statement();
    scmd_internal::throw_on_cass_error(cass_statement_bind_bytes(stmt.get_statement(), 0, reinterpret_cast<const cass_byte_t *>(&response), sizeof response));
    scmd_internal::throw_on_cass_error(cass_statement_bind_int64(stmt.get_statement(), 1, queue_id));
    scmd_internal::throw_on_cass_error(cass_statement_bind_int64(stmt.get_statement(), 2, id));
    _session->execute(stmt);
}

bool scylla_blas::scylla_queue::is_finished(int64_t id) {
    try{
        auto result = _session->execute(*check_task_finished_prepared, queue_id, id);
        if (!result.next_row()) {
            throw std::runtime_error("No task with given id");
        }
        return result.get_column<bool>("is_finished");
    } catch (const scmd::exception &ex) {
        LogWarn("Received exception while checking if task is finished: {}", ex.what());
        return false;
    }
}

std::optional<response> scylla_blas::scylla_queue::get_response(int64_t id) {
    auto result = _session->execute(*get_task_response, queue_id, id);
    if (!result.next_row()) {
        throw std::runtime_error("No task with given id");
    }
    if (result.is_column_null("response")) {
        return std::nullopt;
    }
    const CassValue *v = result.get_column_raw("response");
    return response_from_value(v);
}

// =========== PRIVATE METHODS ===========

void init_prepared(scylla_queue::shared_prepared &stmt, const std::shared_ptr<scmd::session> &sess, const std::string &str) {
    stmt = std::make_shared<scmd::prepared_query>(std::move(sess->prepare(str)));
}

void scylla_blas::scylla_queue::prepare_statements() {
    init_prepared(update_new_counter_prepared, _session, "UPDATE blas.queue_meta SET cnt_new = ? WHERE queue_id = ?");
    init_prepared(update_new_counter_trans_prepared, _session, "UPDATE blas.queue_meta SET cnt_new = ? WHERE queue_id = ? IF cnt_new = ?");

    init_prepared(fetch_counters_stmt, _session, "SELECT cnt_new, cnt_used FROM blas.queue_meta WHERE queue_id = ?");
    init_prepared(update_used_counter_prepared, _session, "UPDATE blas.queue_meta SET cnt_used = ? WHERE queue_id = ?");
    init_prepared(update_used_counter_trans_prepared, _session, "UPDATE blas.queue_meta SET cnt_used = ? WHERE queue_id = ? IF cnt_used = ? AND cnt_new >= ?");
    init_prepared(fetch_task_by_id_prepared, _session, "SELECT value FROM blas.queue_data WHERE queue_id = ? AND task_id = ?");
    init_prepared(insert_task_prepared, _session, "INSERT INTO blas.queue_data (queue_id, task_id, is_finished, value) VALUES (?, ?, False, ?)");
    init_prepared(mark_task_finished_prepared, _session, "UPDATE blas.queue_data SET is_finished = True, response = ? WHERE queue_id = ? AND task_id = ?");
    init_prepared(check_task_finished_prepared, _session, "SELECT is_finished FROM blas.queue_data WHERE queue_id = ? AND task_id = ?");
    init_prepared(get_task_response, _session, "SELECT response FROM blas.queue_data WHERE queue_id = ? AND task_id = ?");
}

void scylla_blas::scylla_queue::copy_statements_from(scylla_blas::scylla_queue *other) {
    this->update_new_counter_prepared           = other->update_new_counter_prepared;
    this->update_new_counter_trans_prepared     = other->update_new_counter_trans_prepared;
    this->fetch_counters_stmt                   = other->fetch_counters_stmt;
    this->update_used_counter_prepared          = other->update_used_counter_prepared;
    this->update_used_counter_trans_prepared    = other->update_used_counter_trans_prepared;
    this->fetch_task_by_id_prepared             = other->fetch_task_by_id_prepared;
    this->insert_task_prepared                  = other->insert_task_prepared;
    this->mark_task_finished_prepared           = other->mark_task_finished_prepared;
    this->check_task_finished_prepared          = other->check_task_finished_prepared;
    this->get_task_response                     = other->get_task_response;
}

void scylla_blas::scylla_queue::update_counters() {
    auto result = _session->execute(*fetch_counters_stmt, queue_id);
    if (!result.next_row()) {
        throw std::runtime_error("Queue deleted while working?");
    }
    cnt_new = result.get_column<int64_t>("cnt_new");
    cnt_used = result.get_column<int64_t>("cnt_used");
}

scmd::statement scylla_blas::scylla_queue::prepare_insert_query(int64_t task_id, const task &task) {
    scmd::statement insert_task = insert_task_prepared->get_statement();
    insert_task.bind(queue_id, task_id);
    // TODO: implement binding/retrieving bytes in driver and get rid of this ugliness.
    scmd_internal::throw_on_cass_error(cass_statement_bind_bytes(insert_task.get_statement(), 2,
                                                                 reinterpret_cast<const cass_byte_t *>(&task), sizeof task));
    return insert_task;
}

scmd::batch_query scylla_blas::scylla_queue::prepare_batch_insert_query(int64_t base_id, const std::vector<task> &tasks) {
    scmd::batch_query batch(CASS_BATCH_TYPE_UNLOGGED);
    for(int64_t i = 0; i < tasks.size(); i++) {
        auto stmt = prepare_insert_query(base_id + i, tasks[i]);
        batch.add_statement(stmt);
    }

    return batch;
}

scmd::future scylla_blas::scylla_queue::insert_task(int64_t task_id, const task &task) {
    auto insert_task = prepare_insert_query(task_id, task);
    return _session->execute_async(insert_task);
}

int64_t scylla_blas::scylla_queue::produce_simple(const task &task) {
    auto future_1 = insert_task(cnt_new, task);
    cnt_new++;
    auto future_2 = _session->execute_async(*update_new_counter_prepared, cnt_new, queue_id);

    future_1.wait();
    future_2.wait();

    return cnt_new - 1;
}

int64_t scylla_blas::scylla_queue::produce_many_simple(const std::vector<task> &tasks) {
    auto batch = prepare_batch_insert_query(cnt_new, tasks);
    auto future_1 = _session->execute_async(batch);
    auto future_2 = _session->execute_async(*update_new_counter_prepared, (int64_t)(cnt_new + tasks.size()), queue_id);

    int64_t first_id = cnt_new;
    cnt_new += tasks.size();

    future_1.wait();
    future_2.wait();

    return first_id;
}

int64_t scylla_blas::scylla_queue::produce_multi(const task &task) {
    update_counters();
    while(true) {
        auto result = _session->execute(*update_new_counter_trans_prepared, cnt_new + 1, queue_id, cnt_new);
        if (!result.next_row()) {
            throw std::runtime_error("Queue deleted while working?");
        }
        if (result.get_column<bool>("[applied]")) {
            insert_task(cnt_new, task).wait();
            return cnt_new++;
        } else {
            cnt_new = result.get_column<int64_t>("cnt_new");
        }
    }
}

int64_t scylla_blas::scylla_queue::produce_many_multi(const std::vector<task> &tasks) {
    update_counters();
    while(true) {
        auto result = _session->execute(*update_new_counter_trans_prepared, (int64_t)(cnt_new + tasks.size()), queue_id, cnt_new);
        if (!result.next_row()) {
            throw std::runtime_error("Queue deleted while working?");
        }
        if (result.get_column<bool>("[applied]")) {
            auto batch = prepare_batch_insert_query(cnt_new, tasks);
            auto future = _session->execute_async(batch);
            int64_t first_id = cnt_new;
            cnt_new += tasks.size();
            future.wait();
            return first_id;
        } else {
            cnt_new = result.get_column<int64_t>("cnt_new");
        }
    }
}

task scylla_blas::scylla_queue::task_from_value(const CassValue *v) {
    task ret{};
    const cass_byte_t *out_data;
    size_t out_size;
    cass_value_get_bytes(v, &out_data, &out_size);
    if (out_size != sizeof(task)) {
        throw std::runtime_error("Invalid data in queue");
    }
    memcpy(&ret, out_data, out_size);
    return ret;
}

response scylla_blas::scylla_queue::response_from_value(const CassValue *v) {
    response ret{};
    const cass_byte_t *out_data;
    size_t out_size;
    cass_value_get_bytes(v, &out_data, &out_size);
    if (out_size != sizeof(response)) {
        throw std::runtime_error("Invalid response data in queue");
    }
    memcpy(&ret, out_data, out_size);
    return ret;
}

task scylla_blas::scylla_queue::fetch_task_loop(int64_t task_id) {
    // Now we need to fetch task data - it may not be there yet, but it should be rare.
    while(true) {
        auto task_result = _session->execute(*fetch_task_by_id_prepared, queue_id, task_id);
        if (!task_result.next_row()) {
            // Task was not inserted yet, we need to wait.
            // It shouldn't happen too often, requires a race condition.
            // Maybe some sleep here?
            continue;
        }
        // TODO: bytes in driver
        return task_from_value(task_result.get_column_raw("value"));
    }
}

std::optional<std::pair<int64_t, task>> scylla_blas::scylla_queue::consume_simple() {
    // First we need to check if there is task to fetch
    // There is, if used counter is less than new counter
    update_counters();
    if (cnt_used >= cnt_new) {
        return std::nullopt;
    }

    // It is simple consume - so no other consumers, so we can automatically claim task.
    cnt_used++;
    auto future_2 = _session->execute_async(*update_used_counter_prepared, cnt_used, queue_id);
    future_2.wait();

    return std::make_pair(cnt_used - 1, fetch_task_loop(cnt_used - 1));
}

std::optional<std::pair<int64_t, task>> scylla_blas::scylla_queue::consume_multi() {
    // First we need to check if there is task to fetch
    // There is, if used counter is less than new counter
    update_counters();

    while(true) {
        if (cnt_used >= cnt_new) {
            return std::nullopt;
        }

        auto result = _session->execute(*update_used_counter_trans_prepared, cnt_used + 1, queue_id, cnt_used, cnt_new);

        if (!result.next_row()) {
            throw std::runtime_error("Queue deleted while working?");
        }
        cnt_new = result.get_column<int64_t>("cnt_new");
        cnt_used = result.get_column<int64_t>("cnt_used");

        if (result.get_column<bool>("[applied]")) {
            // We claimed a task
            cnt_used++;
            return std::make_pair(cnt_used - 1, fetch_task_loop(cnt_used - 1));
        }
    }
}
