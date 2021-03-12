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
    static void init_meta(const std::shared_ptr<scmd::session>& session);

    [[maybe_unused]] static void deinit_meta(const std::shared_ptr<scmd::session>& session);

    static bool queue_exists(const std::shared_ptr<scmd::session>& session, int64_t id);

    static void create_queue(const std::shared_ptr<scmd::session>& session, int64_t id, bool multi_producer = false, bool multi_consumer = true);

    static void delete_queue(const std::shared_ptr<scmd::session>& session, int64_t id);

    scylla_queue(const std::shared_ptr<scmd::session>& session, int64_t id);

    int64_t produce(const struct task& task);

    struct task consume();

private:
    void update_counters();

    scmd::future insert_task(int64_t task_id, const struct task& task);

    int64_t produce_simple(const struct task& task);

    int64_t produce_multi(const struct task& task);

    static struct task task_from_value(const CassValue *v);

    struct task fetch_task_loop(int64_t task_id);

    struct task consume_simple();

    struct task consume_multi();


};
}
