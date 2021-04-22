#pragma once

#include <iostream>
#include <memory>

#include <fmt/format.h>
#include <scmd.hh>

#include "proto.hh"
#include "scylla_blas/utils/scylla_types.hh"
#include "scylla_blas/utils/utils.hh"

namespace scylla_blas {
class empty_container_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class scylla_queue {
    int64_t queue_id;
    std::shared_ptr<scmd::session> _session;
    bool multi_producer;
    bool multi_consumer;
    int64_t cnt_new;
    int64_t cnt_used;

    scmd::prepared_query fetch_counters_stmt;

    scmd::prepared_query update_new_counter_prepared;
    scmd::prepared_query update_new_counter_trans_prepared;

    scmd::prepared_query update_used_counter_prepared;
    scmd::prepared_query update_used_counter_trans_prepared;

    scmd::prepared_query fetch_task_by_id_prepared;
    scmd::prepared_query insert_task_prepared;

    scmd::prepared_query mark_task_finished_prepared;
    scmd::prepared_query check_task_finished_prepared;
    scmd::prepared_query get_task_response;

public:
    using task = proto::task;
    using response = proto::response;

    static void init_meta(const std::shared_ptr<scmd::session> &session);

    [[maybe_unused]] static void deinit_meta(const std::shared_ptr<scmd::session> &session);

    static bool queue_exists(const std::shared_ptr<scmd::session> &session, int64_t id);

    // Creates new queue, with given id.
    // Should not be called if such queue already exists.
    // multi_producer, multi_consumer - mark the queue as able to handle multiple producers/consumers.
    // 2 objects are considered multi_(producer/consumer), if one is constructed before other is destroyed,
    // and the method (produce/consume) is called at least once on each of them.
    // If you know there are no multi (producers/consumers) you should set relevant parameter to false,
    // as this will improve performance.
    static void create_queue(const std::shared_ptr<scmd::session> &session, int64_t id, bool multi_producer = false, bool multi_consumer = true);

    static void delete_queue(const std::shared_ptr<scmd::session> &session, int64_t id);

    // Creates new queue client, and connects to queue with given id.
    // Queue with given id must be created before constructing this object, \
    // using "create_queue" method.
    scylla_queue(const std::shared_ptr<scmd::session> &session, int64_t id);

    // Serializes given task (by casting to char array), and pushes it to queue.
    // Shouldn't throw exceptions until something is broken, e.g. queue was deleted.
    // It can throw std::runtime_error or scmd::exception in those cases.
    // After successful execution returns id of the inserted task.
    int64_t produce(const task &task);

    // Version of produce that pushes multiple tasks to queue.
    // It should be more performant than calling normal version of produce multiple times.
    // Returns task_id, where { task_id, task_id + 1, ..., task_id + tasks.size() - 1 }
    // are the ids of corresponding tasks.
    int64_t produce(const std::vector<task> &tasks);

    // Tries to fetch first item from queue, deserializes and returns it.
    // Returns std:nullopt if queue is empty.
    // Acts same as produce exception-wise.
    // Returns id of fetched task (the same that produce returned for this task),
    // and deserialized task struct.
    // Returned id can be used to mark the task as finished.
    std::optional<std::pair<int64_t, task>> consume();

    // Marks given task as finished, with empty reponse
    void mark_as_finished(int64_t id);

    // Marks given task as finished, with given reponse
    void mark_as_finished(int64_t id, const response& response);

    bool is_finished(int64_t id);

    std::optional<response> get_response(int64_t id);

private:
    void update_counters();

    scmd::statement prepare_insert_query(int64_t task_id, const task &task);

    scmd::batch_query prepare_batch_insert_query(int64_t base_id, const std::vector<task> &tasks);

    scmd::future insert_task(int64_t task_id, const task &task);

    int64_t produce_simple(const task &task);

    int64_t produce_vec_simple(const std::vector<task> &tasks);

    int64_t produce_multi(const task &task);

    int64_t produce_vec_multi(const std::vector<task> &tasks);

    static task task_from_value(const CassValue *v);

    static response response_from_value(const CassValue *v);

    task fetch_task_loop(int64_t task_id);

    std::optional<std::pair<int64_t, task>> consume_simple();

    std::optional<std::pair<int64_t, task>> consume_multi();
};
}
