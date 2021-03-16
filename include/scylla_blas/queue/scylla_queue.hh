#pragma once

#include <iostream>
#include <memory>

#include <fmt/format.h>
#include <scmd.hh>

#include "scylla_blas/utils/scylla_types.hh"
#include "scylla_blas/utils/utils.hh"

namespace scylla_blas {
class empty_container_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// This is the struct that will be sent trough the queue.
// We can freely modify it, to represent tasks.
// Instance of this struct will be simply cast to char array,
// stored as binary blob in the database,
// then "de-serialized" at the other end.
// This means it should not contain pointers or other data
// that can't survive such brutal transportation.
// The "data" member is there because tests are using it,
// there is no problem with changing/removing it,
// but tests must be modified accordingly.
union task {

struct {
    int64_t data;
} simple_task;

struct {
    int64_t task_queue_id;
    int64_t A_id;
    int64_t B_id;
    int64_t C_id;
} multiplication_order;

struct {
    index_type block_row;
    index_type block_column;
} compute_block;

};

class scylla_queue {
    int64_t _id;
    std::shared_ptr<scmd::session> _session;
    bool multi_producer;
    bool multi_consumer;
    int64_t cnt_new;
    int64_t cnt_used;

    scmd::statement fetch_counters_stmt;

    scmd::prepared_query update_new_counter_prepared;
    scmd::prepared_query update_new_counter_trans_prepared;

    scmd::prepared_query update_used_counter_prepared;
    scmd::prepared_query update_used_counter_trans_prepared;

    scmd::prepared_query fetch_task_by_id_prepared;
    scmd::prepared_query insert_task_prepared;

    scmd::prepared_query mark_task_finished_prepared;
    scmd::prepared_query check_task_finished_prepared;

public:
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

    // Serializes given task (by casting to char array),
    // and pushes it to queue.
    // Shouldn't throw exceptions until something is broken
    // e.g. queue was deleted.
    // It can potentially throw std::runtime_error or scmd::exception in those cases.
    // After successful execution returns id of the inserted task.
    int64_t produce(const task &task);

    // Tries to fetch first item from queue, deserializes and returns it.
    // Will throw empty_container_error if there are no tasks waiting.
    // Otherwise acts as produce exception-wise.
    // Returns id of fetched task (the same that produce returned for this task),
    // and deserialized task struct.
    // Returned id can be used to mark the task as finished.
    std::pair<int64_t, task> consume();

    void mark_as_finished(int64_t id);

    bool is_finished(int64_t id);

private:
    void update_counters();

    scmd::future insert_task(int64_t task_id, const task &task);

    int64_t produce_simple(const task &task);

    int64_t produce_multi(const task &task);

    static task task_from_value(const CassValue *v);

    task fetch_task_loop(int64_t task_id);

    std::pair<int64_t, task> consume_simple();

    std::pair<int64_t, task> consume_multi();
};
}
