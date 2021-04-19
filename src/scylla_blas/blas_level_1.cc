#include  <iostream>

#include "scylla_blas/routines.hh"
#include "scylla_blas/queue/worker_proc.hh"
#include "scylla_blas/utils/utils.hh"

namespace {

template<class T>
void assert_length_equal(const scylla_blas::vector<T> &X,
                         const scylla_blas::vector<T> &Y) {
    if (X.length != Y.length) {
        throw (fmt::format("Vector {0} of length {1} incompatible with vector {2} of length {3}!",
                           X.id, X.length, Y.id, Y.length));
    }
}

template<class T>
void add_segments_as_queue_tasks(scylla_blas::scylla_queue &queue,
                                 const scylla_blas::vector<T> &X) {
    std::cerr << "Scheduling subtasks..." << std::endl;
    for (scylla_blas::index_type i = 1; i <= X.get_segment_count(); i++) {
        queue.produce({
            .type = scylla_blas::proto::NONE,
            .index = i
        });
    }
}

void produce_and_wait(scylla_blas::scylla_queue &queue,
                      const scylla_blas::proto::task &task,
                      scylla_blas::index_type cnt, int64_t sleep_time) {
    std::cerr << "Ordering task to the workers" << std::endl;
    std::vector<int64_t> task_ids;
    for (scylla_blas::index_type i = 0; i < cnt; i++) {
        task_ids.push_back(queue.produce(task));
    }

    std::cerr << "Waiting for workers to complete the order..." << std::endl;
    for (int64_t id : task_ids) {
        while (!queue.is_finished(id)) {
            scylla_blas::wait_seconds(sleep_time);
        }
    }
}

}

void scylla_blas::routine_scheduler::produce_vector_tasks(const proto::task_type type,
                                                          const float salpha,
                                                          const double dalpha,
                                                          const int64_t X_id,
                                                          const int64_t Y_id) {
    produce_and_wait(this->_main_worker_queue, proto::task {
            .type = type,
            .vector_task = {
                    .task_queue_id = this->_subtask_queue_id,
                    .salpha = salpha,
                    .dalpha = dalpha,
                    .X_id = X_id,
                    .Y_id = Y_id
            }}, LIMIT_WORKER_CONCURRENCY, WORKER_SLEEP_TIME_SECONDS);
}

#define NONE 0

void
scylla_blas::routine_scheduler::sswap(vector<float> &X, vector<float> &Y) {
    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks(proto::SSWAP, NONE, NONE, X.id, Y.id);
}

void
scylla_blas::routine_scheduler::dswap(vector<double> &X, vector<double> &Y) {
    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks(proto::DSWAP, NONE, NONE, X.id, Y.id);
}

void
scylla_blas::routine_scheduler::sscal(const float alpha, vector<float> &X) {
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks(proto::SSCAL, alpha, NONE, X.id, NONE);
}

void
scylla_blas::routine_scheduler::dscal(const double alpha, vector<double> &X) {
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks(proto::DSCAL, NONE, alpha, X.id, NONE);
}

void
scylla_blas::routine_scheduler::scopy(const vector<float> &X, vector<float> &Y) {
    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks(proto::SCOPY, NONE, NONE, X.id, Y.id);
}

void
scylla_blas::routine_scheduler::dcopy(const vector<double> &X, vector<double> &Y) {
    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks(proto::DCOPY, NONE, NONE, X.id, Y.id);
}

void
scylla_blas::routine_scheduler::saxpy(const float alpha, const vector<float> &X, vector<float> &Y) {
    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks(proto::SAXPY, alpha, NONE, X.id, Y.id);
}

void
scylla_blas::routine_scheduler::daxpy(const double alpha, const vector<double> &X, vector<double> &Y) {
    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks(proto::DAXPY, NONE, alpha, X.id, Y.id);
}