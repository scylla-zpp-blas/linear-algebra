#include  <iostream>

#include "scylla_blas/routines.hh"
#include "scylla_blas/queue/worker_proc.hh"
#include "scylla_blas/utils/utils.hh"

namespace {

using scylla_blas::proto::task_type;
using scylla_blas::proto::task;

template<class T>
void assert_length_equal(const scylla_blas::vector<T> &X,
                         const scylla_blas::vector<T> &Y) {
    if (X.get_length() != Y.get_length()) {
        throw (std::runtime_error(fmt::format("Vector {0} of length {1} incompatible with vector {2} of length {3}!",
                           X.get_id(), X.get_length(), Y.get_id(), Y.get_length())));
    }
}

template<class T>
void add_segments_as_queue_tasks(scylla_blas::scylla_queue &queue,
                                 const scylla_blas::vector<T> &X) {
    LogInfo("Scheduling subtasks...");
    std::vector<scylla_blas::scylla_queue::task> tasks;
    tasks.reserve(X.get_segment_count());
    for (scylla_blas::index_t i = 1; i <= X.get_segment_count(); i++) {
        tasks.push_back({
            .type = scylla_blas::proto::NONE,
            .index = i
        });
    }
    queue.produce(tasks);
}

}

template<>
float scylla_blas::routine_scheduler::produce_vector_tasks(const proto::task_type type,
                                                           const float alpha,
                                                           const int64_t X_id,
                                                           const int64_t Y_id,
                                                           float acc, updater<float> update) {
    return produce_and_wait(this->_main_worker_queue, proto::task {
        .type = type,
        .vector_task_float = {
            .task_queue_id = this->_subtask_queue.get_id(),
            .alpha = alpha,
            .X_id = X_id,
            .Y_id = Y_id
        }}, _current_worker_count, _scheduler_sleep_time, acc, update);
}

template<>
double scylla_blas::routine_scheduler::produce_vector_tasks(const proto::task_type type,
                                                            const double alpha,
                                                            const int64_t X_id,
                                                            const int64_t Y_id,
                                                            double acc, updater<double> update) {
    return produce_and_wait(this->_main_worker_queue, proto::task {
        .type = type,
        .vector_task_double = {
            .task_queue_id = this->_subtask_queue.get_id(),
            .alpha = alpha,
            .X_id = X_id,
            .Y_id = Y_id
        }}, _current_worker_count, _scheduler_sleep_time, acc, update);
}

#define NONE 0

void
scylla_blas::routine_scheduler::sswap(vector<float> &X, vector<float> &Y) {
    if (X == Y) return;
    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks<float>(proto::SSWAP, NONE, X.get_id(), Y.get_id());
}

void
scylla_blas::routine_scheduler::dswap(vector<double> &X, vector<double> &Y) {
    if (X == Y) return;
    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks<double>(proto::DSWAP, NONE, X.get_id(), Y.get_id());
}

void
scylla_blas::routine_scheduler::sscal(const float alpha, vector<float> &X) {
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks<float>(proto::SSCAL, alpha, X.get_id(), NONE);
}

void
scylla_blas::routine_scheduler::dscal(const double alpha, vector<double> &X) {
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks<double>(proto::DSCAL, alpha, X.get_id(), NONE);
}

void
scylla_blas::routine_scheduler::scopy(const vector<float> &X, vector<float> &Y) {
    if (X == Y) return;
    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks<float>(proto::SCOPY, NONE, X.get_id(), Y.get_id());
}

void
scylla_blas::routine_scheduler::dcopy(const vector<double> &X, vector<double> &Y) {
    if (X == Y) return;
    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks<double>(proto::DCOPY, NONE, X.get_id(), Y.get_id());
}

void
scylla_blas::routine_scheduler::saxpy(const float alpha, const vector<float> &X, vector<float> &Y) {
    /* (X == Y) to be handled by a worker separately */

    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks<float>(proto::SAXPY, alpha, X.get_id(), Y.get_id());
}

void
scylla_blas::routine_scheduler::daxpy(const double alpha, const vector<double> &X, vector<double> &Y) {
    /* (X == Y) to be handled by a worker separately */

    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    produce_vector_tasks<double>(proto::DAXPY, alpha, X.get_id(), Y.get_id());
}

float
scylla_blas::routine_scheduler::sdot(const vector<float> &X, const vector<float> &Y) {
    /* (X == Y) to be handled by a worker separately */

    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    return produce_vector_tasks<float>(proto::SDOT, NONE, X.get_id(), Y.get_id(), float(0),
                                       [](float &result, const proto::response& r) { result += r.result_float; });
}

double
scylla_blas::routine_scheduler::ddot(const vector<double> &X, const vector<double> &Y) {
    /* (X == Y) to be handled by a worker separately */

    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    return produce_vector_tasks<double>(proto::DDOT, NONE, X.get_id(), Y.get_id(), double(0),
                                        [](double &result, const proto::response& r) { result += r.result_double; });
}

float
scylla_blas::routine_scheduler::sdsdot(float B, const vector<float> &X, const vector<float> &Y) {
    /* (X == Y) to be handled by a worker separately */

    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    return produce_vector_tasks<double>(proto::SDSDOT, NONE, X.get_id(), Y.get_id(), double(B),
                                        [](double &result, const proto::response& r) { result += r.result_double; });
}

double
scylla_blas::routine_scheduler::dsdot(const vector<float> &X, const vector<float> &Y) {
    /* (X == Y) to be handled by a worker separately */

    assert_length_equal(X, Y);
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    return produce_vector_tasks<double>(proto::DSDOT, NONE, X.get_id(), Y.get_id(), double(0),
                                        [](double &result, const proto::response& r) { result += r.result_double; });
}

float
scylla_blas::routine_scheduler::snrm2(const vector<float> &X) {
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    return sqrtf(produce_vector_tasks<float>(proto::SNRM2, NONE, X.get_id(), NONE, float(0),
                                             [](float &result, const proto::response& r) { result += r.result_float; }));
}

double
scylla_blas::routine_scheduler::dnrm2(const vector<double> &X) {
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    return sqrt(produce_vector_tasks<double>(proto::DNRM2, NONE, X.get_id(), NONE, double(0),
                                             [](double &result, const proto::response& r) { result += r.result_double; }));
}
float

scylla_blas::routine_scheduler::sasum(const vector<float> &X) {
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    return produce_vector_tasks<float>(proto::SASUM, NONE, X.get_id(), NONE, float(0),
                                       [](float &result, const proto::response& r) { result += r.result_float; });
}

double
scylla_blas::routine_scheduler::dasum(const vector<double> &X) {
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    return produce_vector_tasks<double>(proto::DASUM, NONE, X.get_id(), NONE, double(0),
                                        [](double &result, const proto::response& r) { result += r.result_double; });
}

scylla_blas::index_t
scylla_blas::routine_scheduler::isamax(const vector<float> &X) {
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    index_t iamax = 0;
    produce_vector_tasks<float>(proto::ISAMAX, NONE, X.get_id(), NONE, float(0),
                                [&iamax](float &result, const proto::response& r) {
                                    if (result < r.result_max_float_index.value) {
                                        result = r.result_max_float_index.value;
                                        iamax = r.result_max_float_index.index;
                                    } else if (result == r.result_max_float_index.value) {
                                        iamax = std::min(iamax, r.result_max_float_index.index);
                                    }
                                });
    return iamax;
}

scylla_blas::index_t
scylla_blas::routine_scheduler::idamax(const vector<double> &X) {
    add_segments_as_queue_tasks(this->_subtask_queue, X);

    index_t iamax = 0;
    produce_vector_tasks<double>(proto::IDAMAX, NONE, X.get_id(), NONE, double(0),
                                 [&iamax](double &result, const proto::response& r) {
                                    if (result < r.result_max_double_index.value) {
                                        result = r.result_max_double_index.value;
                                        iamax = r.result_max_double_index.index;
                                    } else if (result == r.result_max_double_index.value) {
                                        iamax = std::min(iamax, r.result_max_double_index.index);
                                    }
                                });
    return iamax;
}