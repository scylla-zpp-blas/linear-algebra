#include  <iostream>

#include "scylla_blas/routines.hh"
#include "scylla_blas/queue/worker_proc.hh"
#include "scylla_blas/utils/utils.hh"

namespace {

using scylla_blas::proto::task_type;
using scylla_blas::proto::task;

template<class T>
void assert_height_length_equal(const scylla_blas::matrix<T> &A,
                                const scylla_blas::vector<T> &Y,
                                scylla_blas::TRANSPOSE transA = scylla_blas::NoTrans) {
    if (A.get_row_count(transA) != Y.get_length()) {
        throw (std::runtime_error(fmt::format("Matrix {0} of height {1} incompatible with vector {2} of length {3}!",
                           A.get_id(), A.get_row_count(transA), Y.get_id(), Y.get_length())));
    }
}

template<class T>
void assert_width_length_equal(const scylla_blas::matrix<T> &A,
                               const scylla_blas::vector<T> &Y,
                               scylla_blas::TRANSPOSE transA = scylla_blas::NoTrans) {
    if (A.get_column_count(transA) != Y.get_length()) {
        throw (std::runtime_error(fmt::format("Matrix {0} of width {1} incompatible with vector {2} of length {3}!",
                           A.get_id(), A.get_column_count(transA), Y.get_id(), Y.get_length())));
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

template<class T>
void add_blocks_as_queue_tasks(scylla_blas::scylla_queue &queue,
                               const scylla_blas::matrix<T> &C) {
    LogInfo("Preparing multiplication task...");
    std::vector<scylla_blas::scylla_queue::task> tasks;
    tasks.reserve(C.get_blocks_height() * C.get_blocks_width());
    for (scylla_blas::index_t i = 1; i <= C.get_blocks_height(); i++) {
        for (scylla_blas::index_t j = 1; j <= C.get_blocks_width(); j++) {
            LogTrace("Scheduling block {} {} for matrix {}", i, j, C.get_id());
            tasks.push_back({
              .type = scylla_blas::proto::NONE,
              .coord {
                      .block_row = i,
                      .block_column = j
              }});
        }
    }
    queue.produce(tasks);
}

}

template<>
float scylla_blas::routine_scheduler::produce_mixed_tasks(const proto::task_type type,
                                                          const index_t KL, const index_t KU,
                                                          const UPLO Uplo, const DIAG Diag,
                                                          const int64_t A_id,
                                                          const TRANSPOSE TransA,
                                                          const float alpha,
                                                          const int64_t X_id,
                                                          const float beta,
                                                          const int64_t Y_id,
                                                          float acc, updater<float> update) {
    return produce_and_wait(this->_main_worker_queue, proto::task{
            .type = type,
            .mixed_task_float = {
                    .task_queue_id = this->_subtask_queue_id,
                    .KL = KL,
                    .KU = KU,
                    .Uplo = Uplo,
                    .Diag = Diag,
                    .A_id = A_id,
                    .TransA = TransA,
                    .alpha = alpha,
                    .X_id = X_id,
                    .beta = beta,
                    .Y_id = Y_id
            }}, _max_used_workers, _scheduler_sleep_time, acc, update);
}

template<>
double scylla_blas::routine_scheduler::produce_mixed_tasks(const proto::task_type type,
                                                           const index_t KL, const index_t KU,
                                                           const UPLO Uplo, const DIAG Diag,
                                                           const int64_t A_id,
                                                           const TRANSPOSE TransA,
                                                           const double alpha,
                                                           const int64_t X_id,
                                                           const double beta,
                                                           const int64_t Y_id,
                                                           double acc, updater<double> update) {
    return produce_and_wait(this->_main_worker_queue, proto::task{
            .type = type,
            .mixed_task_double = {
                    .task_queue_id = this->_subtask_queue_id,
                    .KL = KL,
                    .KU = KU,
                    .A_id = A_id,
                    .TransA = TransA,
                    .alpha = alpha,
                    .X_id = X_id,
                    .beta = beta,
                    .Y_id = Y_id
            }}, _max_used_workers, _scheduler_sleep_time, acc, update);
}

#define NONE 0

scylla_blas::vector<float>&
scylla_blas::routine_scheduler::sgemv(const enum TRANSPOSE TransA,
                                     const float alpha, const matrix<float> &A,
                                     const vector<float> &X, const float beta,
                                     vector<float> &Y) {
    if (X == Y) {
        throw std::runtime_error("Invalid operation: const vector X passed equal to non-const vector Y in sgemv");
    }

    assert_width_length_equal(A, X, TransA);
    assert_height_length_equal(A, Y, TransA);

    add_segments_as_queue_tasks(this->_subtask_queue, Y);

    produce_mixed_tasks<float>(proto::SGEMV, NONE, NONE, Upper, NonUnit, A.get_id(), TransA, alpha, X.get_id(), beta, Y.get_id());
    return Y;
}

scylla_blas::vector<double>&
scylla_blas::routine_scheduler::dgemv(const enum TRANSPOSE TransA,
                                     const double alpha, const matrix<double> &A,
                                     const vector<double> &X, const double beta,
                                     vector<double> &Y) {
    if (X == Y) {
        throw std::runtime_error("Invalid operation: const vector X passed equal to non-const vector Y in dgemv");
    }

    assert_width_length_equal(A, X, TransA);
    assert_height_length_equal(A, Y, TransA);
    add_segments_as_queue_tasks(this->_subtask_queue, Y);

    produce_mixed_tasks<double>(proto::DGEMV, NONE, NONE, Upper, NonUnit, A.get_id(), TransA, alpha, X.get_id(), beta, Y.get_id());
    return Y;
}

scylla_blas::vector<float>&
scylla_blas::routine_scheduler::sgbmv(const enum TRANSPOSE TransA,
                                      const int KL, const int KU,
                                      const float alpha, const matrix<float> &A,
                                      const vector<float> &X, const float beta,
                                      vector<float> &Y) {
    if (X == Y) {
        throw std::runtime_error("Invalid operation: const vector X passed equal to non-const vector Y in sgbmv");
    }

    assert_width_length_equal(A, X, TransA);
    assert_height_length_equal(A, Y, TransA);
    add_segments_as_queue_tasks(this->_subtask_queue, Y);

    produce_mixed_tasks<float>(proto::SGBMV, KL, KU, Upper, NonUnit, A.get_id(), TransA, alpha, X.get_id(), beta, Y.get_id());
    return Y;
}

scylla_blas::vector<double>&
scylla_blas::routine_scheduler::dgbmv(const enum TRANSPOSE TransA,
                                      const int KL, const int KU,
                                      const double alpha, const matrix<double> &A,
                                      const vector<double> &X, const double beta,
                                      vector<double> &Y) {
    if (X == Y) {
        throw std::runtime_error("Invalid operation: const vector X passed equal to non-const vector Y in dgbmv");
    }

    assert_width_length_equal(A, X, TransA);
    assert_height_length_equal(A, Y, TransA);
    add_segments_as_queue_tasks(this->_subtask_queue, Y);

    produce_mixed_tasks<double>(proto::DGBMV, KL, KU, Upper, NonUnit, A.get_id(), TransA, alpha, X.get_id(), beta, Y.get_id());
    return Y;
}

scylla_blas::matrix<float>&
scylla_blas::routine_scheduler::sger(const float alpha, const vector<float> &X,
                                     const vector<float> &Y, matrix<float> &A) {
    /* Leave handling X == Y to a worker */
    assert_height_length_equal(A, X);
    assert_width_length_equal(A, Y);
    add_blocks_as_queue_tasks(this->_subtask_queue, A);

    produce_mixed_tasks<float>(proto::SGER, NONE, NONE, Upper, NonUnit, A.get_id(), NoTrans, alpha, X.get_id(), NONE, Y.get_id());
    return A;
}

scylla_blas::matrix<double>&
scylla_blas::routine_scheduler::dger(const double alpha, const vector<double> &X,
                                     const vector<double> &Y, matrix<double> &A) {
    /* Leave handling X == Y to a worker */
    assert_height_length_equal(A, X);
    assert_width_length_equal(A, Y);
    add_blocks_as_queue_tasks(this->_subtask_queue, A);

    produce_mixed_tasks<double>(proto::DGER, NONE, NONE, Upper, NonUnit, A.get_id(), NoTrans, alpha, X.get_id(), NONE, Y.get_id());
    return A;
}

scylla_blas::vector<float>&
scylla_blas::routine_scheduler::strsv(const enum UPLO Uplo, const enum TRANSPOSE TransA, const enum DIAG Diag,
                                      const matrix<float> &A, vector<float> &X) {
    /* A needs to be a square matrix */
    assert_height_length_equal(A, X);
    assert_width_length_equal(A, X);
    scylla_blas::vector<float>::clear(this->_session, HELPER_FLOAT_VECTOR_ID);

    add_segments_as_queue_tasks(this->_subtask_queue, X);
    produce_vector_tasks<float>(proto::SCOPY, 1, X.get_id(), HELPER_FLOAT_VECTOR_ID);

    float error, sum;
    do {
        sum = 0;

        add_segments_as_queue_tasks(this->_subtask_queue, X);
        error = produce_mixed_tasks<float>(proto::STRSV, NONE, NONE, Uplo, Diag, A.get_id(), TransA, NONE, HELPER_FLOAT_VECTOR_ID, NONE, X.get_id(),
                                           0, [&sum](float &result, const proto::response &r) {
                                                result += r.result_float_pair.first;
                                                sum += r.result_float_pair.second;
                                            });
    } while (error / sum > EPSILON);
    return X;
}

scylla_blas::vector<double>&
scylla_blas::routine_scheduler::dtrsv(const enum UPLO Uplo, const enum TRANSPOSE TransA, const enum DIAG Diag,
                                      const matrix<double> &A, vector<double> &X) {
    /* A needs to be a square matrix */
    assert_height_length_equal(A, X);
    assert_width_length_equal(A, X);
    scylla_blas::vector<double>::clear(this->_session, HELPER_DOUBLE_VECTOR_ID);

    add_segments_as_queue_tasks(this->_subtask_queue, X);
    produce_vector_tasks<float>(proto::DCOPY, 1, X.get_id(), HELPER_DOUBLE_VECTOR_ID);

    double error, sum;
    do {
        sum = 0;

        add_segments_as_queue_tasks(this->_subtask_queue, X);
        error = produce_mixed_tasks<double>(proto::DTRSV, NONE, NONE, Uplo, Diag, A.get_id(), TransA, NONE, HELPER_DOUBLE_VECTOR_ID, NONE, X.get_id(),
                                           0, [&sum](double &result, const proto::response &r) {
                                                result += r.result_double_pair.first;
                                                sum += r.result_double_pair.second;
                                            });
    } while (error / sum > EPSILON);
    return X;
}


scylla_blas::vector<float>&
scylla_blas::routine_scheduler::stbsv(const enum UPLO Uplo, const enum TRANSPOSE TransA, const enum DIAG Diag,
                                      const int K, const matrix<float> &A, vector<float> &X) {
    assert_width_length_equal(A, X, TransA);
    scylla_blas::vector<float>::clear(this->_session, HELPER_FLOAT_VECTOR_ID);

    add_segments_as_queue_tasks(this->_subtask_queue, X);
    produce_vector_tasks<float>(proto::SCOPY, 1, X.get_id(), HELPER_FLOAT_VECTOR_ID);

    float error, sum;
    do {
        sum = 0;

        add_segments_as_queue_tasks(this->_subtask_queue, X);
        error = produce_mixed_tasks<float>(proto::STBSV, K, K, Uplo, Diag, A.get_id(), TransA, NONE, HELPER_FLOAT_VECTOR_ID, NONE, X.get_id(),
                                           0, [&sum](float &result, const proto::response &r) {
                                                result += r.result_float_pair.first;
                                                sum += r.result_float_pair.second;
                                            });
    } while (error / sum > EPSILON);
    return X;
}

scylla_blas::vector<double>&
scylla_blas::routine_scheduler::dtbsv(const enum UPLO Uplo, const enum TRANSPOSE TransA, const enum DIAG Diag,
                                      const int K, const matrix<double> &A, vector<double> &X) {
    assert_width_length_equal(A, X, TransA);
    scylla_blas::vector<double>::clear(this->_session, HELPER_DOUBLE_VECTOR_ID);

    add_segments_as_queue_tasks(this->_subtask_queue, X);
    produce_vector_tasks<float>(proto::DCOPY, 1, X.get_id(), HELPER_DOUBLE_VECTOR_ID);

    double error, sum;
    do {
        sum = 0;

        add_segments_as_queue_tasks(this->_subtask_queue, X);
        error = produce_mixed_tasks<double>(proto::DTBSV, K, K, Uplo, Diag, A.get_id(), TransA, NONE, HELPER_DOUBLE_VECTOR_ID, NONE, X.get_id(),
                                           0, [&sum](double &result, const proto::response &r) {
                                                result += r.result_double_pair.first;
                                                sum += r.result_double_pair.second;
                                            });
    } while (error / sum > EPSILON);
    return X;
}

/* TODO: move to generic / delete */
scylla_blas::matrix<float>&
scylla_blas::routine_scheduler::srmgen(float alpha, scylla_blas::matrix<float> &A) {
    add_blocks_as_queue_tasks(this->_subtask_queue, A);

    produce_mixed_tasks<float>(proto::SRMGEN, NONE, NONE, Lower, NonUnit, A.get_id(), NoTrans, alpha, NONE, NONE, NONE);
    return A;
}

scylla_blas::matrix<double>&
scylla_blas::routine_scheduler::drmgen(double alpha, scylla_blas::matrix<double> &A) {
    add_blocks_as_queue_tasks(this->_subtask_queue, A);

    produce_mixed_tasks<double>(proto::DRMGEN, NONE, NONE, Lower, NonUnit, A.get_id(), NoTrans, alpha, NONE, NONE, NONE);
    return A;
}