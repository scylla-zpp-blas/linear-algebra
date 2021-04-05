#include  <iostream>

#include "scylla_blas/routines.hh"
#include "scylla_blas/queue/worker_proc.hh"
#include "scylla_blas/utils/utils.hh"

namespace {

void assert_multiplication_compatible(const enum scylla_blas::TRANSPOSE TransA, const scylla_blas::basic_matrix &A,
                                      const scylla_blas::basic_matrix &B, const enum scylla_blas::TRANSPOSE TransB,
                                      const scylla_blas::basic_matrix &C) {
    using namespace scylla_blas;

    int row_a = A.row_count; int col_a = A.column_count;
    if (TransA != NoTrans) std::swap(row_a, col_a);

    int row_b = B.row_count; int col_b = B.column_count;
    if (TransB != NoTrans) std::swap(row_b, col_b);

    if (row_b != col_a) {
        throw std::runtime_error(
            fmt::format(
                    "Incompatible matrices {} of size {}x{}{} and {} of size {}x{}{}: multiplication impossible!",
                    A.id, A.row_count, A.column_count, (TransA == NoTrans ? "" : " (transposed)"),
                    B.id, B.row_count, B.column_count, (TransB == NoTrans ? "" : " (transposed)")
            )
        );
    }

    if (row_a != C.row_count || col_b != C.column_count) {
        throw std::runtime_error(
                fmt::format(
                        "Matrix {} of size {}x{} incompatible with multiplication result of matrices sized {}x{} and {}x{}!",
                        C.id, C.row_count, C.column_count, row_a, col_a, row_b, col_b
                )
        );
    }
}

template<class T>
void add_blocks_as_queue_tasks(scylla_blas::scylla_queue &queue,
                               const scylla_blas::matrix<T> &C) {
    std::cerr << "Preparing multiplication task..." << std::endl;
    for (scylla_blas::index_type i = 1; i <= C.get_blocks_height(); i++) {
        for (scylla_blas::index_type j = 1; j <= C.get_blocks_width(); j++) {
            queue.produce({
                 .type = scylla_blas::proto::NONE,
                 .coord {
                         .block_row = i,
                         .block_column = j
                 }});
        }
    }
}

void produce_and_wait(scylla_blas::scylla_queue &queue,
                      const scylla_blas::proto::task &task,
                      scylla_blas::index_type cnt, int64_t sleep_time) {
    std::cerr << "Ordering multiplication to the workers" << std::endl;
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

/* TODO: Can we use define? Or in any other way avoid these boilerplatey signatures? */
scylla_blas::matrix<float>&
scylla_blas::routine_scheduler::sgemm(const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                                      const float alpha, const matrix<float> &A,
                                      const matrix<float> &B,
                                      const float beta, scylla_blas::matrix<float> &C) {
    assert_multiplication_compatible(TransA, A, B, TransB, C);
    add_blocks_as_queue_tasks(this->_subtask_queue, C);

    produce_and_wait(this->_main_worker_queue, proto::task {
            .type = proto::SGEMM,
            .sgemm = {
                    .task_queue_id = this->_subtask_queue_id,
                    .TransA = TransA,
                    .TransB = TransB,
                    .alpha = alpha,
                    .A_id = A.id,
                    .B_id = B.id,
                    .beta = beta,
                    .C_id = C.id
            }}, LIMIT_WORKER_CONCURRENCY, WORKER_SLEEP_TIME_SECONDS);

    return C;
}

scylla_blas::matrix<double>&
scylla_blas::routine_scheduler::dgemm(const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                                      const double alpha, const matrix<double> &A,
                                      const matrix<double> &B,
                                      const double beta, scylla_blas::matrix<double> &C) {
    assert_multiplication_compatible(TransA, A, B, TransB, C);
    add_blocks_as_queue_tasks(this->_subtask_queue, C);

    produce_and_wait(this->_main_worker_queue, proto::task {
            .type = proto::DGEMM,
            .dgemm = {
                    .task_queue_id = this->_subtask_queue_id,
                    .TransA = TransA,
                    .TransB = TransB,
                    .alpha = alpha,
                    .A_id = A.id,
                    .B_id = B.id,
                    .beta = beta,
                    .C_id = C.id
            }}, LIMIT_WORKER_CONCURRENCY, WORKER_SLEEP_TIME_SECONDS);

    return C;
}

/* TODO: Do this right – only use the part of the matrices pointed to by 'Uplo' */
scylla_blas::matrix<float>&
scylla_blas::routine_scheduler::ssymm(const enum SIDE Side, const enum UPLO Uplo,
                                      const float alpha, const matrix<float> &A,
                                      const matrix<float> &B,
                                      const float beta, matrix<float> &C) {
    return sgemm(NoTrans, NoTrans,
                 alpha, (Side == Left ? A : B),
                 (Side == Left ? B : A), beta, C);
}

scylla_blas::matrix<double>&
scylla_blas::routine_scheduler::dsymm(const enum SIDE Side, const enum UPLO Uplo,
                                      const double alpha, const matrix<double> &A,
                                      const matrix<double> &B,
                                      const double beta, matrix<double> &C) {
    return dgemm(NoTrans, NoTrans,
                 alpha, (Side == Left ? A : B),
                 (Side == Left ? B : A), beta, C);
}