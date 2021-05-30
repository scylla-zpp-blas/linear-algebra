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

}

template<>
float scylla_blas::routine_scheduler::produce_matrix_tasks(const proto::task_type type,
                                                           const int64_t A_id, const enum TRANSPOSE TransA, const float alpha,
                                                           const int64_t B_id, const enum TRANSPOSE TransB, const float beta,
                                                           const int64_t C_id, float acc, updater<float> update) {
    return produce_and_wait(this->_main_worker_queue, proto::task {
        .type = type,
        .matrix_task_float = {
            .task_queue_id = this->_subtask_queue_id,

            .A_id = A_id,
            .TransA = TransA,
            .alpha = alpha,

            .B_id = B_id,
            .TransB = TransB,
            .beta = beta,

            .C_id = C_id
        }}, LIMIT_WORKER_CONCURRENCY, WORKER_SLEEP_TIME_SECONDS, acc, update);
}

template<>
double scylla_blas::routine_scheduler::produce_matrix_tasks(const proto::task_type type,
                                                            const int64_t A_id, const enum TRANSPOSE TransA, const double alpha,
                                                            const int64_t B_id, const enum TRANSPOSE TransB, const double beta,
                                                            const int64_t C_id, double acc, updater<double> update) {
    return produce_and_wait(this->_main_worker_queue, proto::task{
        .type = type,
        .matrix_task_double = {
            .task_queue_id = this->_subtask_queue_id,

            .A_id = A_id,
            .TransA = TransA,
            .alpha = alpha,

            .B_id = B_id,
            .TransB = TransB,
            .beta = beta,

            .C_id = C_id
        }}, LIMIT_WORKER_CONCURRENCY, WORKER_SLEEP_TIME_SECONDS, acc, update);
}

#define NONE 0

scylla_blas::matrix<float>&
scylla_blas::routine_scheduler::sgemm(const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                                      const float alpha, const matrix<float> &A,
                                      const matrix<float> &B,
                                      const float beta, scylla_blas::matrix<float> &C) {
    assert_multiplication_compatible(TransA, A, B, TransB, C);
    add_blocks_as_queue_tasks(this->_subtask_queue, C);

    produce_matrix_tasks<float>(proto::SGEMM, A.id, TransA, alpha, B.id, TransB, beta, C.id);

    return C;
}

scylla_blas::matrix<double>&
scylla_blas::routine_scheduler::dgemm(const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                                      const double alpha, const matrix<double> &A,
                                      const matrix<double> &B, const double beta, scylla_blas::matrix<double> &C) {
    assert_multiplication_compatible(TransA, A, B, TransB, C);
    add_blocks_as_queue_tasks(this->_subtask_queue, C);

    produce_matrix_tasks<double>(proto::DGEMM, A.id, TransA, alpha, B.id, TransB, beta, C.id);

    return C;
}

scylla_blas::matrix<float>&
scylla_blas::routine_scheduler::ssyrk(__attribute__((unused)) const enum UPLO Uplo,
                                      const enum TRANSPOSE TransA, const float alpha, const matrix<float> &A,
                                      const float beta, matrix<float> &C) {
    assert_multiplication_compatible(TransA, A, A, anti_trans(TransA), C);
    add_blocks_as_queue_tasks(this->_subtask_queue, C);

    produce_matrix_tasks<float>(proto::SSYRK, A.id, TransA, alpha, NONE, NoTrans, beta, C.id);

    return C;
}

scylla_blas::matrix<double>&
scylla_blas::routine_scheduler::dsyrk(__attribute__((unused)) const enum UPLO Uplo,
                                      const enum TRANSPOSE TransA, const double alpha, const matrix<double> &A,
                                      const double beta, matrix<double> &C) {
    assert_multiplication_compatible(TransA, A, A, anti_trans(TransA), C);
    add_blocks_as_queue_tasks(this->_subtask_queue, C);

    produce_matrix_tasks<float>(proto::DSYRK, A.id, TransA, alpha, NONE, NoTrans, beta, C.id);

    return C;
}

scylla_blas::matrix<float>&
scylla_blas::routine_scheduler::ssyr2k(__attribute__((unused)) const enum UPLO Uplo,
                                      const enum TRANSPOSE Trans, const float alpha, const matrix<float> &A,
                                      const float beta, const matrix<float> &B, matrix<float> &C) {
    assert_multiplication_compatible(Trans, A, B, anti_trans(Trans), C);
    assert_multiplication_compatible(anti_trans(Trans), A, B, Trans, C);
    add_blocks_as_queue_tasks(this->_subtask_queue, C);

    produce_matrix_tasks<float>(proto::SSYR2K, A.id, Trans, alpha, B.id, NoTrans, beta, C.id);

    return C;
}

scylla_blas::matrix<double>&
scylla_blas::routine_scheduler::dsyr2k(__attribute__((unused)) const enum UPLO Uplo,
                                      const enum TRANSPOSE TransA, const double alpha, const matrix<double> &A,
                                      const double beta, const matrix<double> &B, matrix<double> &C) {
    assert_multiplication_compatible(Trans, A, B, anti_trans(Trans), C);
    assert_multiplication_compatible(anti_trans(Trans), A, B, Trans, C);
    add_blocks_as_queue_tasks(this->_subtask_queue, C);

    produce_matrix_tasks<float>(proto::DSYR2K, A.id, Trans, alpha, B.id, NoTrans, beta, C.id);

    return C;
}