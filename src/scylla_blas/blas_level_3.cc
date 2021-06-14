#include  <iostream>

#include "scylla_blas/routines.hh"
#include "scylla_blas/queue/worker_proc.hh"
#include "scylla_blas/utils/utils.hh"

namespace {

void assert_multiplication_compatible(const enum scylla_blas::TRANSPOSE TransA, const scylla_blas::basic_matrix &A,
                                      const scylla_blas::basic_matrix &B, const enum scylla_blas::TRANSPOSE TransB,
                                      const scylla_blas::basic_matrix &C) {
    using namespace scylla_blas;

    int row_a = A.get_row_count(); int col_a = A.get_column_count();
    if (TransA != NoTrans) std::swap(row_a, col_a);

    int row_b = B.get_row_count(); int col_b = B.get_column_count();
    if (TransB != NoTrans) std::swap(row_b, col_b);

    if (row_b != col_a) {
        throw std::runtime_error(
            fmt::format(
                    "Incompatible matrices {} of size {}x{}{} and {} of size {}x{}{}: multiplication impossible!",
                    A.get_id(), A.get_row_count(), A.get_column_count(), (TransA == NoTrans ? "" : " (transposed)"),
                    B.get_id(), B.get_row_count(), B.get_column_count(), (TransB == NoTrans ? "" : " (transposed)")
            )
        );
    }

    if (row_a != C.get_row_count() || col_b != C.get_column_count()) {
        throw std::runtime_error(
                fmt::format(
                        "Matrix {} of size {}x{} incompatible with multiplication result of matrices sized {}x{} and {}x{}!",
                        C.get_id(), C.get_row_count(), C.get_column_count(), row_a, col_a, row_b, col_b
                )
        );
    }
}

}

template<>
float scylla_blas::routine_scheduler::produce_matrix_tasks(const proto::task_type type,
                                                           const id_t A_id, const enum TRANSPOSE TransA, const float alpha,
                                                           const id_t B_id, const enum TRANSPOSE TransB, const float beta,
                                                           const id_t C_id, float acc, updater<float> update) {
    std::vector<proto::task> tasks;

    for (const auto &q : this->_subtask_queues) {
        tasks.push_back({
            .type = type,
            .matrix_task_float = {
                .task_queue_id = q.get_id(),

                .A_id = A_id,
                .TransA = TransA,
                .alpha = alpha,

                .B_id = B_id,
                .TransB = TransB,
                .beta = beta,

                .C_id = C_id
            }
        });
    }

    return produce_and_wait(tasks, acc, update);
}

template<>
double scylla_blas::routine_scheduler::produce_matrix_tasks(const proto::task_type type,
                                                            const id_t A_id, const enum TRANSPOSE TransA, const double alpha,
                                                            const id_t B_id, const enum TRANSPOSE TransB, const double beta,
                                                            const id_t C_id, double acc, updater<double> update) {
    std::vector<proto::task> tasks;

    for (const auto &q : this->_subtask_queues) {
        tasks.push_back({
            .type = type,
            .matrix_task_double = {
                .task_queue_id = q.get_id(),

                .A_id = A_id,
                .TransA = TransA,
                .alpha = alpha,

                .B_id = B_id,
                .TransB = TransB,
                .beta = beta,

                .C_id = C_id
            }
        });
    }

    return produce_and_wait(tasks, acc, update);
}

#define NONE 0

scylla_blas::matrix<float>&
scylla_blas::routine_scheduler::sgemm(const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                                      const float alpha, const matrix<float> &A,
                                      const matrix<float> &B,
                                      const float beta, scylla_blas::matrix<float> &C) {
    assert_multiplication_compatible(TransA, A, B, TransB, C);
    add_blocks_as_queue_tasks(C);

    produce_matrix_tasks<float>(proto::SGEMM, A.get_id(), TransA, alpha, B.get_id(), TransB, beta, C.get_id());

    return C;
}

scylla_blas::matrix<double>&
scylla_blas::routine_scheduler::dgemm(const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                                      const double alpha, const matrix<double> &A,
                                      const matrix<double> &B, const double beta, scylla_blas::matrix<double> &C) {
    assert_multiplication_compatible(TransA, A, B, TransB, C);
    add_blocks_as_queue_tasks(C);

    produce_matrix_tasks<double>(proto::DGEMM, A.get_id(), TransA, alpha, B.get_id(), TransB, beta, C.get_id());

    return C;
}

scylla_blas::matrix<float>&
scylla_blas::routine_scheduler::ssyrk(__attribute__((unused)) const enum UPLO Uplo,
                                      const enum TRANSPOSE TransA, const float alpha, const matrix<float> &A,
                                      const float beta, matrix<float> &C) {
    assert_multiplication_compatible(TransA, A, A, anti_trans(TransA), C);
    add_blocks_as_queue_tasks(C);

    produce_matrix_tasks<float>(proto::SSYRK, A.get_id(), TransA, alpha, NONE, NoTrans, beta, C.get_id());

    return C;
}

scylla_blas::matrix<double>&
scylla_blas::routine_scheduler::dsyrk(__attribute__((unused)) const enum UPLO Uplo,
                                      const enum TRANSPOSE TransA, const double alpha, const matrix<double> &A,
                                      const double beta, matrix<double> &C) {
    assert_multiplication_compatible(TransA, A, A, anti_trans(TransA), C);
    add_blocks_as_queue_tasks(C);

    produce_matrix_tasks<float>(proto::DSYRK, A.get_id(), TransA, alpha, NONE, NoTrans, beta, C.get_id());

    return C;
}

scylla_blas::matrix<float>&
scylla_blas::routine_scheduler::ssyr2k(__attribute__((unused)) const enum UPLO Uplo,
                                      const enum TRANSPOSE Trans, const float alpha, const matrix<float> &A,
                                      const float beta, const matrix<float> &B, matrix<float> &C) {
    assert_multiplication_compatible(Trans, A, B, anti_trans(Trans), C);
    assert_multiplication_compatible(anti_trans(Trans), A, B, Trans, C);
    add_blocks_as_queue_tasks(C);

    produce_matrix_tasks<float>(proto::SSYR2K, A.get_id(), Trans, alpha, B.get_id(), NoTrans, beta, C.get_id());

    return C;
}

scylla_blas::matrix<double>&
scylla_blas::routine_scheduler::dsyr2k(__attribute__((unused)) const enum UPLO Uplo,
                                      const enum TRANSPOSE TransA, const double alpha, const matrix<double> &A,
                                      const double beta, const matrix<double> &B, matrix<double> &C) {
    assert_multiplication_compatible(Trans, A, B, anti_trans(Trans), C);
    assert_multiplication_compatible(anti_trans(Trans), A, B, Trans, C);
    add_blocks_as_queue_tasks(C);

    produce_matrix_tasks<float>(proto::DSYR2K, A.get_id(), Trans, alpha, B.get_id(), NoTrans, beta, C.get_id());

    return C;
}