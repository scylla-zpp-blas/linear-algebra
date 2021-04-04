#include  <iostream>

#include "scylla_blas/routines.hh"
#include "scylla_blas/queue/worker_proc.hh"
#include "scylla_blas/utils/utils.hh"

namespace {

void assert_multiplication_compatible(const enum scylla_blas::TRANSPOSE TransA, const scylla_blas::basic_matrix &A,
                                      const scylla_blas::basic_matrix &B, const enum scylla_blas::TRANSPOSE TransB) {
    using namespace scylla_blas;

    index_type dim_a = (TransA == NoTrans ? A.column_count : A.row_count);
    index_type dim_b = (TransB == NoTrans ? B.row_count : B.column_count);

    if (dim_a != dim_b) {
        throw std::runtime_error(
            fmt::format(
                    "Incompatible matrices {} of size {}x{}{} and {} of size {}x{}{}: multiplication impossible!",
                    A.id, A.row_count, A.column_count, (TransA == NoTrans ? "" : " (transposed)"),
                    B.id, B.row_count, B.column_count, (TransB == NoTrans ? "" : " (transposed)")
            )
        );
    }
}

}

template<class T>
scylla_blas::matrix<T>
scylla_blas::routine_scheduler::gemm(const enum scylla_blas::ORDER Order,
                                     const enum scylla_blas::TRANSPOSE TransA,
                                     const enum scylla_blas::TRANSPOSE TransB,
                                     const T alpha, const scylla_blas::matrix<T> &A,
                                     const scylla_blas::matrix<T> &B, const T beta) {
    assert_multiplication_compatible(TransA, A, B, TransB);

    int64_t base_id = get_timestamp();
    int64_t C_id = base_id;

    matrix<T> C = matrix<T>::init_and_return(this->_session, C_id, A.row_count, B.column_count);

    std::cerr << "Preparing multiplication task..." << std::endl;
    for (index_type i = 1; i <= C.get_blocks_height(); i++) {
        for (index_type j = 1; j <= C.get_blocks_width(); j++) {
            this->_subtask_queue.produce({
                 .type = proto::NONE,
                 .coord {
                     .block_row = i,
                     .block_column = j
                 }});
        }
    }

    std::cerr << "Ordering multiplication to the workers" << std::endl;
    std::vector<int64_t> task_ids;
    for (index_type i = 0; i < LIMIT_WORKER_CONCURRENCY; i++) {
        task_ids.push_back(_main_worker_queue.produce({
             .type = worker::get_task_type_for_procedure(worker::gemm<T>),
             .blas_binary {
                 .task_queue_id = this->_subtask_queue_id,
                 .A_id = A.id,
                 .B_id = B.id,
                 .C_id = C_id
             }}));
    }

    std::cerr << "Waiting for workers to complete the order..." << std::endl;
    for (int64_t id : task_ids) {
        while (!_main_worker_queue.is_finished(id)) {
            scylla_blas::wait_seconds(WORKER_SLEEP_TIME_SECONDS);
        }
    }

    return C;
}

/* TODO: Can we use define? Or in any other way avoid these boilerplatey signatures? */
scylla_blas::matrix<float>
scylla_blas::routine_scheduler::sgemm(const enum ORDER Order, const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                                      const float alpha, const matrix<float> &A,
                                      const matrix<float> &B, const float beta) {
    return gemm<float>(Order, TransA, TransB, alpha, A, B, beta);
}

scylla_blas::matrix<double>
scylla_blas::routine_scheduler::dgemm(const enum ORDER Order, const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                                      const double alpha, const matrix<double> &A,
                                      const matrix<double> &B, const double beta) {
    return gemm<double>(Order, TransA, TransB, alpha, A, B, beta);
}
