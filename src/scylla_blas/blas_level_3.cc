#include  <iostream>

#include "scylla_blas/routines.hh"
#include "scylla_blas/queue/scylla_queue.hh"
#include "scylla_blas/queue/worker_proc.hh"
#include "scylla_blas/utils/utils.hh"

template<class T>
scylla_blas::matrix<T>
scylla_blas::routine_factory::gemm(const enum scylla_blas::ORDER Order,
                                   const enum scylla_blas::TRANSPOSE TransA,
                                   const enum scylla_blas::TRANSPOSE TransB,
                                   const T alpha, const scylla_blas::matrix<T> &A,
                                   const scylla_blas::matrix<T> &B, const T beta) {
    int64_t base_id = scylla_blas::get_timestamp();
    int64_t C_id = base_id;
    int64_t multiplication_queue_id = base_id;

    auto C = scylla_blas::matrix<T>(this->_session, C_id, false);

    scylla_blas::scylla_queue::create_queue(this->_session, multiplication_queue_id, false, true);
    scylla_blas::scylla_queue multiplication_queue(this->_session, multiplication_queue_id);

    std::cerr << "Preparing multiplication task..." << std::endl;

    for (scylla_blas::index_type i = 1; i <= MATRIX_BLOCK_HEIGHT; i++) {
        for (scylla_blas::index_type j = 1; j <= MATRIX_BLOCK_WIDTH; j++) {
            multiplication_queue.produce({
                 .type = scylla_blas::proto::NONE,
                 .coord {
                     .block_row = i,
                     .block_column = j
                 }});
        }
    }

    scylla_blas::scylla_queue main_worker_queue(this->_session, DEFAULT_WORKER_QUEUE_ID);
    std::vector<int64_t> task_ids;

    std::cerr << "Ordering multiplication to the workers" << std::endl;
    for (index_type i = 0; i < LIMIT_WORKER_CONCURRENCY; i++) {
        task_ids.push_back(main_worker_queue.produce({
             .type = scylla_blas::worker::get_task_type_for_procedure(scylla_blas::worker::gemm<T>),
             .blas_binary {
                 .task_queue_id = multiplication_queue_id,
                 .A_id = A.get_id(),
                 .B_id = B.get_id(),
                 .C_id = C_id
             }}));
    }

    std::cerr << "Waiting for workers to complete the order..." << std::endl;
    for (int64_t id : task_ids) {
        while (!main_worker_queue.is_finished(id)) {
            scylla_blas::wait_seconds(WORKER_SLEEP_TIME_SECONDS);
        }
    }

    return C;
}

/* TODO: Can we use define? Or in any other way avoid these boilerplatey signatures? */
scylla_blas::matrix<float>
scylla_blas::routine_factory::sgemm(const enum ORDER Order, const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                                    const float alpha, const matrix<float> &A,
                                    const matrix<float> &B, const float beta) {
    return gemm<float>(Order, TransA, TransB, alpha, A, B, beta);
}

scylla_blas::matrix<double>
scylla_blas::routine_factory::dgemm(const enum ORDER Order, const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                                    const double alpha, const matrix<double> &A,
                                    const matrix<double> &B, const double beta) {
    return gemm<double>(Order, TransA, TransB, alpha, A, B, beta);
}
