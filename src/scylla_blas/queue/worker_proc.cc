#include "scylla_blas/queue/worker_proc.hh"

namespace {

void consume_tasks(scylla_blas::scylla_queue &task_queue,
                   std::function<void(scylla_blas::proto::task&)> consume) {
    using namespace scylla_blas;

    while (true) {
        try {
            auto [task_id, binary_subtask] = task_queue.consume();
            std::cerr << "New secondary task obtained; id = " << task_id << std::endl;

            consume(binary_subtask);
        } catch (const empty_container_error& e) {
            // The task queue is empty – nothing left to do.
            break;
        }
    }
}

/* LEVEL 1 */
template<class T>
void swap(const std::shared_ptr<scmd::session> &session, const auto &task_details) {
    scylla_blas::vector<T> X(session, task_details.X_id);
    scylla_blas::vector<T> Y(session, task_details.Y_id);
    scylla_blas::scylla_queue task_queue = scylla_blas::scylla_queue(session, task_details.task_queue_id);

    auto swap_segment = [&X, &Y] (scylla_blas::proto::task &subtask) {
        scylla_blas::vector_segment<T> X_segm = X.get_segment(subtask.index);
        scylla_blas::vector_segment<T> Y_segm = Y.get_segment(subtask.index);

        X.clear_segment(subtask.index);
        Y.clear_segment(subtask.index);

        X.update_segment(subtask.index, Y_segm);
        Y.update_segment(subtask.index, X_segm);
    };

    consume_tasks(task_queue, swap_segment);
}

template<class T>
void scal(const std::shared_ptr<scmd::session> &session, auto &task_details, const T alpha) {
    scylla_blas::vector<T> X(session, task_details.X_id);
    scylla_blas::scylla_queue task_queue = scylla_blas::scylla_queue(session, task_details.task_queue_id);

    auto swap_segment = [alpha, &X] (scylla_blas::proto::task &subtask) {
        scylla_blas::vector_segment<T> X_segm = X.get_segment(subtask.index);
        for (auto &entry : X_segm)
            entry.value *= alpha;

        X.clear_segment(subtask.index);
        X.update_segment(subtask.index, X_segm);
    };

    consume_tasks(task_queue, swap_segment);
}

template<class T>
void copy(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    scylla_blas::vector<T> X(session, task_details.X_id);
    scylla_blas::vector<T> Y(session, task_details.Y_id);
    scylla_blas::scylla_queue task_queue = scylla_blas::scylla_queue(session, task_details.task_queue_id);

    auto swap_segment = [&X, &Y] (scylla_blas::proto::task &subtask) {
        Y.clear_segment(subtask.index);
        Y.update_segment(subtask.index, X.get_segment(subtask.index));
    };

    consume_tasks(task_queue, swap_segment);
}

template<class T>
void axpy(const std::shared_ptr<scmd::session> &session, auto &task_details, const T alpha) {
    scylla_blas::vector<T> X(session, task_details.X_id);
    scylla_blas::vector<T> Y(session, task_details.Y_id);
    scylla_blas::scylla_queue task_queue = scylla_blas::scylla_queue(session, task_details.task_queue_id);

    auto swap_segment = [alpha, &X, &Y] (scylla_blas::proto::task &subtask) {
        auto X_segm = X.get_segment(subtask.index);
        auto Y_segm = Y.get_segment(subtask.index);

        auto itx = X_segm.begin();
        auto ity = Y_segm.begin();

        while(itx != X_segm.end() && ity != Y_segm.end()) {
            if (itx->index < ity->index) itx++;
            else if (itx->index > ity->index) ity++;
            else {
                ity->value += alpha * itx->value;
                itx++;
                ity++;
            }
        }

        Y.clear_segment(subtask.index);
        Y.update_segment(subtask.index, Y_segm);
    };

    consume_tasks(task_queue, swap_segment);
}

/* LEVEL 3 */
template<class T>
void gemm(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    using namespace scylla_blas;

    matrix<T> A(session, task_details.A_id);
    matrix<T> B(session, task_details.B_id);
    matrix<T> C(session, task_details.C_id);
    scylla_queue task_queue = scylla_queue(session, task_details.task_queue_id);

    auto compute_result_block = [&A, &B, &C, &task_details] (proto::task &subtask) mutable {
        auto [row, column] = subtask.coord;

        index_type blocks_to_multiply = A.get_blocks_width(task_details.TransA);
        matrix_block<T> result_block = C.get_block(row, column) * task_details.beta;

        for (index_type i = 1; i <= blocks_to_multiply; i++) {
            matrix_block block_A = A.get_block(row, i, task_details.TransA);
            matrix_block block_B = B.get_block(i, column, task_details.TransB);

            result_block += block_A * block_B * task_details.alpha;
        }

        C.insert_block(row, column, result_block);
    };

    consume_tasks(task_queue, compute_result_block);
}

}

void scylla_blas::worker::sswap(const std::shared_ptr<scmd::session> &session, const proto::task &task) {
    swap<float>(session, task.vector_task);
}

void scylla_blas::worker::sscal(const std::shared_ptr<scmd::session> &session, const proto::task &task) {
    scal<float>(session, task.vector_task, task.vector_task.salpha);
}

void scylla_blas::worker::scopy(const std::shared_ptr<scmd::session> &session, const proto::task &task) {
    copy<float>(session, task.vector_task);
}

void scylla_blas::worker::saxpy(const std::shared_ptr<scmd::session> &session, const proto::task &task) {
    axpy<float>(session, task.vector_task, task.vector_task.salpha);
}

void scylla_blas::worker::dswap(const std::shared_ptr<scmd::session> &session, const proto::task &task) {
    swap<double>(session, task.vector_task);
}

void scylla_blas::worker::dscal(const std::shared_ptr<scmd::session> &session, const proto::task &task) {
    scal<double>(session, task.vector_task, task.vector_task.dalpha);
}

void scylla_blas::worker::dcopy(const std::shared_ptr<scmd::session> &session, const proto::task &task) {
    copy<double>(session, task.vector_task);
}

void scylla_blas::worker::daxpy(const std::shared_ptr<scmd::session> &session, const proto::task &task) {
    axpy<double>(session, task.vector_task, task.vector_task.dalpha);
}

/* LEVEL 3 */

void scylla_blas::worker::sgemm(const std::shared_ptr<scmd::session> &session, const proto::task &task) {
    gemm<float>(session, task.sgemm);
}

void scylla_blas::worker::dgemm(const std::shared_ptr<scmd::session> &session, const proto::task &task) {
    gemm<double>(session, task.dgemm);
}