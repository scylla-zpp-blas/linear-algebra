#include "scylla_blas/queue/worker_proc.hh"

namespace {

void consume_tasks(scylla_blas::scylla_queue &task_queue,
                   std::function<void(scylla_blas::proto::task&)> consume) {
    using namespace scylla_blas;

    while (true) {
        std::optional<std::pair<int64_t, proto::task>> opt = task_queue.consume();
        if (!opt.has_value()) {
            // The task queue is empty – nothing left to do.
            break;
        }

        std::cerr << "New secondary task obtained; id = " << opt.value().first << std::endl;
        consume(opt.value().second);
    }
}

/* LEVEL 1 */
template<class T>
void swap(const std::shared_ptr<scmd::session> &session, const auto &task_details) {
    scylla_blas::scylla_queue task_queue = scylla_blas::scylla_queue(session, task_details.task_queue_id);
    scylla_blas::vector<T> X(session, task_details.X_id);
    scylla_blas::vector<T> Y(session, task_details.Y_id);

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
void scal(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    scylla_blas::scylla_queue task_queue = scylla_blas::scylla_queue(session, task_details.task_queue_id);
    scylla_blas::vector<T> X(session, task_details.X_id);

    auto scal_segment = [&task_details, &X] (scylla_blas::proto::task &subtask) {
        scylla_blas::vector_segment<T> X_segm = X.get_segment(subtask.index);
        for (auto &entry : X_segm)
            entry.value *= task_details.alpha;

        X.clear_segment(subtask.index);
        X.update_segment(subtask.index, X_segm);
    };

    consume_tasks(task_queue, scal_segment);
}

template<class T>
void copy(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    scylla_blas::scylla_queue task_queue = scylla_blas::scylla_queue(session, task_details.task_queue_id);
    scylla_blas::vector<T> X(session, task_details.X_id);
    scylla_blas::vector<T> Y(session, task_details.Y_id);

    auto copy_segment = [&X, &Y] (scylla_blas::proto::task &subtask) {
        Y.clear_segment(subtask.index);
        Y.update_segment(subtask.index, X.get_segment(subtask.index));
    };

    consume_tasks(task_queue, copy_segment);
}

template<class T>
void axpy(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    scylla_blas::scylla_queue task_queue = scylla_blas::scylla_queue(session, task_details.task_queue_id);
    scylla_blas::vector<T> X(session, task_details.X_id);
    scylla_blas::vector<T> Y(session, task_details.Y_id);

    auto axpy_segment = [&task_details, &X, &Y] (scylla_blas::proto::task &subtask) {
        auto X_segm = X.get_segment(subtask.index);
        auto Y_segm = Y.get_segment(subtask.index);

        auto itx = X_segm.begin();
        auto ity = Y_segm.begin();

        while(itx != X_segm.end() && ity != Y_segm.end()) {
            if (itx->index < ity->index) itx++;
            else if (itx->index > ity->index) ity++;
            else {
                ity->value += task_details.alpha * itx->value;
                itx++;
                ity++;
            }
        }

        Y.clear_segment(subtask.index);
        Y.update_segment(subtask.index, Y_segm);
    };

    consume_tasks(task_queue, axpy_segment);
}

/* T -> source type;
 * U -> precision
 */
template<class T, class U>
U dot(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    scylla_blas::scylla_queue task_queue = scylla_blas::scylla_queue(session, task_details.task_queue_id);
    scylla_blas::vector<T> X(session, task_details.X_id);
    scylla_blas::vector<T> Y(session, task_details.Y_id);
    U acc = 0;

    auto dot_segment = [&X, &Y, &acc] (scylla_blas::proto::task &subtask) {
        auto X_segm = X.get_segment(subtask.index);
        auto Y_segm = Y.get_segment(subtask.index);

        auto itx = X_segm.begin();
        auto ity = Y_segm.begin();

        while(itx != X_segm.end() && ity != Y_segm.end()) {
            if (itx->index < ity->index) itx++;
            else if (itx->index > ity->index) ity++;
            else {
                acc += U(itx->value) * U(ity->value);
                itx++;
                ity++;
            }
        }
    };

    consume_tasks(task_queue, dot_segment);
    return acc;
}

template<class T>
T nrm2(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    scylla_blas::scylla_queue task_queue = scylla_blas::scylla_queue(session, task_details.task_queue_id);
    scylla_blas::vector<T> X(session, task_details.X_id);
    T acc = 0;

    auto nrm2_segment = [&X, &acc] (scylla_blas::proto::task &subtask) {
        for (auto &val : X.get_segment(subtask.index)) {
            acc += val.value * val.value;
        }
    };

    consume_tasks(task_queue, nrm2_segment);
    return acc;
}

template<class T>
T asum(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    scylla_blas::scylla_queue task_queue = scylla_blas::scylla_queue(session, task_details.task_queue_id);
    scylla_blas::vector<T> X(session, task_details.X_id);
    T acc = 0;

    auto nrm2_segment = [&X, &acc] (scylla_blas::proto::task &subtask) {
        for (auto &val : X.get_segment(subtask.index)) {
            acc += std::abs(val.value);
        }
    };

    consume_tasks(task_queue, nrm2_segment);
    return acc;
}

template<class T>
std::pair<scylla_blas::index_type, T> iamax(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    scylla_blas::scylla_queue task_queue = scylla_blas::scylla_queue(session, task_details.task_queue_id);
    scylla_blas::vector<T> X(session, task_details.X_id);
    T max_abs = 0;
    scylla_blas::index_type imax = 0;

    auto iamax_segment = [&X, &max_abs, &imax] (scylla_blas::proto::task &subtask) {
        scylla_blas::index_type offset = X.get_segment_offset(subtask.index);

        for (auto &val : X.get_segment(subtask.index)) {
            if (std::abs(val.value) > max_abs) {
                max_abs = std::abs(val.value);
                imax = val.index + offset;
            } else if (std::abs(val.value) == max_abs) {
                imax = std::min(imax, val.index + offset);
            }
        }
    };

    consume_tasks(task_queue, iamax_segment);
    return { imax, max_abs };
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

#define DEFINE_WORKER_FUNCTION(function_name, function_body) \
std::optional<scylla_blas::proto::response> \
scylla_blas::worker::function_name(const std::shared_ptr<scmd::session> &session, const scylla_blas::proto::task &task) function_body

DEFINE_WORKER_FUNCTION(sswap, {
    swap<float>(session, task.vector_task_float);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(sscal, {
    scal<float>(session, task.vector_task_float);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(scopy, {
    copy<float>(session, task.vector_task_float);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(saxpy, {
    axpy<float>(session, task.vector_task_float);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(sdot, {
    float result = (dot<float, float>(session, task.vector_task_float));
    return proto::response{ .result_float = result };
})

DEFINE_WORKER_FUNCTION(sdsdot, {
    double result = (dot<float, double>(session, task.vector_task_float));
    return proto::response{ .result_double = result };
})

DEFINE_WORKER_FUNCTION(snrm2, {
    float result = nrm2<float>(session, task.vector_task_float);
    return proto::response{ .result_float = result };
})

DEFINE_WORKER_FUNCTION(sasum, {
    float result = asum<float>(session, task.vector_task_float);
    return proto::response{ .result_float = result };
})

DEFINE_WORKER_FUNCTION(isamax, {
    auto result = iamax<float>(session, task.vector_task_float);
    return (proto::response{ .result_max_float_index { .index = result.first, .value = result.second }});
})

DEFINE_WORKER_FUNCTION(dswap, {
    swap<double>(session, task.vector_task_double);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(dscal, {
    scal<double>(session, task.vector_task_double);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(dcopy, {
    copy<double>(session, task.vector_task_double);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(daxpy, {
    axpy<double>(session, task.vector_task_double);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(ddot, {
    double result = (dot<double, double>(session, task.vector_task_double));
    return proto::response{ .result_double = result };
})

DEFINE_WORKER_FUNCTION(dsdot, {
    double result = (dot<float, double>(session, task.vector_task_double));
    return proto::response{ .result_double = result };
})

DEFINE_WORKER_FUNCTION(dnrm2, {
    double result = (nrm2<double>(session, task.vector_task_double));
    return proto::response{ .result_double = result };
})

DEFINE_WORKER_FUNCTION(dasum, {
    double result = (asum<double>(session, task.vector_task_double));
    return proto::response{ .result_double = result };
})

DEFINE_WORKER_FUNCTION(idamax, {
    auto result = iamax<double>(session, task.vector_task_double);
    return (proto::response{ .result_max_double_index { .index = result.first, .value = result.second }});
})

/* LEVEL 3 */

DEFINE_WORKER_FUNCTION(sgemm, {
    gemm<float>(session, task.matrix_task_float);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(dgemm, {
    gemm<double>(session, task.matrix_task_double);
    return std::nullopt;
})

#undef DEFINE_WORKER_FUNCTION