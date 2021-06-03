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

        LogDebug("New secondary task obtained; id = {}", opt.value().first);
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

        Y_segm = Y_segm + (X_segm * task_details.alpha);

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

        acc += X_segm.template dot_prod<U>(Y_segm);
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
        acc += X.get_segment(subtask.index).mod2();
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
std::pair<scylla_blas::index_t, T> iamax(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    scylla_blas::scylla_queue task_queue = scylla_blas::scylla_queue(session, task_details.task_queue_id);
    scylla_blas::vector<T> X(session, task_details.X_id);
    T max_abs = 0;
    scylla_blas::index_t imax = 0;

    auto iamax_segment = [&X, &max_abs, &imax] (scylla_blas::proto::task &subtask) {
        scylla_blas::index_t offset = X.get_segment_offset(subtask.index);

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

/* LEVEL 2 */
template<class T>
void gemv(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    using namespace scylla_blas;

    matrix<T> A(session, task_details.A_id);
    vector<T> X(session, task_details.X_id);
    vector<T> Y(session, task_details.Y_id);
    scylla_queue task_queue = scylla_queue(session, task_details.task_queue_id);

    auto compute_result_segment = [&A, &X, &Y, &task_details] (proto::task &subtask) {
        index_t prod_segments = A.get_blocks_width(task_details.TransA);
        vector_segment result = Y.get_segment(subtask.index) * task_details.beta;

        for (index_t i = 1; i <= prod_segments; i++) {
            matrix_block block_A = A.get_block(subtask.index, i, task_details.TransA);
            vector_segment segment_X = X.get_segment(i);

            result += block_A.mult_vect(segment_X) * task_details.alpha;
        }

        Y.update_segment(subtask.index, result);
    };

    consume_tasks(task_queue, compute_result_segment);
}

template<class T>
void gbmv(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    using namespace scylla_blas;

    matrix<T> A(session, task_details.A_id);
    vector<T> X(session, task_details.X_id);
    vector<T> Y(session, task_details.Y_id);
    scylla_queue task_queue = scylla_queue(session, task_details.task_queue_id);

    auto compute_result_segment = [&A, &X, &Y, &task_details] (proto::task &subtask) {
        auto [start, end] = A.get_banded_block_limits_for_row(subtask.index, task_details.KL, task_details.KU, task_details.TransA);
        vector_segment result = Y.get_segment(subtask.index) * task_details.beta;

        for (index_t i = start; i <= end; i++) {
            matrix_block block_A = A.get_block(subtask.index, i, task_details.TransA);
            vector_segment segment_X = X.get_segment(subtask.index);

            result += block_A.mult_vect(segment_X) * task_details.alpha;
        }

        Y.update_segment(subtask.index, result);
    };

    consume_tasks(task_queue, compute_result_segment);
}

template<class T>
std::pair<T, T> tsv_generic(const std::shared_ptr<scmd::session> &session,
                            auto &task_details, auto get_right_limit_for_row) {
    /* Based on Jacobi method and Gauss-Seidel method – no difference between the two
     * as the matrix is triangular, i.e. L = 0 (or U = 0 if it is lower-triangular).
     */
    using namespace scylla_blas;

    matrix<T> A(session, task_details.A_id);
    vector<T> b(session, task_details.X_id);
    vector<T> X(session, task_details.Y_id);
    scylla_queue task_queue = scylla_queue(session, task_details.task_queue_id);

    T diff = 0, total = 0;

    auto compute_result_segment = [&A, &b, &X, &diff, &total, &task_details, &get_right_limit_for_row] (proto::task &subtask) {
        index_t limit = get_right_limit_for_row(A, subtask.index);
        vector_segment sum = vector_segment<T>(); // stores the computed segment of Ux_n for given row

        for (index_t i = subtask.index + 1; i <= limit; i++) {
            /* For non-diagonals: a straightforward matrix * vector computation gives Ux_n */
            matrix_block block_A = A.get_block(subtask.index, i, task_details.TransA);
            vector_segment segment_X = X.get_segment(i);

            sum += block_A.mult_vect(segment_X);
        }

        /* The block on the diagonal has to be processed separately by splitting into D and U */
        vector_segment old = X.get_segment(subtask.index);
        matrix_block block_A = A.get_block(subtask.index, subtask.index, task_details.TransA);
        std::vector<matrix_value<T>> anti_diag, rest;
        for (auto &val : block_A.get_values_raw()) {
            if (val.row_index == val.col_index) {
                anti_diag.emplace_back(val.row_index, val.col_index, 1.0 / val.value);
            } else {
                rest.emplace_back(val);
            }
        }

        sum += matrix_block(rest).mult_vect(old);

        /* x_{n+1} = D^(-1)(b - Ux_n) */
        vector_segment result_segment = b.get_segment(subtask.index) + sum * (-1);
        result_segment = matrix_block(anti_diag).mult_vect(result_segment);

        /* Get total and error values for total error evaluation in scheduler */
        total += result_segment.nrminf();
        diff += (result_segment + (old * -1)).nrminf();

        X.update_segment(subtask.index, result_segment);
    };

    consume_tasks(task_queue, compute_result_segment);
    return {diff, total};
}

template<class T>
std::pair<T, T> trsv(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    return tsv_generic<T>(session, task_details,
                          [&task_details](const scylla_blas::matrix<T> &A, scylla_blas::index_t row) {
                              return A.get_blocks_width(task_details.TransA);
                          });
}

template<class T>
std::pair<T, T> tbsv(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    return tsv_generic<T>(session, task_details,
                          [&task_details](const scylla_blas::matrix<T> &A, scylla_blas::index_t row) {
                              return A.get_banded_block_limits_for_row(row, 0, task_details.KU, task_details.TransA).second;
                          });
}

template<class T>
void ger(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    using namespace scylla_blas;

    vector<T> X(session, task_details.X_id);
    vector<T> Y(session, task_details.Y_id);
    matrix<T> A(session, task_details.A_id);
    scylla_queue task_queue = scylla_queue(session, task_details.task_queue_id);

    auto compute_product_block = [&X, &Y, &A, &task_details] (proto::task &subtask) {
        auto [row, column] = subtask.coord;

        matrix_block result_block = A.get_block(row, column);
        vector_segment seg_X = X.get_segment(row);
        vector_segment seg_Y = Y.get_segment(column);

        matrix_block computed = matrix_block<T>::outer_prod(seg_X, seg_Y) * task_details.alpha + result_block;

        A.insert_block(row, column, computed);
    };

    consume_tasks(task_queue, compute_product_block);
}

/* LEVEL 3 */
template<class T>
void gemm(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    using namespace scylla_blas;

    matrix<T> A(session, task_details.A_id);
    matrix<T> B(session, task_details.B_id);
    matrix<T> C(session, task_details.C_id);
    scylla_queue task_queue = scylla_queue(session, task_details.task_queue_id);

    auto compute_result_block = [&A, &B, &C, &task_details] (proto::task &subtask) {
        auto [row, column] = subtask.coord;

        index_t blocks_to_multiply = A.get_blocks_width(task_details.TransA);
        matrix_block result_block = C.get_block(row, column) * task_details.beta;

        for (index_t i = 1; i <= blocks_to_multiply; i++) {
            matrix_block block_A = A.get_block(row, i, task_details.TransA);
            matrix_block block_B = B.get_block(i, column, task_details.TransB);

            result_block += block_A * block_B * task_details.alpha;
        }

        C.insert_block(row, column, result_block);
    };

    consume_tasks(task_queue, compute_result_block);
}

template<class T>
void syrk_generic(const std::shared_ptr<scmd::session> &session,
                  const scylla_blas::matrix<T> &A,
                  const scylla_blas::matrix<T> &B,
                  scylla_blas::matrix<T> &C,
                  double scaling, auto &task_details) {

    using namespace scylla_blas;

    scylla_queue task_queue = scylla_queue(session, task_details.task_queue_id);

    auto compute_result_block = [&A, &B, &C, scaling, &task_details] (proto::task &subtask) {
        auto [row, column] = subtask.coord;

        index_t blocks_to_multiply = A.get_blocks_width(task_details.TransA);
        matrix_block result_block = C.get_block(row, column) * task_details.beta;

        for (index_t i = 1; i <= blocks_to_multiply; i++) {
            matrix_block block_left_A = A.get_block(row, i, task_details.TransA);
            matrix_block block_left_B = B.get_block(i, column, anti_trans(task_details.TransA));

            matrix_block block_right_B = B.get_block(row, i, task_details.TransA);
            matrix_block block_right_A = A.get_block(i, column, anti_trans(task_details.TransA));
            result_block += (block_left_A * block_left_B + block_right_B * block_right_A) * (task_details.alpha * scaling);
        }

        C.insert_block(row, column, result_block);
    };

    consume_tasks(task_queue, compute_result_block);
}

template<class T>
void syrk(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    scylla_blas::matrix<T> A(session, task_details.A_id);
    scylla_blas::matrix<T> C(session, task_details.C_id);

    syrk_generic<T>(session, A, A, C, 0.5, task_details);
}

template<class T>
void syr2k(const std::shared_ptr<scmd::session> &session, auto &task_details) {
    scylla_blas::matrix<T> A(session, task_details.A_id);
    scylla_blas::matrix<T> B(session, task_details.B_id);
    scylla_blas::matrix<T> C(session, task_details.C_id);

    syrk_generic<T>(session, A, B, C, 1, task_details);
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

/* LEVEL 2 */
DEFINE_WORKER_FUNCTION(sgemv, {
    gemv<float>(session, task.mixed_task_float);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(dgemv, {
    gemv<double>(session, task.mixed_task_double);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(sgbmv, {
    gbmv<float>(session, task.mixed_task_float);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(dgbmv, {
    gbmv<double>(session, task.mixed_task_double);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(strsv, {
    auto result = trsv<float>(session, task.mixed_task_float);
    return (proto::response{ .result_float_pair = { .first = result.first, .second = result.second } });
})

DEFINE_WORKER_FUNCTION(dtrsv, {
    auto result = trsv<double>(session, task.mixed_task_double);
    return (proto::response{ .result_double_pair = { .first = result.first, .second = result.second } });
})

DEFINE_WORKER_FUNCTION(stbsv, {
    auto result = tbsv<float>(session, task.mixed_task_float);
    return (proto::response{ .result_float_pair = { .first = result.first, .second = result.second } });
})

DEFINE_WORKER_FUNCTION(dtbsv, {
    auto result = tbsv<double>(session, task.mixed_task_double);
    return (proto::response{ .result_double_pair = { .first = result.first, .second = result.second } });
})

DEFINE_WORKER_FUNCTION(sger, {
    ger<float>(session, task.mixed_task_float);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(dger, {
    ger<double>(session, task.mixed_task_double);
    return std::nullopt;
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

DEFINE_WORKER_FUNCTION(ssyrk, {
    syrk<float>(session, task.matrix_task_float);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(dsyrk, {
    syrk<double>(session, task.matrix_task_double);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(ssyr2k, {
    syr2k<float>(session, task.matrix_task_float);
    return std::nullopt;
})

DEFINE_WORKER_FUNCTION(dsyr2k, {
    syr2k<double>(session, task.matrix_task_double);
    return std::nullopt;
})
#undef DEFINE_WORKER_FUNCTION