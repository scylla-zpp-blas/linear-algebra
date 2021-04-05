#pragma once

#include <algorithm>
#include <vector>

#include <scmd.hh>

#include "proto.hh"
#include "scylla_queue.hh"

#include "scylla_blas/matrix.hh"

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

template<class T>
void gemm(const std::shared_ptr<scmd::session> &session,
          int64_t task_queue_id, scylla_blas::TRANSPOSE TransA, scylla_blas::TRANSPOSE TransB,
          T alpha, int64_t A_id, int64_t B_id, T beta, int64_t C_id) {
    using namespace scylla_blas;

    matrix<T> A(session, A_id);
    matrix<T> B(session, B_id);
    matrix<T> C(session, C_id);
    scylla_queue task_queue = scylla_queue(session, task_queue_id);

    auto compute_result_block = [TransA, TransB, alpha, &A, &B, beta, &C] (proto::task &subtask) mutable {
        auto [row, column] = subtask.coord;

        index_type blocks_to_multiply = A.get_blocks_width(TransA);
        matrix_block<T> result_block = C.get_block(row, column) * beta;

        for (index_type i = 1; i <= blocks_to_multiply; i++) {
            matrix_block block_A = A.get_block(row, i, TransA);
            matrix_block block_B = B.get_block(i, column, TransB);

            result_block += block_A * block_B * alpha;
        }

        C.insert_block(row, column, result_block);
    };

    consume_tasks(task_queue, compute_result_block);
}

}

namespace scylla_blas::worker {

void sgemm(const std::shared_ptr<scmd::session> &session, const proto::task &task) {
    auto [task_queue_id, TransA, TransB, alpha, A_id, B_id, beta, C_id] = task.sgemm;

    gemm<float>(session, task_queue_id, TransA, TransB, alpha, A_id, B_id, beta, C_id);
}

void dgemm(const std::shared_ptr<scmd::session> &session, const proto::task &task) {
    auto [task_queue_id, TransA, TransB, alpha, A_id, B_id, beta, C_id] = task.dgemm;

    gemm<double>(session, task_queue_id, TransA, TransB, alpha, A_id, B_id, beta, C_id);
}

using procedure_t = void(const std::shared_ptr<scmd::session>&, const proto::task&);

constexpr std::array<std::pair<proto::task_type, const procedure_t&>, 2> task_to_procedure = {{
        { proto::SGEMM, sgemm },
        { proto::DGEMM, dgemm }
}};

constexpr procedure_t& get_procedure_for_task(proto::task t) {
    auto pred = [=](auto &val){ return val.first == t.type; };

    auto it = std::find_if(task_to_procedure.begin(), task_to_procedure.end(), pred);

    if (it == task_to_procedure.end()) {
        throw std::runtime_error("Operation type " + std::to_string(t.type) +  " not implemented!");
    }

    return it->second;
}

constexpr proto::task_type get_task_type_for_procedure(const procedure_t &proc) {
    auto pred = [=](auto &val){ return val.second == proc; };

    auto it = std::find_if(task_to_procedure.begin(), task_to_procedure.end(), pred);

    if (it == task_to_procedure.end()) {
        throw std::runtime_error("Procedure not implemented!");
    }

    return it->first;
}

}