#pragma once

#include <algorithm>
#include <vector>

#include <scmd.hh>

#include "proto.hh"
#include "scylla_queue.hh"

#include "scylla_blas/matrix.hh"

namespace scylla_blas {

template<class T>
using binary_consumer_t = void(const proto::task&, const matrix<T>&, const matrix<T>&, matrix<T>&);

template<class T>
void compute_product_block(const proto::task &task_info,
                           const matrix<T> &A, const matrix<T> &B, matrix<T> &C) {
    auto [row, column] = task_info.coord;
    index_type blocks_to_multiply = A.get_blocks_width();
    matrix_block<T> result_block({}, C.id, row, column);

    for (index_type i = 1; i <= blocks_to_multiply; i++) {
        matrix_block block_A = A.get_block(row, i);
        matrix_block block_B = B.get_block(i, column);

        result_block += block_A * block_B;
    }

    C.update_block(row, column, result_block);
}

template<class T>
void consume_binary(const std::shared_ptr<scmd::session> &session,
                    const proto::task &task,
                    binary_consumer_t<T>* consume) {
    auto [task_queue_id, A_id, B_id, C_id] = task.blas_binary;

    matrix<T> A(session, A_id);
    matrix<T> B(session, B_id);
    matrix<T> C(session, C_id);
    auto task_queue = scylla_queue(session, task_queue_id);
    std::cerr << "Linked to queue " << task_queue_id << std::endl;

    while (true) {
        try {
            auto [task_id, binary_subtask] = task_queue.consume();
            std::cerr << "New secondary task obtained; id = " << task_id << std::endl;

            consume(binary_subtask, A, B, C);
        } catch (const empty_container_error& e) {
            // The task queue is empty – nothing left to do.
            break;
        }
    }
}

}

namespace scylla_blas::worker {

using procedure_t = void(const std::shared_ptr<scmd::session>&, const proto::task&);

template<class T>
void gemm(const std::shared_ptr<scmd::session> &session, const proto::task &task) {
    consume_binary(session, task, &compute_product_block<T>);
}

constexpr std::array<std::pair<proto::task_type, const procedure_t&>, 2> task_to_procedure = {{
        { proto::SGEMM, gemm<float> },
        { proto::DGEMM, gemm<double> }
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