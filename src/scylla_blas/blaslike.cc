#include "scylla_blas/routines.hh"

template<class T>
T scylla_blas::routine_scheduler::produce_generation_tasks(const proto::task_type type,
                                                           const id_t structure_id, const double alpha,
                                                           T acc, updater<T> update) {
    std::vector<proto::task> tasks;

    for (const auto &q : this->_subtask_queues) {
        tasks.push_back({
            .type = type,
            .generation_task = {
                .task_queue_id = q.get_id(),
                .structure_id = structure_id,
                .alpha = alpha
            }
        });
    }

    return produce_and_wait(tasks, acc, update);
}

#define NONE 0

scylla_blas::vector<float>&
scylla_blas::routine_scheduler::srvgen(scylla_blas::vector<float> &X) {
    add_segments_as_queue_tasks(X);

    /* TODO: is there a `none_type` that we could provide? */
    produce_generation_tasks<float>(proto::SRVGEN, X.get_id(), NONE);
    return X;
}

scylla_blas::vector<double>&
scylla_blas::routine_scheduler::drvgen(scylla_blas::vector<double> &X) {
    add_segments_as_queue_tasks(X);

    /* TODO: is there a `none_type` that we could provide? */
    produce_generation_tasks<double>(proto::DRVGEN, X.get_id(), NONE);
    return X;
}

scylla_blas::matrix<float>&
scylla_blas::routine_scheduler::srmgen(double alpha, scylla_blas::matrix<float> &A) {
    add_blocks_as_queue_tasks(A);

    /* TODO: is there a `none_type` that we could provide? */
    produce_generation_tasks<float>(proto::SRMGEN, A.get_id(), alpha);
    return A;
}

scylla_blas::matrix<double>&
scylla_blas::routine_scheduler::drmgen(double alpha, scylla_blas::matrix<double> &A) {
    add_blocks_as_queue_tasks(A);

    /* TODO: is there a `none_type` that we could provide? */
    produce_generation_tasks<double>(proto::DRMGEN, A.get_id(), alpha);
    return A;
}