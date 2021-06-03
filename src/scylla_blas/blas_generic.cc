#include "scylla_blas/routines.hh"

template<>
float scylla_blas::routine_scheduler::produce_and_wait(scylla_queue &queue, const proto::task &task,
                                                       index_t cnt, int64_t sleep_time,
                                                       float acc, updater<float> update) {
    std::vector<int64_t> task_ids;
    for (scylla_blas::index_t i = 0; i < cnt; i++) {
        task_ids.push_back(queue.produce(task));
    }

    for (int64_t id : task_ids) {
        while (!queue.is_finished(id)) {
            scylla_blas::wait_microseconds(sleep_time);
        }

        auto response = queue.get_response(id);

        if (response.value().type == proto::R_NONE) continue;

        try {
            update(acc, response.value());
        } catch(std::exception &e) {
            LogError("Result update failed: {}", e.what());
        }
    }

    return acc;
}

template<>
double scylla_blas::routine_scheduler::produce_and_wait(scylla_queue &queue, const proto::task &task,
                                                        index_t cnt, int64_t sleep_time,
                                                        double acc, updater<double> update) {
    std::vector<int64_t> task_ids;
    for (scylla_blas::index_t i = 0; i < cnt; i++) {
        task_ids.push_back(queue.produce(task));
    }

    for (int64_t id : task_ids) {
        while (!queue.is_finished(id)) {
            scylla_blas::wait_microseconds(sleep_time);
        }

        auto response = queue.get_response(id);

        if (response.value().type == proto::R_NONE) continue;

        try {
            update(acc, response.value());
        } catch(std::exception &e) {
            LogError("Result update failed: {}", e.what());
        }
    }

    return acc;
}
