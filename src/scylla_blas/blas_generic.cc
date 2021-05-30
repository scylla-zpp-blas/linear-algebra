#include "scylla_blas/routines.hh"

template<>
float scylla_blas::routine_scheduler::produce_and_wait(scylla_queue &queue, const proto::task &task,
                                                        index_type cnt, int64_t sleep_time,
                                                        float acc, updater<float> update) {
    std::vector<int64_t> task_ids;
    for (scylla_blas::index_type i = 0; i < cnt; i++) {
        task_ids.push_back(queue.produce(task));
    }

    for (int64_t id : task_ids) {
        while (!queue.is_finished(id)) {
            scylla_blas::wait_seconds(sleep_time);
        }

        auto response = queue.get_response(id);

        if (response.value().type == proto::R_NONE) continue;

        try {
            update(acc, response.value());
        } catch(std::exception &e) {
            std::cerr << "Result update failed: " << e.what() << std::endl;
        }
    }

    return acc;
}

template<>
double scylla_blas::routine_scheduler::produce_and_wait(scylla_queue &queue, const proto::task &task,
                                                        index_type cnt, int64_t sleep_time,
                                                        double acc, updater<double> update) {
    std::vector<int64_t> task_ids;
    for (scylla_blas::index_type i = 0; i < cnt; i++) {
        task_ids.push_back(queue.produce(task));
    }

    for (int64_t id : task_ids) {
        while (!queue.is_finished(id)) {
            scylla_blas::wait_seconds(sleep_time);
        }

        auto response = queue.get_response(id);

        if (response.value().type == proto::R_NONE) continue;

        try {
            update(acc, response.value());
        } catch(std::exception &e) {
            std::cerr << "Result update failed: " << e.what() << std::endl;
        }
    }

    return acc;
}
