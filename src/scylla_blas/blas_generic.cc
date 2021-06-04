#include "scylla_blas/routines.hh"

template<>
float scylla_blas::routine_scheduler::produce_and_wait(scylla_queue &queue, const proto::task &task,
                                                       index_t cnt, int64_t sleep_time,
                                                       float acc, updater<float> update) {
    std::vector<scylla_blas::scylla_queue::task> tasks(cnt, task);
    int64_t task_id = queue.produce(tasks);
    LogDebug("Scheduled tasks {}-{}", task_id, task_id + cnt - 1);
    for (int64_t id = task_id; id < task_id + cnt; id++) {
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
    std::vector<scylla_blas::scylla_queue::task> tasks(cnt, task);
    int64_t task_id = queue.produce(tasks);
    LogDebug("Scheduled tasks {}-{}", task_id, task_id + cnt - 1);
    for (int64_t id = task_id; id < task_id + cnt; id++) {
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
