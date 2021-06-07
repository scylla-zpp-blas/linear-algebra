#pragma once

/* BASED ON cblas.h */

#include <scmd.hh>

#include "queue/scylla_queue.hh"
#include "utils/scylla_types.hh"
#include "matrix.hh"
#include "vector.hh"

namespace scylla_blas {

/* If we implement routines the C-like way, passing pointers
 * to results as function parameters, remembering scmd::session
 * in routine_factory is likely redundant.
 *
 * However, we will likely prefer to simplify by reducing
 * the number of arguments and returning results the C++ way.
 */
class routine_scheduler {
    using none_type = void*;

    template<class T>
    using updater = std::function<void(T&, const proto::response&)>;

    std::shared_ptr <scmd::session> _session;

    std::vector<scylla_queue> _subtask_queues;
    scylla_queue _main_worker_queue;

    int64_t _current_worker_count;
    int64_t _scheduler_sleep_time;

    /* Produces `cnt` copies of `task`, waits until all of them are completed.
     * Partial results from completion reports are accumulated in `acc`
     * with `update`, and returned in the end.
     *
     * If there is no result to be accumulated, `update` can be a pointer to null.
     * Otherwise, accumulation errors will be reported if `update` is not a valid function.
     */
    template<class T>
    T produce_and_wait(const std::vector<proto::task> &tasks, T acc, updater<T> update) {
        id_t task_id = _main_worker_queue.produce(tasks);
        LogInfo("Scheduled tasks {}-{} to queue {}", task_id, task_id + tasks.size() - 1, _main_worker_queue.get_id());
        for (id_t id = task_id; id < task_id + tasks.size(); id++) {
            while (!_main_worker_queue.is_finished(id)) {
                scylla_blas::wait_microseconds(_scheduler_sleep_time);
            }

            auto response = _main_worker_queue.get_response(id);

            if (response.value().type == proto::R_NONE) continue;

            try {
                update(acc, response.value());
            } catch(std::exception &e) {
                LogError("Result update failed: {}", e.what());
            }
        }

        return acc;
    }

    /* Produces a number of vector-only primary tasks for workers, waits
     * until they are reported to be complete, accumulating result using
     * the 'update' function, provided that there is any.
     */
    template<class T>
    T produce_vector_tasks(const proto::task_type type, const T alpha,
                           const int64_t X_id, const int64_t Y_id,
                           T acc = 0, updater<T> update = nullptr);

    /* Produces a number of matrix-to-wektor primary tasks for workers, waits
     * until they are reported to be complete, accumulating result using
     * the 'update' function, provided that there is any.
     */
    template<class T>
    T produce_mixed_tasks(const proto::task_type type,
                          const index_t KL, const index_t KU,
                          const UPLO Uplo, const DIAG Diag,
                          const int64_t A_id, const TRANSPOSE TransA, const T alpha,
                          const int64_t X_id, const T beta,
                          const int64_t Y_id, T acc = 0, updater<T> update = nullptr);

    /* Produces a number of matrix-only primary tasks for workers, waits
     * until they are reported to be complete, accumulating result using
     * the 'update' function, provided that there is any.
     */
    template<class T>
    T produce_matrix_tasks(const proto::task_type type,
                           const int64_t A_id, const enum TRANSPOSE TransA, const T alpha,
                           const int64_t B_id, const enum TRANSPOSE TransB, const T beta,
                           const int64_t C_id, T acc = 0, updater<T> update = nullptr);

    void produce_tasks_in_queues(std::vector<scylla_queue::task> &tasks) {
        std::vector<std::vector<scylla_queue::task>> split(_current_worker_count);

        for (size_t i = 0; i < tasks.size(); i++)
            split[i % _current_worker_count].emplace_back(tasks[i]);

        for (size_t i = 0; i < _current_worker_count; i++)
            _subtask_queues[i].produce(split[i]);
    }

    template<class T>
    void add_segments_as_queue_tasks(const vector<T> &X) {
        LogInfo("Scheduling subtasks...");
        std::vector<scylla_queue::task> tasks;
        tasks.reserve(X.get_segment_count());
        for (scylla_blas::index_t i = 1; i <= X.get_segment_count(); i++) {
            tasks.push_back({
                    .type = scylla_blas::proto::NONE,
                    .index = i
            });
        }

        produce_tasks_in_queues(tasks);
    }

    template<class T>
    void add_blocks_as_queue_tasks(const matrix<T> &C) {
        LogDebug("Creating block-based subtasks");
        std::vector<scylla_blas::scylla_queue::task> tasks;
        tasks.reserve(C.get_blocks_height() * C.get_blocks_width());
        for (scylla_blas::index_t i = 1; i <= C.get_blocks_height(); i++) {
            for (scylla_blas::index_t j = 1; j <= C.get_blocks_width(); j++) {
                tasks.push_back({
                    .type = scylla_blas::proto::NONE,
                    .coord {
                            .block_row = i,
                            .block_column = j
                    }});
            }
        }
        produce_tasks_in_queues(tasks);
    }

    void prepare_queues(size_t queue_count) {
        if (_subtask_queues.size() > queue_count) {
            _subtask_queues.erase(_subtask_queues.begin() + queue_count, _subtask_queues.end());
            return;
        }

        while (_subtask_queues.size() < queue_count) {
            id_t queue_id = get_timestamp();
            scylla_queue::create_queue(_session, queue_id, false, false);
            _subtask_queues.emplace_back(_session, queue_id);
        }
    }
public:
    /* The queue used for subroutines requested in methods */
    routine_scheduler(const std::shared_ptr <scmd::session> &session) :
            _session(session),
            _subtask_queues(),
            _main_worker_queue(_session, DEFAULT_WORKER_QUEUE_ID),
            _current_worker_count(DEFAULT_WORKER_COUNT),
            _scheduler_sleep_time(DEFAULT_SCHEDULER_SLEEP_TIME_MICROSECONDS)
    {
        prepare_queues(_current_worker_count);
    }

    int64_t get_max_used_workers() {
        return this->_current_worker_count;
    }

    void set_max_used_workers(int64_t new_max_used_workers) {
        this->_current_worker_count = new_max_used_workers;
        prepare_queues(new_max_used_workers);
    }

    int64_t get_scheduler_sleep_time() {
        return this->_scheduler_sleep_time;
    }

    void set_scheduler_sleep_time(int64_t new_scheduler_sleep_time) {
        this->_scheduler_sleep_time = new_scheduler_sleep_time;
    }

// TODO: shouldn't we ignore N, incX, incY? Maybe skip them altogether for the sake of simplification?

/*
* ===========================================================================
* Prototypes for level 1 BLAS functions
* ===========================================================================
*/
    float sdsdot(const float alpha, const vector<float> &X, const vector<float> &Y);
    double dsdot(const vector<float> &X, const vector<float> &Y);
    float sdot(const vector<float> &X, const vector<float> &Y);
    double ddot(const vector<double> &X, const vector<double> &Y);

    float snrm2(const vector<float> &X);
    float sasum(const vector<float> &X);

    double dnrm2(const vector<double> &X);
    double dasum(const vector<double> &X);

    index_t isamax(const vector<float> &X);
    index_t idamax(const vector<double> &X);

/*
* ===========================================================================
* Prototypes for level 1 BLAS routines
* ===========================================================================
*/

    void sswap(vector<float> &X, vector<float> &Y);
    void scopy(const vector<float> &X, vector<float> &Y);
    void saxpy(const float alpha, const vector<float> &X, vector<float> &Y);

    void dswap(vector<double> &X, vector<double> &Y);
    void dcopy(const vector<double> &X, vector<double> &Y);
    void daxpy(const double alpha, const vector<double> &X, vector<double> &Y);

    void srotg(float *a, float *b, float *c, float *s);
    void srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
    void srot(vector<float> &X, vector<float> &Y, const float c, const float s);
    void srotm(vector<float> &X, vector<float> &Y, const float *P);

    void drotg(double *a, double *b, double *c, double *s);
    void drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
    void drot(vector<double> &X, vector<double> &Y, const double c, const double s);
    void drotm(vector<double> &X, vector<double> &Y, const double *P);

    void sscal(const float alpha, vector<float> &X);
    void dscal(const double alpha, vector<double> &X);

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

    vector<float>& sgemv(const enum TRANSPOSE TransA,
                        const float alpha, const matrix<float> &A,
                        const vector<float> &X, const float beta, vector<float> &Y);

    vector<float>& sgbmv(const enum TRANSPOSE TransA,
                        const int KL, const int KU,
                        const float alpha, const matrix<float> &A,
                        const vector<float> &X, const float beta, vector<float> &Y);

    vector<float>& strmv(const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const matrix<float> &A, vector<float> &X);

    vector<float>& stbmv(const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const int K, const matrix<float> &A, vector<float> &X);

    vector<float>& strsv(__attribute__((unused)) const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const matrix<float> &A, vector<float> &X);

    vector<float>& stbsv(__attribute__((unused)) const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const int K, const matrix<float> &A, vector<float> &X);

    vector<double>& dgemv(const enum TRANSPOSE TransA,
                         const double alpha, const matrix<double> &A,
                         const vector<double> &X, const double beta, vector<double> &Y);

    vector<double>& dgbmv(const enum TRANSPOSE TransA,
                         const int KL, const int KU,
                         const double alpha, const matrix<double> &A,
                         const vector<double> &X, const double beta, vector<double> &Y);

    vector<double>& dtrmv(const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const matrix<double> &A, vector<double> &X);

    vector<double>& dtbmv(const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const int K, const matrix<double> &A, vector<double> &X);

    vector<double>& dtrsv(__attribute__((unused)) const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const matrix<double> &A, vector<double> &X);

    vector<double>& dtbsv(__attribute__((unused)) const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const int K, const matrix<double> &A, vector<double> &X);

    /* Routines with S and D prefixes only */
    vector<float>& ssymv(const enum UPLO Uplo,
                        const float alpha, const matrix<float> &A,
                        const vector<float> &X, const float beta, vector<float> &Y);

    vector<float>& ssbmv(const enum UPLO Uplo,
                        const int K, const float alpha, const matrix<float> &A,
                        const vector<float> &X, const float beta, vector<float> &Y);

    matrix<float>& sger(const float alpha,
                       const vector<float> &X, const vector<float> &Y, matrix<float> &A);

    matrix<float>& ssyr(const enum UPLO Uplo,
                       const float alpha, const vector<float> &X, matrix<float> &A);

    matrix<float>& ssyr2(const enum UPLO Uplo, const float alpha,
                        const vector<float> &X, const vector<float> &Y, matrix<float> &A);

    vector<double>& dsymv(const enum UPLO Uplo,
                         const double alpha, const matrix<double> &A,
                         const vector<double> &X, const double beta, vector<double> &Y);

    vector<double>& dsbmv(const enum UPLO Uplo,
                         const int K, const double alpha, const matrix<double> &A,
                         const vector<double> &X, const double beta, vector<double> &Y);

    matrix<double>& dger(const double alpha,
                       const vector<double> &X, const vector<double> &Y, matrix<double> &A);

    matrix<double>& dsyr(const enum UPLO Uplo,
                       const double alpha, const vector<double> &X, matrix<double> &A);

    matrix<double>& dsyr2(const enum UPLO Uplo, const double alpha,
                        const vector<double> &X, const vector<double> &Y, matrix<double> &A);

/*
* ===========================================================================
* Prototypes for level 3 BLAS
* ===========================================================================
*/

    /* Routines with standard 4 prefixes (S, D, C, Z) */
    matrix<float>& sgemm(const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                        const float alpha, const matrix<float> &A,
                        const matrix<float> &B,
                        const float beta, matrix<float> &C);

    matrix<float>& ssymm(const enum SIDE Side, const enum UPLO Uplo,
                        const float alpha, const matrix<float> &A,
                        const matrix<float> &B,
                        const float beta, matrix<float> &C);

    matrix<float>& ssyrk(__attribute__((unused)) const enum UPLO Uplo, const enum TRANSPOSE Trans,
                         const float alpha, const matrix<float> &A,
                         const float beta, matrix<float> &C);

    matrix<float>& ssyr2k(__attribute__((unused)) const enum UPLO Uplo, const enum TRANSPOSE Trans,
                          const float alpha, const matrix<float> &A,
                          const float beta, const matrix<float> &B, matrix<float> &C);

    matrix<float>& strmm(const enum SIDE Side, const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const float alpha, const matrix<float> &A, matrix<float> &B);

    matrix<float>& strsm(const enum SIDE Side, const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const float alpha, const matrix<float> &A, matrix<float> &B);

    matrix<double> &dgemm(const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                          const double alpha, const matrix<double> &A,
                          const matrix<double> &B,
                          const double beta, matrix<double> &C);

    matrix<double> &dsymm(const enum SIDE Side, const enum UPLO Uplo,
                          const double alpha, const matrix<double> &A,
                          const matrix<double> &B,
                          const double beta, matrix<double> &C);

    matrix<double> &dsyrk(__attribute__((unused)) const enum UPLO Uplo, const enum TRANSPOSE Trans,
                          const double alpha, const matrix<double> &A,
                          const double beta, matrix<double> &C);

    matrix<double> &dsyr2k(__attribute__((unused)) const enum UPLO Uplo, const enum TRANSPOSE Trans,
                           const double alpha, const matrix<double> &A,
                           const double beta, const matrix<double> &B, matrix<double> &C);

    matrix<double> &dtrmm(const enum SIDE Side, const enum UPLO Uplo,
                          const enum TRANSPOSE TransA, const enum DIAG Diag,
                          const double alpha, const matrix<double> &A, matrix<double> &B);

    matrix<double> &dtrsm(const enum SIDE Side, const enum UPLO Uplo,
                          const enum TRANSPOSE TransA, const enum DIAG Diag,
                          const double alpha, const matrix<double> &A, matrix<double> &B);

    /* MISC */
    /* alpha denotes the suggested proportion of non-zero values */
    matrix<float> &srmgen(float alpha, matrix<float> &A);
    matrix<double> &drmgen(double alpha, matrix<double> &A);
};

}
