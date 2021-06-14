#include <chrono>

#include <scylla_blas/matrix.hh>
#include <scylla_blas/vector.hh>

#include "const.hh"
#include "benchmark.hh"

// Matrix * Matrix

void benchmark_mm::init() {
    scylla_blas::matrix<float>::init(session, l_matrix_id, 0, 0, true);
    scylla_blas::matrix<float>::init(session, r_matrix_id, 0, 0, true);
    scylla_blas::matrix<float>::init(session, w_matrix_id, 0, 0, true);
}

void benchmark_mm::setup(int64_t block_size, int64_t length) {
    left_matrix = std::make_unique<scylla_blas::matrix<float>>(session, l_matrix_id);
    right_matrix = std::make_unique<scylla_blas::matrix<float>>(session, r_matrix_id);
    result_matrix = std::make_unique<scylla_blas::matrix<float>>(session, w_matrix_id);

    left_matrix->resize(length, length);
    right_matrix->resize(length, length);
    result_matrix->resize(length, length);

    left_matrix->set_block_size(block_size);
    right_matrix->set_block_size(block_size);
    result_matrix->set_block_size(block_size);

    scheduler.srmgen(matrix_load, *right_matrix);
    scheduler.srmgen(matrix_load, *left_matrix);
}

void benchmark_mm::proc() {
    scheduler.sgemm(scylla_blas::NoTrans, scylla_blas::NoTrans, 1.0, *left_matrix, *right_matrix, 0.0, *result_matrix);
}

void benchmark_mm::teardown() {
    left_matrix->clear_all();
    right_matrix->clear_all();
    result_matrix->clear_all();
}

void benchmark_mm::destroy() {
    scylla_blas::matrix<float>::drop(session, l_matrix_id);
    scylla_blas::matrix<float>::drop(session, r_matrix_id);
    scylla_blas::matrix<float>::drop(session, w_matrix_id);
}

// Matrix * Vector

void benchmark_mv::init() {
    scylla_blas::matrix<float>::init(session, l_matrix_id, 0, 0, true);
    scylla_blas::vector<float>::init(session, r_vector_id, 0, true);
    scylla_blas::vector<float>::init(session, w_vector_id, 0, true);
}

void benchmark_mv::setup(int64_t block_size, int64_t length) {
    left_matrix = std::make_unique<scylla_blas::matrix<float>>(session, l_matrix_id);
    right_vector = std::make_unique<scylla_blas::vector<float>>(session, r_vector_id);
    result_vector = std::make_unique<scylla_blas::vector<float>>(session, w_vector_id);

    left_matrix->resize(length, length);
    right_vector->resize(length);
    result_vector->resize(length);

    left_matrix->set_block_size(block_size);
    right_vector->set_block_size(block_size);
    result_vector->set_block_size(block_size);

    scheduler.srmgen(matrix_load, *left_matrix);
    fill_vector(*right_vector, length);
}

void benchmark_mv::proc() {
    scheduler.sgemv(scylla_blas::NoTrans, 1.0, *left_matrix, *right_vector, 0.0, *result_vector);
}

void benchmark_mv::teardown() {
    left_matrix->clear_all();
    right_vector->clear_all();
    result_vector->clear_all();
}

void benchmark_mv::destroy() {
    scylla_blas::matrix<float>::drop(session, l_matrix_id);
    scylla_blas::vector<float>::drop(session, r_vector_id);
    scylla_blas::vector<float>::drop(session, w_vector_id);
}

// Vector * Vector

void benchmark_vv::init() {
    scylla_blas::vector<float>::init(session, l_vector_id, 0, true);
    scylla_blas::vector<float>::init(session, r_vector_id, 0, true);
}

void benchmark_vv::setup(int64_t block_size, int64_t length) {
    left_vector = std::make_unique<scylla_blas::vector<float>>(session, l_vector_id);
    right_vector = std::make_unique<scylla_blas::vector<float>>(session, r_vector_id);

    left_vector->resize(length);
    right_vector->resize(length);

    left_vector->set_block_size(block_size);
    right_vector->set_block_size(block_size);

    fill_vector(*left_vector, length);
    fill_vector(*right_vector, length);
}

void benchmark_vv::proc() {
    scheduler.sdot(*left_vector, *right_vector);
}

void benchmark_vv::teardown() {
    left_vector->clear_all();
    right_vector->clear_all();
}

void benchmark_vv::destroy() {
    scylla_blas::vector<float>::drop(session, l_vector_id);
    scylla_blas::vector<float>::drop(session, r_vector_id);
}

template<typename F, typename... Args>
double measure_time(F callable, Args... args) {
    auto t1 = std::chrono::high_resolution_clock::now();
    callable(args...);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = t2-t1;
    return duration.count();
}

benchmark_result perform_benchmark(std::unique_ptr<base_benchmark> tester,
                                   const std::vector<int64_t> &block_sizes,
                                   const std::vector<int64_t> &problem_sizes,
                                   bool autoclean) {
    benchmark_result results{};

    LogInfo("Starting initialization...  ");
    results.init_time = measure_time([&](){tester->init();});
    LogInfo("Initialization took {}ms", results.init_time);

    for(int64_t block_size : block_sizes) {
        for(int64_t problem_size : problem_sizes) {
            benchmark_result::result_t current_result{};
            LogInfo("Block size: {}, problem size: {}", block_size, problem_size);

            LogInfo("\tStarting setup");
            current_result.setup_time = measure_time([&](int64_t b, int64_t l){tester->setup(b, l);}, block_size, problem_size);
            LogInfo("\tSetup took {}ms", current_result.setup_time);

            LogInfo("\tStarting procedure");
            current_result.proc_time = measure_time([&](){tester->proc();});
            LogInfo("\tProcedure took {}ms", current_result.proc_time);

            if (autoclean) {
                LogInfo("\tStarting teardown");
                current_result.teardown_time = measure_time([&](){tester->teardown();});
                LogInfo("\tTeardown took {}ms", current_result.teardown_time);
            } else {
                current_result.teardown_time = 0;
                LogDebug("\tAutoclean off: skipping teardown");
            }

            results.tests.emplace_back(block_size, problem_size, current_result);
        }
    }

    if (autoclean) {
        LogInfo("Starting destroy");
        results.destroy_time = measure_time([&](){tester->destroy();});
        LogInfo("Destroy took {}ms\n", results.destroy_time);
    } else {
        results.destroy_time = 0;
        LogDebug("\tAutoclean off: skipping destroy");
    }

    return results;
}