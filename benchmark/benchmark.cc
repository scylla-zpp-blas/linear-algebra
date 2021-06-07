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
    lm = std::make_unique<scylla_blas::matrix<float>>(session, l_matrix_id);
    rm = std::make_unique<scylla_blas::matrix<float>>(session, r_matrix_id);
    wm = std::make_unique<scylla_blas::matrix<float>>(session, w_matrix_id);

    lm->resize(length, length);
    rm->resize(length, length);
    wm->resize(length, length);

    lm->set_block_size(block_size);
    rm->set_block_size(block_size);
    wm->set_block_size(block_size);

    scheduler.srmgen(matrix_load, *rm);
    scheduler.srmgen(matrix_load, *lm);
}

void benchmark_mm::proc() {
    scheduler.sgemm(scylla_blas::NoTrans, scylla_blas::NoTrans, 1.0, *lm, *rm, 0.0, *wm);
}

void benchmark_mm::teardown() {
    //lm->clear_all();
    //rm->clear_all();
    //wm->clear_all();
}

void benchmark_mm::destroy() {
    //scylla_blas::matrix<float>::drop(session, l_matrix_id);
    //scylla_blas::matrix<float>::drop(session, r_matrix_id);
    //scylla_blas::matrix<float>::drop(session, w_matrix_id);
}

// Matrix * Vector

void benchmark_mv::init() {
    scylla_blas::matrix<float>::init(session, l_matrix_id, 0, 0, true);
    scylla_blas::vector<float>::init(session, r_vector_id, 0, true);
    scylla_blas::vector<float>::init(session, w_vector_id, 0, true);
}

void benchmark_mv::setup(int64_t block_size, int64_t length) {
    lm = std::make_unique<scylla_blas::matrix<float>>(session, l_matrix_id);
    rv = std::make_unique<scylla_blas::vector<float>>(session, r_vector_id);
    wv = std::make_unique<scylla_blas::vector<float>>(session, w_vector_id);


    lm->resize(length, length);
    rv->resize(length);
    wv->resize(length);

    lm->set_block_size(block_size);
    rv->set_block_size(block_size);
    wv->set_block_size(block_size);

    scheduler.srmgen(matrix_load, *lm);
    fill_vector(*rv, length);
}

void benchmark_mv::proc() {
    scheduler.sgemv(scylla_blas::NoTrans, 1.0, *lm, *rv, 0.0, *wv);
}

void benchmark_mv::teardown() {
    //lm->clear_all();
    //rv->clear_all();
    //wv->clear_all();
}

void benchmark_mv::destroy() {
    //scylla_blas::matrix<float>::drop(session, l_matrix_id);
    //scylla_blas::vector<float>::drop(session, r_vector_id);
    //scylla_blas::vector<float>::drop(session, w_vector_id);
}

// Vector * Vector

void benchmark_vv::init() {
    scylla_blas::vector<float>::init(session, l_vector_id, 0, true);
    scylla_blas::vector<float>::init(session, r_vector_id, 0, true);
}

void benchmark_vv::setup(int64_t block_size, int64_t length) {
    lv = std::make_unique<scylla_blas::vector<float>>(session, l_vector_id);
    rv = std::make_unique<scylla_blas::vector<float>>(session, r_vector_id);

    lv->resize(length);
    rv->resize(length);

    lv->set_block_size(block_size);
    rv->set_block_size(block_size);

    fill_vector(*lv, length);
    fill_vector(*rv, length);
}

void benchmark_vv::proc() {
    scheduler.sdot(*lv, *rv);
}

void benchmark_vv::teardown() {
    //lv->clear_all();
    //rv->clear_all();
}

void benchmark_vv::destroy() {
    //scylla_blas::vector<float>::drop(session, l_vector_id);
    //scylla_blas::vector<float>::drop(session, r_vector_id);
}

template<typename F, typename... Args>
double measure_time(F callable, Args... args) {
    auto t1 = std::chrono::high_resolution_clock::now();
    callable(args...);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = t2-t1;
    return duration.count();
}

benchmark_result perform_benchmark(std::unique_ptr<base_benchmark> tester, const std::vector<int64_t> &block_sizes, const std::vector<int64_t> &problem_sizes) {
    benchmark_result result{};

    LogInfo("Starting initialization...  ");
    result.init_time = measure_time([&](){tester->init();});
    LogInfo("Initialization took {}ms", result.init_time);

    for(int64_t block_size : block_sizes) {
        for(int64_t problem_size : problem_sizes) {
            benchmark_result::result_t r{};
            LogInfo("Block size: {}, problem size: {}", block_size, problem_size);

            LogInfo("\tStarting setup");
            r.setup_time = measure_time([&](int64_t b, int64_t l){tester->setup(b, l);}, block_size, problem_size);
            LogInfo("\tSetup took {}ms", r.setup_time);

            LogInfo("\tStarting procedure");
            r.proc_time = measure_time([&](){tester->proc();});
            LogInfo("\tProcedure took {}ms", r.proc_time);

            LogInfo("\tStarting teardown");
            r.teardown_time = measure_time([&](){tester->teardown();});
            LogInfo("\tTeardown took {}ms", r.teardown_time);

            result.tests.emplace_back(block_size, problem_size, r);
        }
    }

    LogInfo("Starting destroy");
    result.destroy_time = measure_time([&](){tester->destroy();});
    LogInfo("Destroy took {}ms\n", result.destroy_time);

    return result;
}