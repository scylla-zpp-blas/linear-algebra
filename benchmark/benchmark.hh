#pragma once
#include <map>

#include <scmd.hh>

#include "scylla_blas/matrix.hh"
#include "scylla_blas/vector.hh"
#include "scylla_blas/routines.hh"

#include "sparse_matrix_value_generator.hh"
#include "random_value_factory.hh"

#include "const.hh"

template<class T>
void load_vector_from_generator(value_factory<T> &gen, scylla_blas::vector<T> &vector) {
    scylla_blas::vector_segment<T> next_segment;
    scylla_blas::index_t in_segment_index = 1;
    scylla_blas::index_t segment_number = 1;
    scylla_blas::index_t segment_offset = 0;

    LogDebug("Filling vector with length {} and block size {}", vector.get_length(), vector.get_block_size());
    for(size_t i = 0; i < vector.get_length(); i++) {
        if (in_segment_index > vector.get_block_size()) {
            vector.insert_segment(segment_number, next_segment);
            next_segment.clear();

            in_segment_index = 1;
            segment_number++;
            segment_offset += vector.get_block_size();
        }
        T next_val = gen.next();
        next_segment.emplace_back(in_segment_index, next_val);
        in_segment_index++;
    }

    if (in_segment_index != 1) {
        vector.insert_segment(segment_number, next_segment);
        next_segment.clear();
    }

    LogInfo("Loaded a vector {} from a generator", vector.get_id());
}

template<typename T>
void fill_vector(scylla_blas::vector<T> &v, scylla_blas::index_t length) {
    std::shared_ptr<value_factory<T>> f = std::make_shared<random_value_factory<T>>(0, 9, RANDOM_SEED);
    load_vector_from_generator(*f, v);
}

struct benchmark_result {
    using result_t = struct { double setup_time; double proc_time; double teardown_time; };
    double init_time;
    double destroy_time;
    std::vector<std::tuple<int64_t, int64_t, result_t>> tests;
};

class base_benchmark {
protected:
    std::shared_ptr <scmd::session> session;
    scylla_blas::routine_scheduler scheduler;
    double matrix_load;
public:
    explicit base_benchmark(const std::shared_ptr<scmd::session> &session) : session(session), scheduler(session) {}
    virtual void init() = 0;
    virtual void setup(int64_t block_size, int64_t length) = 0;
    virtual void proc() = 0;
    virtual void teardown() = 0;
    virtual void destroy() = 0;
    void set_max_workers(int64_t new_max_workers) {
        scheduler.set_max_used_workers(new_max_workers);
    }
    void set_matrix_load(double load) {
        this->matrix_load = load;
    }
};

class benchmark_mm : public base_benchmark {
    std::unique_ptr<scylla_blas::matrix<float>> lm;
    std::unique_ptr<scylla_blas::matrix<float>> rm;
    std::unique_ptr<scylla_blas::matrix<float>> wm;
public:
    explicit benchmark_mm(const std::shared_ptr<scmd::session> &session) : base_benchmark(session) {}
    void init() override;
    void setup(int64_t block_size, int64_t length) override;
    void proc() override;
    void teardown() override;
    void destroy() override;
};

class benchmark_mv : public base_benchmark {
    std::unique_ptr<scylla_blas::matrix<float>> lm;
    std::unique_ptr<scylla_blas::vector<float>> rv;
    std::unique_ptr<scylla_blas::vector<float>> wv;
public:
    explicit benchmark_mv(const std::shared_ptr<scmd::session> &session) : base_benchmark(session) {}
    void init() override;
    void setup(int64_t block_size, int64_t length) override;
    void proc() override;
    void teardown() override;
    void destroy() override;
};

class benchmark_vv : public base_benchmark {
    std::unique_ptr<scylla_blas::vector<float>> lv;
    std::unique_ptr<scylla_blas::vector<float>> rv;
public:
    explicit benchmark_vv(const std::shared_ptr<scmd::session> &session) : base_benchmark(session) {}
    void init() override;
    void setup(int64_t block_size, int64_t length) override;
    void proc() override;
    void teardown() override;
    void destroy() override;
};

benchmark_result perform_benchmark(std::unique_ptr<base_benchmark> tester,
                                   const std::vector<int64_t> &block_sizes,
                                   const std::vector<int64_t> &problem_sizes,
                                   bool autoclean);