#pragma once

#include <memory>
#include <scmd.hh>

#include "scylla_blas/matrix.hh"
#include "scylla_blas/scylla_blas.hh"

#include "config.hh"
#include "generators/sparse_matrix_value_generator.hh"

class scylla_fixture {
public:
    std::shared_ptr<scmd::session> session;
    std::shared_ptr<scylla_blas::routine_scheduler> scheduler;

    scylla_fixture() : 
        session(nullptr),
        scheduler(nullptr)
    {
        global_config::init();
        connect();
        this->scheduler = std::make_shared<scylla_blas::routine_scheduler>(session);
    }

    ~scylla_fixture() {
    }

    void connect(const std::string& ip = global_config::scylla_ip, const std::string& port = global_config::scylla_port) {
        this->session = std::make_shared<scmd::session>(ip, port);
    }
};

class matrix_fixture : public scylla_fixture {
    template<class T>
    void init_matrix(std::shared_ptr<scylla_blas::matrix<T>>& matrix_ptr,
                     scylla_blas::index_type w, scylla_blas::index_type h, int64_t id,
                     std::shared_ptr<scylla_blas::value_factory<T>> value_factory = nullptr) {
        scylla_blas::matrix<T>::init(session, id, w, h, true);
        matrix_ptr = std::make_shared<scylla_blas::matrix<T>>(session, id);

        if (value_factory != nullptr) {
            scylla_blas::sparse_matrix_value_generator<T> gen(w, h, 5 * h, id, value_factory);
            scylla_blas::load_matrix_from_generator(session, gen, *matrix_ptr);
        }
    }
public:
    std::shared_ptr<scylla_blas::matrix<float>> float_5x6;
    std::shared_ptr<scylla_blas::matrix<float>> float_6x5;
    std::shared_ptr<scylla_blas::matrix<float>> float_6x6;

    std::shared_ptr<scylla_blas::matrix<double>> double_5x6;
    std::shared_ptr<scylla_blas::matrix<double>> double_6x5;
    std::shared_ptr<scylla_blas::matrix<double>> double_6x6;

    matrix_fixture() :
        scylla_fixture(),
        float_5x6(nullptr),
        float_6x5(nullptr),
        float_6x6(nullptr),
        double_5x6(nullptr),
        double_6x5(nullptr),
        double_6x6(nullptr) {
        init_matrices(session);
    }

    void init_matrices(const std::shared_ptr<scmd::session> &session) {
        std::cerr << "Initializing test matrices..." << std::endl;

        std::shared_ptr<scylla_blas::value_factory<float>> f =
                std::make_shared<scylla_blas::value_factory<float>>(0, 9, 142);
        init_matrix(this->float_5x6, 2 * BLOCK_SIZE + 3, 2 * BLOCK_SIZE + 6, 1, f);
        init_matrix(this->float_6x5, 2 * BLOCK_SIZE + 6, 2 * BLOCK_SIZE + 3, 2, f);
        init_matrix(this->float_6x6, 2 * BLOCK_SIZE + 6, 2 * BLOCK_SIZE + 6, 3);

        std::shared_ptr<scylla_blas::value_factory<double>> d =
                std::make_shared<scylla_blas::value_factory<double>>(0, 9, 242);
        init_matrix(this->double_5x6, 2 * BLOCK_SIZE + 3, 2 * BLOCK_SIZE + 6, 11, d);
        init_matrix(this->double_6x5, 2 * BLOCK_SIZE + 6, 2 * BLOCK_SIZE + 3, 12, d);
        init_matrix(this->double_6x6, 2 * BLOCK_SIZE + 6, 2 * BLOCK_SIZE + 6, 13);

        std::cerr << "Test matrices initialized!" << std::endl;
    }
};

class vector_fixture : public scylla_fixture {
    template<class T>
    void init_vector(std::shared_ptr<scylla_blas::vector<T>>& vector_ptr,
                     scylla_blas::index_type len, int64_t id,
                     std::shared_ptr<scylla_blas::value_factory<T>> value_factory = nullptr) {
        scylla_blas::vector<T>::init(session, id, len, true);
        vector_ptr = std::make_shared<scylla_blas::vector<T>>(session, id);

        if (value_factory != nullptr) {
            std::vector<scylla_blas::vector_value<T>> values;

            for (scylla_blas::index_type i = 1; i <= len; i++)
                values.emplace_back(i, value_factory->next());

            vector_ptr->update_values(values);
        }
    }
public:
    std::shared_ptr<scylla_blas::vector<float>> float_A;
    std::shared_ptr<scylla_blas::vector<float>> float_B;
    std::shared_ptr<scylla_blas::vector<float>> float_C;

    std::shared_ptr<scylla_blas::vector<double>> double_A;
    std::shared_ptr<scylla_blas::vector<double>> double_B;
    std::shared_ptr<scylla_blas::vector<double>> double_C;

    vector_fixture() :
            scylla_fixture(),
            float_A(nullptr),
            float_B(nullptr),
            float_C(nullptr),
            double_A(nullptr),
            double_B(nullptr),
            double_C(nullptr) {
        init_vectors(session);
    }

    void init_vectors(const std::shared_ptr<scmd::session> &session) {
        std::cerr << "Initializing test vectors..." << std::endl;
        scylla_blas::index_type len = 2 * BLOCK_SIZE + 3;

        std::shared_ptr<scylla_blas::value_factory<float>> f =
                std::make_shared<scylla_blas::value_factory<float>>(0, 9, 142);
        init_vector(this->float_A, len, 1, f);
        init_vector(this->float_B, len, 2, f);
        init_vector(this->float_C, len, 3);

        std::shared_ptr<scylla_blas::value_factory<double>> d =
                std::make_shared<scylla_blas::value_factory<double>>(0, 9, 242);
        init_vector(this->double_A, len, 11, d);
        init_vector(this->double_B, len, 12, d);
        init_vector(this->double_C, len, 13);

        std::cerr << "Test vectors initialized!" << std::endl;
    }
};