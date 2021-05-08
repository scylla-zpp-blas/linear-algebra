#pragma once

#include <memory>
#include <scmd.hh>

#include "scylla_blas/matrix.hh"
#include "scylla_blas/scylla_blas.hh"

#include "config.hh"
#include "generators/sparse_matrix_value_generator.hh"
#include "generators/preset_value_factory.hh"

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
                     std::shared_ptr<scylla_blas::random_value_factory<T>> value_factory = nullptr) {
        matrix_ptr = std::make_shared<scylla_blas::matrix<T>>(session, id);
        matrix_ptr->clear_all();

        if (value_factory != nullptr) {
            scylla_blas::sparse_matrix_value_generator<T> gen(w, h, 5 * h, id, value_factory);
            scylla_blas::load_matrix_from_generator(session, gen, *matrix_ptr);
        }
    }
public:
    std::shared_ptr<scylla_blas::matrix<float>> float_AxB;
    std::shared_ptr<scylla_blas::matrix<float>> float_BxA;
    std::shared_ptr<scylla_blas::matrix<float>> float_BxB;

    std::shared_ptr<scylla_blas::matrix<double>> double_AxB;
    std::shared_ptr<scylla_blas::matrix<double>> double_BxA;
    std::shared_ptr<scylla_blas::matrix<double>> double_BxB;

    matrix_fixture() :
            scylla_fixture(),
            float_AxB(nullptr),
            float_BxA(nullptr),
            float_BxB(nullptr),
            double_AxB(nullptr),
            double_BxA(nullptr),
            double_BxB(nullptr) {
        init_matrices(session);
    }

    void init_matrices(const std::shared_ptr<scmd::session> &session) {
        std::cerr << "Initializing test matrices..." << std::endl;

        scylla_blas::index_type A = 2 * BLOCK_SIZE + 3;
        scylla_blas::index_type B = 2 * BLOCK_SIZE + 6;

        std::shared_ptr<scylla_blas::random_value_factory<float>> f =
                std::make_shared<scylla_blas::random_value_factory<float>>(0, 9, 142);
        init_matrix(this->float_AxB, A, B, test_const::float_matrix_AxB_id, f);
        init_matrix(this->float_BxA, B, A, test_const::float_matrix_BxA_id, f);
        init_matrix(this->float_BxB, B, B, test_const::float_matrix_BxB_id);

        std::shared_ptr<scylla_blas::random_value_factory<double>> d =
                std::make_shared<scylla_blas::random_value_factory<double>>(0, 9, 242);
        init_matrix(this->double_AxB, A, B, test_const::double_matrix_AxB_id, d);
        init_matrix(this->double_BxA, B, A, test_const::double_matrix_BxA_id, d);
        init_matrix(this->double_BxB, B, B, test_const::double_matrix_BxB_id);

        std::cerr << "Test matrices initialized!" << std::endl;
    }
};

class vector_fixture : public scylla_fixture {
private:
    template<class T>
    void init_vector(std::shared_ptr<scylla_blas::vector<T>> &vector_ptr,
                     scylla_blas::index_type len, int64_t id,
                     std::shared_ptr<scylla_blas::value_factory<T>> value_factory = nullptr) {
        scylla_blas::vector<T>::clear(session, id);
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

    std::shared_ptr<scylla_blas::vector<float>> getScyllaVectorOf(std::vector<float> values) {
        std::shared_ptr<scylla_blas::value_factory<float>> factory =
                std::make_shared<scylla_blas::preset_value_factory<float>>(values);
        init_vector(float_A,
                    values.size(),
                    test_const::float_vector_1_id,
                    factory);
        return float_A;
    }

    std::shared_ptr<scylla_blas::vector<double>> getScyllaVectorOf(std::vector<double> values) {
        std::shared_ptr<scylla_blas::value_factory<double>> factory =
                std::make_shared<scylla_blas::preset_value_factory<double>>(values);
        init_vector(double_A,
                    values.size(),
                    test_const::double_vector_1_id,
                    factory);
        return double_A;
    }

    void init_vectors(const std::shared_ptr<scmd::session> &session) {
        std::cerr << "Initializing test vectors..." << std::endl;
        scylla_blas::index_type len = test_const::test_vector_len;

        std::shared_ptr<scylla_blas::value_factory<float>> f =
                std::make_shared<scylla_blas::random_value_factory<float>>(0, 9, 142);
        init_vector(this->float_A, len, test_const::float_vector_1_id, f);
        init_vector(this->float_B, len, test_const::float_vector_2_id,  f);
        init_vector(this->float_C, len, test_const::float_vector_3_id);

        std::shared_ptr<scylla_blas::value_factory<double>> d =
                std::make_shared<scylla_blas::random_value_factory<double>>(0, 9, 242);
        init_vector(this->double_A, len, test_const::double_vector_1_id, d);
        init_vector(this->double_B, len, test_const::double_vector_2_id, d);
        init_vector(this->double_C, len, test_const::double_vector_3_id);

        std::cerr << "Test vectors initialized!" << std::endl;
    }
};