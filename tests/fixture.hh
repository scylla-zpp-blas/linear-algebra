#pragma once

#include <memory>
#include <scmd.hh>

#include "scylla_blas/matrix.hh"
#include "test_utils.hh"

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
                     std::shared_ptr<value_factory<T>> value_factory = nullptr) {
        matrix_ptr = std::make_shared<scylla_blas::matrix<T>>(session, id);
        matrix_ptr->clear_all();

        if (value_factory != nullptr) {
            sparse_matrix_value_generator<T> gen(w, h, 5 * h, id, value_factory);
            load_matrix_from_generator(session, gen, *matrix_ptr);
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

        scylla_blas::index_type A = test_const::matrix_A;
        scylla_blas::index_type B = test_const::matrix_B;

        std::shared_ptr<value_factory<float>> f =
                std::make_shared<value_factory<float>>(0, 9, 142);
        init_matrix(this->float_AxB, A, B, test_const::float_matrix_AxB_id, f);
        init_matrix(this->float_BxA, B, A, test_const::float_matrix_BxA_id, f);
        init_matrix(this->float_BxB, B, B, test_const::float_matrix_BxB_id, f);

        std::shared_ptr<value_factory<double>> d =
                std::make_shared<value_factory<double>>(0, 9, 242);
        init_matrix(this->double_AxB, A, B, test_const::double_matrix_AxB_id, d);
        init_matrix(this->double_BxA, B, A, test_const::double_matrix_BxA_id, d);
        init_matrix(this->double_BxB, B, B, test_const::double_matrix_BxB_id, d);

        std::cerr << "Test matrices initialized!" << std::endl;
    }
};

class vector_fixture : public scylla_fixture {
protected:
    template<class T>
    void init_vector(std::shared_ptr<scylla_blas::vector<T>>& vector_ptr,
                     scylla_blas::index_type len, int64_t id,
                     std::shared_ptr<value_factory<T>> value_factory = nullptr) {
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
    std::shared_ptr<scylla_blas::vector<float>> float_A2;

    std::shared_ptr<scylla_blas::vector<double>> double_A;
    std::shared_ptr<scylla_blas::vector<double>> double_A2;

    vector_fixture() :
            scylla_fixture(),
            float_A(nullptr),
            float_A2(nullptr),
            double_A(nullptr),
            double_A2(nullptr) {
        init_vectors(session);
    }

    void init_vectors(const std::shared_ptr<scmd::session> &session) {
        std::cerr << "Initializing test vectors..." << std::endl;
        scylla_blas::index_type len_A = test_const::test_vector_len_A;

        std::shared_ptr<value_factory<float>> f =
                std::make_shared<value_factory<float>>(0, 9, 142);
        init_vector(this->float_A, len_A, test_const::float_vector_1_id, f);
        init_vector(this->float_A2, len_A, test_const::float_vector_2_id,  f);

        std::shared_ptr<value_factory<double>> d =
                std::make_shared<value_factory<double>>(0, 9, 242);
        init_vector(this->double_A, len_A, test_const::double_vector_1_id, d);
        init_vector(this->double_A2, len_A, test_const::double_vector_2_id, d);

        std::cerr << "Test vectors initialized!" << std::endl;
    }
};

class vector_fixture_large : public vector_fixture {
public:
    std::shared_ptr<scylla_blas::vector<float>> float_B;
    std::shared_ptr<scylla_blas::vector<float>> float_B2;

    std::shared_ptr<scylla_blas::vector<double>> double_B;
    std::shared_ptr<scylla_blas::vector<double>> double_B2;

    vector_fixture_large() :
            vector_fixture(),
            float_B(nullptr),
            float_B2(nullptr),
            double_B(nullptr),
            double_B2(nullptr) {
        init_vectors(session);
    }

    void init_vectors(const std::shared_ptr<scmd::session> &session) {
        std::cerr << "Initializing extra test vectors..." << std::endl;
        scylla_blas::index_type len_B = test_const::test_vector_len_B;

        std::shared_ptr<value_factory<float>> f =
                std::make_shared<value_factory<float>>(0, 9, 323);

        init_vector(this->float_B, len_B, test_const::float_vector_3_id,  f);
        init_vector(this->float_B2, len_B, test_const::float_vector_4_id,  f);

        std::shared_ptr<value_factory<double>> d =
                std::make_shared<value_factory<double>>(0, 9, 323);

        init_vector(this->double_B, len_B, test_const::double_vector_3_id, d);
        init_vector(this->double_B2, len_B, test_const::double_vector_4_id, d);

        std::cerr << "Test vectors initialized!" << std::endl;
    }
};

class mixed_fixture : public vector_fixture_large, public matrix_fixture {
public:
    std::shared_ptr<scylla_blas::routine_scheduler> scheduler;
    std::shared_ptr<scmd::session> session;

    mixed_fixture() :
        vector_fixture_large(),
        matrix_fixture(),
        session(matrix_fixture::session),
        scheduler(matrix_fixture::scheduler)
    {}
};