#pragma once

#include "generators/value_factory.hh"
#include "generators/sparse_matrix_value_generator.hh"
#include <scylla_blas/routines.hh>

class arnoldi {
private:
    std::shared_ptr <scmd::session> _session;
    scylla_blas::routine_scheduler _scheduler;

    static void transfer_row_to_vector(std::shared_ptr<scylla_blas::matrix<float>> mat,
                                scylla_blas::index_type row_index,
                                std::shared_ptr<scylla_blas::vector<float>> vec) {
        scylla_blas::vector_segment<float> row = mat->get_row(row_index);
        vec->clear_all();
        vec->update_values(row);
    }

    static void transfer_vector_to_row(std::shared_ptr<scylla_blas::matrix<float>> mat,
                                       scylla_blas::index_type row_index,
                                       std::shared_ptr<scylla_blas::vector<float>> vec) {
        scylla_blas::vector_segment<float> row = vec->get_whole();
        mat->update_row(row_index, row);
    }

public:
    arnoldi(std::shared_ptr <scmd::session> session) : _session(session), _scheduler(session) {}

    /**
     *
     * @param A - m x m matrix
     * @param b - vector of m length
     * @param n - number of iterations
     * @param[out] h - result (n + 1) x n matrix h
     * @param[out] Q^T - result m x (n + 1) matrix Q
     * @param v - helper vector v of length m
     * @param q - helper vector q of length m
     * @param t - helper vector t of length m
     */
    void compute(std::shared_ptr<scylla_blas::matrix<float>> A,
                 std::shared_ptr<scylla_blas::vector<float>> b,
                 scylla_blas::index_type n,
                 std::shared_ptr<scylla_blas::matrix<float>> h,
                 std::shared_ptr<scylla_blas::matrix<float>> Q,
                 std::shared_ptr<scylla_blas::vector<float>> v,
                 std::shared_ptr<scylla_blas::vector<float>> q,
                 std::shared_ptr<scylla_blas::vector<float>> t) {
        auto m = A->get_row_count();
        h->clear_all();
        Q->clear_all();
        _scheduler.scopy(*b, *q);
        _scheduler.sscal(_scheduler.snrm2(*q), *q);
        transfer_vector_to_row(Q, 0, q);

        for (scylla_blas::index_type k = 0; k < n; k++) {
            _scheduler.sgemv(scylla_blas::NoTrans, 1.0f, *A, *q, 0, *v);
            for (scylla_blas::index_type j = 0; j < k + 1; j++) {
                transfer_row_to_vector(Q, j, t);
                h->insert_value(j, k, _scheduler.sdot(*t, *v));
                transfer_row_to_vector(Q, j, t);
                _scheduler.saxpy(-h->get_value(j, k), *t, *v); // v = v - h[j, k] * Q[:, j]
            }
            h->insert_value(k + 1, k, _scheduler.snrm2(*v));
            const float eps = 1e-12;
            if (h->get_value(k + 1, k) > eps) {
                _scheduler.scopy(*v, *q);
                _scheduler.sscal(1.0f / h->get_value(k + 1, k), *q);
                transfer_vector_to_row(Q, k + 1, q);
            }
            else {
                return; // Q, h;
            }
        }
        return; // Q, h;
    }

    struct containers {
        std::shared_ptr<scylla_blas::matrix<float>> A;
        std::shared_ptr<scylla_blas::vector<float>> b;
        std::shared_ptr<scylla_blas::matrix<float>> h;
        std::shared_ptr<scylla_blas::matrix<float>> Q;
        std::shared_ptr<scylla_blas::vector<float>> v;
        std::shared_ptr<scylla_blas::vector<float>> q;
        std::shared_ptr<scylla_blas::vector<float>> t;

        containers(scylla_blas::index_type m, scylla_blas::index_type n) {
            A = std::make_shared<scylla_blas::matirx>();
        }
    };


    template<class T>
    void init_vector(std::shared_ptr<scylla_blas::vector<T>> &vector_ptr,
                     scylla_blas::index_type len, int64_t id,
                     std::shared_ptr<value_factory<T>> value_factory = nullptr) {
        scylla_blas::vector<T>::clear(_session, id);
        vector_ptr = std::make_shared<scylla_blas::vector<T>>(_session, id);

        if (value_factory != nullptr) {
            std::vector<scylla_blas::vector_value<T>> values;

            for (scylla_blas::index_type i = 1; i <= len; i++)
                values.emplace_back(i, value_factory->next());

            vector_ptr->update_values(values);
        }
    }

    template<class T>
    void init_matrix(std::shared_ptr<scylla_blas::matrix<T>>& matrix_ptr,
                     scylla_blas::index_type w, scylla_blas::index_type h, int64_t id,
                     std::shared_ptr<value_factory<T>> value_factory = nullptr) {
        matrix_ptr = std::make_shared<scylla_blas::matrix<T>>(_session, id);
        matrix_ptr->clear_all();

        if (value_factory != nullptr) {
            sparse_matrix_value_generator<T> gen(w, h, 5 * h, id, value_factory);
            load_matrix_from_generator(_session, gen, *matrix_ptr);
        }
    }
};
