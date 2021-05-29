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
    arnoldi(std::shared_ptr<scmd::session> session) : _session(session), _scheduler(session) {}

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
        h->clear_all();
        Q->clear_all();
        _scheduler.scopy(*b, *q);
        _scheduler.sscal(1.0f / _scheduler.snrm2(*q), *q);
        transfer_vector_to_row(Q, 1, q);

        for (scylla_blas::index_type k = 1; k <= n; k++) {
            _scheduler.sgemv(scylla_blas::NoTrans, 1.0f, *A, *q, 0, *v);
            for (scylla_blas::index_type j = 1; j <= k; j++) {
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

    template<class T>
    struct containers {
        std::shared_ptr<scylla_blas::matrix<T>> A;
        std::shared_ptr<scylla_blas::vector<T>> b;
        std::shared_ptr<scylla_blas::matrix<T>> h;
        std::shared_ptr<scylla_blas::matrix<T>> Q;
        std::shared_ptr<scylla_blas::vector<T>> v;
        std::shared_ptr<scylla_blas::vector<T>> q;
        std::shared_ptr<scylla_blas::vector<T>> t;

        int64_t A_id,
            b_id,
            h_id,
            Q_id,
            v_id,
            q_id,
            t_id;

        void init(std::shared_ptr<scmd::session> session, int64_t initial_id) {
            A_id = initial_id++;
            A = std::make_shared<scylla_blas::matrix<T>>(session, A_id);
            b_id = initial_id++;
            b = std::make_shared<scylla_blas::vector<T>>(session, b_id);
            h_id = initial_id++;
            h = std::make_shared<scylla_blas::matrix<T>>(session, h_id);
            Q_id = initial_id++;
            Q = std::make_shared<scylla_blas::matrix<T>>(session, Q_id);
            v_id = initial_id++;
            v = std::make_shared<scylla_blas::vector<T>>(session, v_id);
            q_id = initial_id++;
            q = std::make_shared<scylla_blas::vector<T>>(session, q_id);
            t_id = initial_id++;
            t = std::make_shared<scylla_blas::vector<T>>(session, t_id);
        }

        containers(std::shared_ptr<scmd::session> session, int64_t initial_id) {
            init(session, initial_id);
        }

        containers(std::shared_ptr<scmd::session> session, int64_t initial_id, scylla_blas::index_type m, scylla_blas::index_type n) {
            int64_t initial_id_bak = initial_id;
            scylla_blas::matrix<T>::init(session, initial_id++, m, m);
            scylla_blas::vector<T>::init(session, initial_id++, m);
            scylla_blas::matrix<T>::init(session, initial_id++, n + 1, n);
            scylla_blas::matrix<T>::init(session, initial_id++, m, n + 1);
            scylla_blas::vector<T>::init(session, initial_id++, m);
            scylla_blas::vector<T>::init(session, initial_id++, m);
            scylla_blas::vector<T>::init(session, initial_id++, m);
            init(session, initial_id_bak);
        }
    };
};
