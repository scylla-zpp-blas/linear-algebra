#pragma once

/* BASED ON cblas.h */

#include <complex>
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
    std::shared_ptr <scmd::session> _session;
    using cfloat = std::complex<float>;
    using zdouble = std::complex<double>;

    template<class T>
    matrix<T>& gemm(const enum scylla_blas::TRANSPOSE TransA,
                    const enum scylla_blas::TRANSPOSE TransB,
                    const T alpha, const scylla_blas::matrix<T> &A,
                    const scylla_blas::matrix<T> &B,
                    const T beta, scylla_blas::matrix<T> &C);

    const int64_t _subtask_queue_id;
    scylla_queue _subtask_queue;
    scylla_queue _main_worker_queue;

    scylla_queue prepare_queue() const {
        scylla_queue::create_queue(_session, _subtask_queue_id, false, true);
        return scylla_queue(_session, _subtask_queue_id);
    }
public:
    /* The queue used for subroutines requested in methods */
    routine_scheduler(const std::shared_ptr <scmd::session> &session) :
        _session(session),
        _subtask_queue_id(get_timestamp()),
        _subtask_queue(prepare_queue()),
        _main_worker_queue(_session, DEFAULT_WORKER_QUEUE_ID)
    {}

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

    index_type isamax(const vector<float> &X);
    index_type idamax(const vector<double> &X);

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

    vector<float> sgemv(const enum TRANSPOSE TransA,
                        const float alpha, const matrix<float> &A,
                        const vector<float> &X, const float beta);

    vector<float> sgbmv(const enum TRANSPOSE TransA,
                        const int KL, const int KU,
                        const float alpha, const matrix<float> &A,
                        const vector<float> &X, const float beta);

    vector<float> strmv(const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const matrix<float> &A, vector<float> &X);

    vector<float> stbmv(const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const int K, const matrix<float> &A, vector<float> &X);

    vector<float> stpmv(const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const matrix<float> &Ap, vector<float> &X);

    vector<float> strsv(const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const matrix<float> &A, vector<float> &X);

    vector<float> stbsv(const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const int K, const matrix<float> &A, vector<float> &X);

    vector<float> stpsv(const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const matrix<float> &Ap, vector<float> &X);

    vector<double> dgemv(const enum TRANSPOSE TransA,
                         const double alpha, const matrix<double> &A,
                         const vector<double> &X, const double beta);

    vector<double> dgbmv(const enum TRANSPOSE TransA,
                         const int KL, const int KU,
                         const double alpha, const matrix<double> &A,
                         const vector<double> &X, const double beta);

    vector<double> dtrmv(const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const matrix<double> &A, vector<double> &X);

    vector<double> dtbmv(const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const int K, const matrix<double> &A, vector<double> &X);

    vector<double> dtpmv(const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const matrix<double> &Ap, vector<double> &X);

    vector<double> dtrsv(const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const matrix<double> &A, vector<double> &X);

    vector<double> dtbsv(const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const int K, const matrix<double> &A, vector<double> &X);

    vector<double> dtpsv(const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const matrix<double> &Ap, vector<double> &X);

    /* Routines with S and D prefixes only */
    vector<float> ssymv(const enum UPLO Uplo,
                        const float alpha, const matrix<float> &A,
                        const vector<float> &X, const float beta);

    vector<float> ssbmv(const enum UPLO Uplo,
                        const int K, const float alpha, const matrix<float> &A,
                        const vector<float> &X, const float beta);

    vector<float> sspmv(const enum UPLO Uplo,
                        const float alpha, const matrix<float> &Ap,
                        const vector<float> &X, const float beta);

    matrix<float> sger(const float alpha, const vector<float> &X, const vector<float> &Y);

    matrix<float> ssyr(const enum UPLO Uplo,
                       const float alpha, const vector<float> &X);

    matrix<float> sspr(const enum UPLO Uplo,
                       const float alpha, const vector<float> &X);

    matrix<float> ssyr2(const enum UPLO Uplo,
                        const float alpha, const vector<float> &X, const vector<float> &Y);

    matrix<double> sspr2(const enum UPLO Uplo,
                         const double alpha, const vector<double> &X, const vector<double> &Y);

    vector<double> dsymv(const enum UPLO Uplo,
                         const double alpha, const matrix<double> &A,
                         const vector<double> &X, const double beta);

    vector<double> dsbmv(const enum UPLO Uplo,
                         const int K, const double alpha, const matrix<double> &A,
                         const vector<double> &X, const double beta);

    vector<double> dspmv(const enum UPLO Uplo,
                         const double alpha, const matrix<double> &Ap,
                         const vector<double> &X, const double beta);

    matrix<double> dger(const double alpha, const vector<double> &X, const vector<double> &Y);

    matrix<double> dsyr(const enum UPLO Uplo,
                        const double alpha, const vector<double> &X);

    matrix<double> dspr(const enum UPLO Uplo,
                        const double alpha, const vector<double> &X);

    matrix<double> dsyr2(const enum UPLO Uplo,
                         const double alpha, const vector<double> &X, const vector<double> &Y);

    matrix<double> dspr2(const enum UPLO Uplo,
                         const double alpha, const vector<double> &X, const vector<double> &Y);

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

    matrix<float>& ssyrk(const enum UPLO Uplo, const enum TRANSPOSE Trans, const int K,
                        const float alpha, const matrix<float> &A,
                        const float beta, matrix<float> &B);

    matrix<float>& ssyr2k(const enum UPLO Uplo, const enum TRANSPOSE Trans, const int K,
                         const float alpha, const matrix<float> &A,
                         const float beta, matrix<float> &B);

    matrix<float>& strmm(const enum SIDE Side, const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const float alpha, const matrix<float> &A, matrix<float> &B);

    matrix<float>& strsm(const enum SIDE Side, const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const float alpha, const matrix<float> &A, matrix<float> &B);

    matrix<double>& dgemm(const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                        const double alpha, const matrix<double> &A,
                        const matrix<double> &B,
                        const double beta, matrix<double> &C);

    matrix<double>& dsymm(const enum SIDE Side, const enum UPLO Uplo,
                        const double alpha, const matrix<double> &A,
                        const matrix<double> &B,
                        const double beta, matrix<double> &C);

    matrix<double>& dsyrk(const enum UPLO Uplo, const enum TRANSPOSE Trans, const int K,
                        const double alpha, const matrix<double> &A,
                        const double beta, matrix<double> &B);

    matrix<double>& dsyr2k(const enum UPLO Uplo, const enum TRANSPOSE Trans, const int K,
                         const double alpha, const matrix<double> &A,
                         const double beta, matrix<double> &B);

    matrix<double>& dtrmm(const enum SIDE Side, const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const double alpha, const matrix<double> &A, matrix<double> &B);

    matrix<double>& dtrsm(const enum SIDE Side, const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const double alpha, const matrix<double> &A, matrix<double> &B);
};

}
