#pragma once

/* BASED ON cblas.h */

#include <matrix.hh>
#include <vector.hh>

namespace scylla_blas {

enum ORDER {
    RowMajor = 101, ColMajor = 102
};
enum TRANSPOSE {
    NoTrans = 111, Trans = 112, ConjTrans = 113
};
enum UPLO {
    Upper = 121, Lower = 122
};
enum DIAG {
    NonUnit = 131, Unit = 132
};
enum SIDE {
    Left = 141, Right = 142
};

/* If we implement routines the C-like way, passing pointers
 * to results as function parameters, remembering scmd::session
 * in routine_factory is likely redundant.
 *
 * However, we will likely prefer to simplify by reducing
 * the number of arguments and returning results the C++ way.
 */
class routine_factory {
    std::shared_ptr <scmd::session> _session;
    using cfloat = std::complex<float>;
    using zdouble = std::complex<double>;

public:
    routine_factory(const std::shared_ptr <scmd::session> &session)

    _session(session) {}

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

    /* Functions having prefixes Z and C only [[UNUSED]] */
    cfloat cdotu(const vector <cfloat> &X, const vector <cfloat> &Y);
    cfloat cdotc(const vector <cfloat> &X, const vector <cfloat> &Y);

    /* [[UNUSED]] */
    zdouble zdotu(const vector <zdouble> &X, const vector <zdouble> &Y);
    zdouble zdotc(const vector <zdouble> &X, const vector <zdouble> &Y);


    /* Functions having prefixes S D SC DZ */
    float snrm2(const vector<float> &X);
    float sasum(const vector<float> &X);

    double dnrm2(const vector<double> &X);
    double dasum(const vector<double> &X);

    /* [[UNUSED]] */
    float scnrm2(const vector <cfloat> &X);
    float scasum(const vector <cfloat> &X);

    /* [[UNUSED]] */
    double dznrm2(const vector <zdouble> &X);
    double dzasum(const vector <zdouble> &X);

    /* Functions having standard 4 prefixes (S D C Z) */
    index_type isamax(const vector<float> &X);
    index_type idamax(const vector<double> &X);

    /* [[UNUSED]] */
    index_type icamax(const vector <cfloat> &X);
    index_type izamax(const vector <zdouble> &X);


/*
* ===========================================================================
* Prototypes for level 1 BLAS routines
* ===========================================================================
*/

    /* Routines with standard 4 prefixes (s, d, c, z) */
    void sswap(vector<float> &X vector<float> &Y);
    void scopy(const vector<float> &X vector<float> &Y);
    void saxpy(const float alpha, const vector<float> &X, vector<float> &Y);

    void dswap(vector<double> &X, vector<double> &Y);
    void dcopy(const vector<double> &X, vector<double> &Y);
    void daxpy(const double alpha, const vector<double> &X, vector<double> &Y);

    /* [[UNUSED]] */
    void cswap(vector <cfloat> &X, vector <cfloat> &Y);
    void ccopy(const vector <cfloat> &X, vector <cfloat> &Y);
    void caxpy(const cfloat alpha, const vector <cfloat> &X, vector <cfloat> &Y);

    /* [[UNUSED]] */
    void zswap(vector <zdouble> &X, vector <zdouble> &Y);
    void zcopy(const vector <zdouble> &X, vector <zdouble> &Y);
    void zaxpy(const zdouble alpha, const vector <zdouble> &X, vector <zdouble> &Y);

    /* Routines with S and D prefix only */
    void srotg(float *a, float *b, float *c, float *s);
    void srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
    void srot(vector<float> &X, vector<float> &Y, const float c, const float s);
    void srotm(vector<float> &X, vector<float> &Y, const float *P);

    void drotg(double *a, double *b, double *c, double *s);
    void drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
    void drot(vector<double> &X, vector<double> &Y, const double c, const double s);
    void drotm(vector<double> &X, vector<double> &Y, const double *P);

    /* Routines with S D C Z CS and ZD prefixes */
    void sscal(const float alpha, vector<float> &X);
    void dscal(const double alpha, vector<double> &X);

    /* [[UNUSED]] */
    void cscal(const cfloat alpha, vector <cfloat> &X);
    void zscal(const zdouble alpha, vector <zdouble> &X);

    void csscal(const float alpha, vector <cfloat> &X);
    void zdscal(const double alpha, vector <zdouble> &X);

/*
* ===========================================================================
* Prototypes for level 2 BLAS
* ===========================================================================
*/

    /* Routines with standard 4 prefixes (S, D, C, Z) */
    vector<float> sgemv(const enum ORDER order, const enum TRANSPOSE TransA,
                        const float alpha, const matrix<float> &A,
                        const vector<float> &X, const float beta);

    vector<float> sgbmv(const enum ORDER order, const enum TRANSPOSE TransA,
                        const int KL, const int KU,
                        const float alpha, const matrix<float> &A,
                        const vector<float> &X, const float beta);

    vector<float> strmv(const enum ORDER order, const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const matrix<float> &A, vector<float> &X);

    vector<float> stbmv(const enum ORDER order, const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const int K, const matrix<float> &A, vector<float> &X);

    vector<float> stpmv(const enum ORDER order, const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const matrix<float> &Ap, vector<float> &X);

    vector<float> strsv(const enum ORDER order, const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const matrix<float> &A, vector<float> &X);

    vector<float> stbsv(const enum ORDER order, const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const int K, const matrix<float> &A, vector<float> &X);

    vector<float> stpsv(const enum ORDER order, const enum UPLO Uplo,
                        const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const matrix<float> &Ap, vector<float> &X);

    vector<double> dgemv(const enum ORDER order, const enum TRANSPOSE TransA,
                         const double alpha, const matrix<double> &A,
                         const vector<double> &X, const double beta);

    vector<double> dgbmv(const enum ORDER order, const enum TRANSPOSE TransA,
                         const int KL, const int KU,
                         const double alpha, const matrix<double> &A,
                         const vector<double> &X, const double beta);

    vector<double> dtrmv(const enum ORDER order, const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const matrix<double> &A, vector<double> &X);

    vector<double> dtbmv(const enum ORDER order, const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const int K, const matrix<double> &A, vector<double> &X);

    vector<double> dtpmv(const enum ORDER order, const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const matrix<double> &Ap, vector<double> &X);

    vector<double> dtrsv(const enum ORDER order, const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const matrix<double> &A, vector<double> &X;

    vector<double> dtbsv(const enum ORDER order, const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const int K, const matrix<double> &A, vector<double> &X);

    vector<double> dtpsv(const enum ORDER order, const enum UPLO Uplo,
                         const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const matrix<double> &Ap, vector<double> &X);

    /* [[UNUSED]] */
    vector <cfloat> cgemv(const enum ORDER order, const enum TRANSPOSE TransA,
                          const cfloat alpha, const matrix <cfloat> &A,
                          const vector <cfloat> &X, const cfloat beta);

    vector <cfloat> cgbmv(const enum ORDER order, const enum TRANSPOSE TransA,
                          const int KL, const int KU,
                          const cfloat alpha, const matrix <cfloat> &A,
                          const vector <cfloat> &X, const cfloat beta);

    vector <cfloat> ctrmv(const enum ORDER order, const enum UPLO Uplo,
                          const enum TRANSPOSE TransA, const enum DIAG Diag,
                          const matrix <cfloat> &A, vector <cfloat> &X);

    vector <cfloat> ctbmv(const enum ORDER order, const enum UPLO Uplo,
                          const enum TRANSPOSE TransA, const enum DIAG Diag,
                          const int K, const matrix <cfloat> &A, vector <cfloat> &X);

    vector <cfloat> ctpmv(const enum ORDER order, const enum UPLO Uplo,
                          const enum TRANSPOSE TransA, const enum DIAG Diag,
                          const matrix <cfloat> &Ap, vector <cfloat> &X);

    vector <cfloat> ctrsv(const enum ORDER order, const enum UPLO Uplo,
                          const enum TRANSPOSE TransA, const enum DIAG Diag,
                          const matrix <cfloat> &A, vector <cfloat> &X);

    vector <cfloat> ctbsv(const enum ORDER order, const enum UPLO Uplo,
                          const enum TRANSPOSE TransA, const enum DIAG Diag,
                          const int K, const matrix <cfloat> &A, vector <cfloat> &X);

    vector <cfloat> ctpsv(const enum ORDER order, const enum UPLO Uplo,
                          const enum TRANSPOSE TransA, const enum DIAG Diag,
                          const matrix <cfloat> &Ap, vector <cfloat> &X);

    vector <zdouble> zgemv(const enum ORDER order, const enum TRANSPOSE TransA,
                           const zdouble alpha, const matrix <zdouble> &A,
                           const vector <zdouble> &X, const zdouble beta);

    vector <zdouble> zgbmv(const enum ORDER order, const enum TRANSPOSE TransA,
                           const int KL, const int KU,
                           const zdouble alpha, const matrix <zdouble> &A,
                           const vector <zdouble> &X, const zdouble beta);

    vector <zdouble> ztrmv(const enum ORDER order, const enum UPLO Uplo,
                           const enum TRANSPOSE TransA, const enum DIAG Diag,
                           const matrix <zdouble> &A, vector <zdouble> &X);

    vector <zdouble> ztbmv(const enum ORDER order, const enum UPLO Uplo,
                           const enum TRANSPOSE TransA, const enum DIAG Diag,
                           const int K, const matrix <zdouble> &A, vector <zdouble> &X);

    vector <zdouble> ztpmv(const enum ORDER order, const enum UPLO Uplo,
                           const enum TRANSPOSE TransA, const enum DIAG Diag,
                           const matrix <zdouble> &Ap, vector <zdouble> &X);

    vector <zdouble> ztrsv(const enum ORDER order, const enum UPLO Uplo,
                           const enum TRANSPOSE TransA, const enum DIAG Diag,
                           const matrix <zdouble> &A, vector <zdouble> &X;

    vector <zdouble> ztbsv(const enum ORDER order, const enum UPLO Uplo,
                           const enum TRANSPOSE TransA, const enum DIAG Diag,
                           const int K, const matrix <zdouble> &A, vector <zdouble> &X);

    vector <zdouble> ztpsv(const enum ORDER order, const enum UPLO Uplo,
                           const enum TRANSPOSE TransA, const enum DIAG Diag,
                           const matrix <zdouble> &Ap, vector <zdouble> &X);

    /* Routines with S and D prefixes only */
    vector<float> ssymv(const enum ORDER order, const enum UPLO Uplo,
                        const float alpha, const matrix<float> &A,
                        const vector<float> &X, const float beta);

    vector<float> ssbmv(const enum ORDER order, const enum UPLO Uplo,
                        const int K, const float alpha, const matrix<float> &A,
                        const vector<float> &X, const float beta);

    vector<float> sspmv(const enum ORDER order, const enum UPLO Uplo,
                        const float alpha, const matrix<float> &Ap,
                        const vector<float> &X, const float beta);

    matrix<float> sger(const enum ORDER order,
                       const float alpha, const vector<float> &X, const vector<float> &Y);

    matrix<float> ssyr(const enum ORDER order, const enum UPLO Uplo,
                       const float alpha, const vector<float> &X);

    matrix<float> sspr(const enum ORDER order, const enum UPLO Uplo,
                       const float alpha, const vector<float> &X);

    matrix<float> ssyr2(const enum ORDER order, const enum UPLO Uplo,
                        const float alpha, const vector<float> &X, const vector<float> &Y);

    matrix<double> dspr2(const enum ORDER order, const enum UPLO Uplo,
                         const double alpha, const vector<double> &X, const vector<double> &Y);

    vector<double> dsymv(const enum ORDER order, const enum UPLO Uplo,
                         const double alpha, const matrix<double> &A,
                         const vector<double> &X, const double beta);

    vector<double> dsbmv(const enum ORDER order, const enum UPLO Uplo,
                         const int K, const double alpha, const matrix<double> &A,
                         const vector<double> &X, const double beta);

    vector<double> dspmv(const enum ORDER order, const enum UPLO Uplo,
                         const double alpha, const matrix<double> &Ap,
                         const vector<double> &X, const double beta);

    matrix<double> dger(const enum ORDER order,
                        const double alpha, const vector<double> &X, const vector<double> &Y);

    matrix<double> dsyr(const enum ORDER order, const enum UPLO Uplo,
                        const double alpha, const vector<double> &X);

    matrix<double> dspr(const enum ORDER order, const enum UPLO Uplo,
                        const double alpha, const vector<double> &X);

    matrix<double> dsyr2(const enum ORDER order, const enum UPLO Uplo,
                         const double alpha, const vector<double> &X, const vector<double> &Y);

    matrix<double> dspr2(const enum ORDER order, const enum UPLO Uplo,
                         const double alpha, const vector<double> &X, const vector<double> &Y);

    /* Routines with C and Z prefixes only */

    /* [[UNUSED]] */
    vector <cfloat> csymv(const enum ORDER order, const enum UPLO Uplo,
                          const cfloat alpha, const matrix <cfloat> &A,
                          const vector <cfloat> &X, const cfloat beta);

    vector <cfloat> csbmv(const enum ORDER order, const enum UPLO Uplo,
                          const int K, const cfloat alpha, const matrix <cfloat> &A,
                          const vector <cfloat> &X, const cfloat beta);

    vector <cfloat> cspmv(const enum ORDER order, const enum UPLO Uplo,
                          const cfloat alpha, const matrix <cfloat> &Ap,
                          const vector <cfloat> &X, const cfloat beta);

    matrix <cfloat> cger(const enum ORDER order,
                         const cfloat alpha, const vector <cfloat> &X, const vector <cfloat> &Y);

    matrix <cfloat> csyr(const enum ORDER order, const enum UPLO Uplo,
                         const cfloat alpha, const vector <cfloat> &X);

    matrix <cfloat> cspr(const enum ORDER order, const enum UPLO Uplo,
                         const cfloat alpha, const vector <cfloat> &X);

    matrix <cfloat> csyr2(const enum ORDER order, const enum UPLO Uplo,
                          const cfloat alpha, const vector <cfloat> &X, const vector <cfloat> &Y);

    matrix <zdouble> zspr2(const enum ORDER order, const enum UPLO Uplo,
                           const zdouble alpha, const vector <zdouble> &X, const vector <zdouble> &Y);

    vector <zdouble> zsymv(const enum ORDER order, const enum UPLO Uplo,
                           const zdouble alpha, const matrix <zdouble> &A,
                           const vector <zdouble> &X, const zdouble beta);

    vector <zdouble> zsbmv(const enum ORDER order, const enum UPLO Uplo,
                           const int K, const zdouble alpha, const matrix <zdouble> &A,
                           const vector <zdouble> &X, const zdouble beta);

    vector <zdouble> zspmv(const enum ORDER order, const enum UPLO Uplo,
                           const zdouble alpha, const matrix <zdouble> &Ap,
                           const vector <zdouble> &X, const zdouble beta);

    matrix <zdouble> zger(const enum ORDER order,
                          const zdouble alpha, const vector <zdouble> &X, const vector <zdouble> &Y);

    matrix <zdouble> zsyr(const enum ORDER order, const enum UPLO Uplo,
                          const zdouble alpha, const vector <zdouble> &X);

    matrix <zdouble> zspr(const enum ORDER order, const enum UPLO Uplo,
                          const zdouble alpha, const vector <zdouble> &X);

    matrix <zdouble> zsyr2(const enum ORDER order, const enum UPLO Uplo,
                           const zdouble alpha, const vector <zdouble> &X, const vector <zdouble> &Y);

    matrix <zdouble> zspr2(const enum ORDER order, const enum UPLO Uplo,
                           const zdouble alpha, const vector <zdouble> &X, const vector <zdouble> &Y);

    /* [[UNUSED]] */
    void zhemv(const enum ORDER order, const enum UPLO Uplo,
               const int N, const void *alpha, const void *A,
               const int lda, const void *X, const int incX,
               const void *beta, void *Y, const int incY);

    void zhbmv(const enum ORDER order, const enum UPLO Uplo,
               const int N, const int K, const void *alpha, const void *A,
               const int lda, const void *X, const int incX,
               const void *beta, void *Y, const int incY);

    void zhpmv(const enum ORDER order, const enum UPLO Uplo,
               const int N, const void *alpha, const void *Ap,
               const void *X, const int incX,
               const void *beta, void *Y, const int incY);

    void zgeru(const enum ORDER order, const int M, const int N,
               const void *alpha, const void *X, const int incX,
               const void *Y, const int incY, void *A, const int lda);

    void zgerc(const enum ORDER order, const int M, const int N,
               const void *alpha, const void *X, const int incX,
               const void *Y, const int incY, void *A, const int lda);

    void zher(const enum ORDER order, const enum UPLO Uplo,
              const int N, const double alpha, const void *X, const int incX,
              void *A, const int lda);

    void zhpr(const enum ORDER order, const enum UPLO Uplo,
              const int N, const double alpha, const void *X,
              const int incX, void *A);

    void zher2(const enum ORDER order, const enum UPLO Uplo, const int N,
               const void *alpha, const void *X, const int incX,
               const void *Y, const int incY, void *A, const int lda);

    void zhpr2(const enum ORDER order, const enum UPLO Uplo, const int N,
               const void *alpha, const void *X, const int incX,
               const void *Y, const int incY, void *Ap);

/*
* ===========================================================================
* Prototypes for level 3 BLAS
* ===========================================================================
*/

    /* Routines with standard 4 prefixes (S, D, C, Z) */
    matrix<float> sgemm(const enum ORDER Order, const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                        const float alpha, const matrix<float> &A,
                        const matrix<float> &B, const float beta);

    matrix<float> ssymm(const enum ORDER Order, const enum SIDE Side, const enum UPLO Uplo,
                        const float alpha, const matrix<float> &A,
                        const matrix<float> &B, const float beta);

    matrix<float> ssyrk(const enum ORDER Order, const enum UPLO Uplo,
                        const enum TRANSPOSE Trans, const int K,
                        const float alpha, const matrix<float> &A, const float beta);

    matrix<float> ssyr2k(const enum ORDER Order, const enum UPLO Uplo,
                         const enum TRANSPOSE Trans, const int K,
                         const float alpha, const matrix<float> &A,
                         const matrix<float> &B, const float beta);

    matrix<float> strmm(const enum ORDER Order, const enum SIDE Side,
                        const enum UPLO Uplo, const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const float alpha, const matrix<float> &A);

    matrix<float> strsm(const enum ORDER Order, const enum SIDE Side,
                        const enum UPLO Uplo, const enum TRANSPOSE TransA, const enum DIAG Diag,
                        const float alpha, const matrix<float> &A);

    matrix<double> dgemm(const enum ORDER Order, const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                         const double alpha, const matrix<double> &A,
                         const matrix<double> &B, const double beta);

    matrix<double> dsymm(const enum ORDER Order, const enum SIDE Side, const enum UPLO Uplo,
                         const double alpha, const matrix<double> &A,
                         const matrix<double> &B, const double beta);

    matrix<double> dsyrk(const enum ORDER Order, const enum UPLO Uplo,
                         const enum TRANSPOSE Trans, const int K,
                         const double alpha, const matrix<double> &A, const double beta);

    matrix<double> dsyr2k(const enum ORDER Order, const enum UPLO Uplo,
                          const enum TRANSPOSE Trans, const int K,
                          const double alpha, const matrix<double> &A,
                          const matrix<double> &B, const double beta);

    matrix<double> dtrmm(const enum ORDER Order, const enum SIDE Side,
                         const enum UPLO Uplo, const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const double alpha, const matrix<double> &A);

    matrix<double> dtrsm(const enum ORDER Order, const enum SIDE Side,
                         const enum UPLO Uplo, const enum TRANSPOSE TransA, const enum DIAG Diag,
                         const double alpha, const matrix<double> &A);

    /* [[UNUSED]] */
    matrix <cfloat> cgemm(const enum ORDER Order, const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                          const cfloat alpha, const matrix <cfloat> &A,
                          const matrix <cfloat> &B, const cfloat beta);

    matrix <cfloat> csymm(const enum ORDER Order, const enum SIDE Side, const enum UPLO Uplo,
                          const cfloat alpha, const matrix <cfloat> &A,
                          const matrix <cfloat> &B, const cfloat beta);

    matrix <cfloat> csyrk(const enum ORDER Order, const enum UPLO Uplo,
                          const enum TRANSPOSE Trans, const int K,
                          const cfloat alpha, const matrix <cfloat> &A, const cfloat beta);

    matrix <cfloat> csyr2k(const enum ORDER Order, const enum UPLO Uplo,
                           const enum TRANSPOSE Trans, const int K,
                           const cfloat alpha, const matrix <cfloat> &A,
                           const matrix <cfloat> &B, const cfloat beta);

    matrix <cfloat> ctrmm(const enum ORDER Order, const enum SIDE Side,
                          const enum UPLO Uplo, const enum TRANSPOSE TransA, const enum DIAG Diag,
                          const cfloat alpha, const matrix <cfloat> &A);

    matrix <cfloat> ctrsm(const enum ORDER Order, const enum SIDE Side,
                          const enum UPLO Uplo, const enum TRANSPOSE TransA, const enum DIAG Diag,
                          const cfloat alpha, const matrix <cfloat> &A);

    /* [[UNUSED]] */
    matrix <zdouble> zgemm(const enum ORDER Order, const enum TRANSPOSE TransA, const enum TRANSPOSE TransB,
                           const zdouble alpha, const matrix <zdouble> &A,
                           const matrix <zdouble> &B, const zdouble beta);

    matrix <zdouble> zsymm(const enum ORDER Order, const enum SIDE Side, const enum UPLO Uplo,
                           const zdouble alpha, const matrix <zdouble> &A,
                           const matrix <zdouble> &B, const zdouble beta);

    matrix <zdouble> zsyrk(const enum ORDER Order, const enum UPLO Uplo,
                           const enum TRANSPOSE Trans, const int K,
                           const zdouble alpha, const matrix <zdouble> &A, const zdouble beta);

    matrix <zdouble> zsyr2k(const enum ORDER Order, const enum UPLO Uplo,
                            const enum TRANSPOSE Trans, const int K,
                            const zdouble alpha, const matrix <zdouble> &A,
                            const matrix <zdouble> &B, const zdouble beta);

    matrix <zdouble> ztrmm(const enum ORDER Order, const enum SIDE Side,
                           const enum UPLO Uplo, const enum TRANSPOSE TransA, const enum DIAG Diag,
                           const zdouble alpha, const matrix <zdouble> &A);

    matrix <zdouble> ztrsm(const enum ORDER Order, const enum SIDE Side,
                           const enum UPLO Uplo, const enum TRANSPOSE TransA, const enum DIAG Diag,
                           const zdouble alpha, const matrix <zdouble> &A);

    /* Routines with prefixes C and Z only */

    /* [[UNUSED]] */
    matrix <cfloat> &chemm(const enum ORDER Order, const enum SIDE Side, const enum UPLO Uplo,
                           const cfloat alpha, const matrix <cfloat> &A,
                           const matrix <cfloat> &B, const cfloat beta);

    matrix <cfloat> cherk(const enum ORDER Order, const enum UPLO Uplo, const enum TRANSPOSE Trans, const int K,
                          const float alpha, const matrix <cfloat> &A, const float beta);

    matrix <cfloat> cher2k(const enum ORDER Order, const enum UPLO Uplo, const enum TRANSPOSE Trans, const int K,
                           const cfloat alpha, const matrix <cfloat> &A,
                           const matrix <cfloat> &B, const float beta);

    /* [[UNUSED]] */
    matrix <zdouble> zhemm(const enum ORDER Order, const enum SIDE Side, const enum UPLO Uplo,
                           const zdouble alpha, const matrix <zdouble> &A,
                           const matrix <zdouble> &B, const zdouble beta);

    matrix <zdouble> zherk(const enum ORDER Order, const enum UPLO Uplo, const enum TRANSPOSE Trans, const int K,
                           const double alpha, const matrix <zdouble> &A, const double beta);

    matrix <zdouble> zher2k(const enum ORDER Order, const enum UPLO Uplo, const enum TRANSPOSE Trans, const int K,
                            const zdouble alpha, const matrix <zdouble> &A,
                            const matrix <zdouble> &B, const double beta);
};

}
