#pragma once

#include "scylla_blas/utils/scylla_types.hh"

/* This header defines tasks types that can be requested from,
 * read and performed by worker instances.
 */

namespace scylla_blas::proto {

/* TODO: narrow the list down. It includes all BLAS task ids,
 * but it is likely we won't need most of them – if anything,
 * because we will want workers to know of and perform only
 * very simple tasks such as multiplication or addition.
 * The more complex ones could be left for the supervisor layer
 * to plan and distribute.
 */
enum task_type {
    NONE,

    /* LEVEL 1 */
    SROTG,
    SROTMG,
    SROT,
    SROTM,
    SSWAP,
    SSCAL,
    SCOPY,
    SAXPY,
    SDOT,
    SDSDOT,
    SNRM2,
    SCNRM2,
    SASUM,
    ISAMAX,

    DROTG,
    DROTMG,
    DROT,
    DROTM,
    DSWAP,
    DSCAL,
    DCOPY,
    DAXPY,
    DDOT,
    DSDOT,
    DNRM2,
    DZNRM2,
    DASUM,
    IDAMAX,

    CROTG,
    CSROT,
    CSWAP,
    CSCAL,
    CSSCAL,
    CCOPY,
    CAXPY,
    CDOTU,
    CDOTC,
    SCASUM,
    ICAMAX,

    ZROTG,
    ZDROTF,
    ZSWAP,
    ZSCAL,
    ZDSCAL,
    ZCOPY,
    ZAXPY,
    ZDOTU,
    ZDOTC,
    DZASUM,
    IZAMAX,

    /* LEVEL 2 */
    SGEMV,
    SGBMV,
    SSYMV,
    SSBMV,
    SSPMV,
    STRMV,
    STBMV,
    STPMV,
    STRSV,
    STBSV,
    STPSV,
    SGER,
    SSYR,
    SSPR,
    SSYR2,
    SSPR2,

    DGEMV,
    DGBMV,
    DSYMV,
    DSBMV,
    DSPMV,
    DTRMV,
    DTBMV,
    DTPMV,
    DTRSV,
    DTBSV,
    DTPSV,
    DGER,
    DSYR,
    DSPR,
    DSYR2,
    DSPR2,

    CGEMV,
    CGBMV,
    CHEMV,
    CHBMV,
    CHPMV,
    CTRMV,
    CTBMV,
    CTPMV,
    CTRSV,
    CTBSV,
    CGERU,
    CGERC,
    CHER,
    CHPR,
    CHER2,
    CHPR2,

    ZGEMV,
    ZGBMV,
    ZHEMV,
    ZHBMV,
    ZHPMV,
    ZTRMV,
    ZTBMV,
    ZTPMV,
    ZTRSV,
    ZTBSV,
    ZGERU,
    ZGERC,
    ZHER,
    ZHPR,
    ZHER2,
    ZHPR2,

    /* LEVEL 3 */
    SGEMM,
    SSYMM,
    SSYRK,
    SSYR2K,
    STRMM,
    STRSM,

    DGEMM,
    DSYMM,
    DSYRK,
    DSYR2K,
    DTRMM,
    DTRSM,

    CGEMM,
    CSYMM,
    CHEMM,
    CSYRK,
    CHERK,
    CSYR2K,
    CHER2K,
    CTRMM,
    CTRSM,

    ZGEMM,
    ZSYMM,
    ZHEMM,
    ZSYRK,
    ZHERK,
    ZSYR2K,
    ZHER2K,
    ZTRMM,
    ZTRSM
};

/* This is the struct that will be sent trough the queue.
 * We can freely modify it, to add different kinds of tasks.
 * Instance of this struct will be cast to char array,
 * stored as binary blob in the database, then "de-serialized" at the other end.
 * This means it probably should not contain data pertaining to local memory,
 * as there is little point in storing such data.
 */
struct task {
    task_type type;

    union {
        index_t index;

        struct {
            int64_t data;
        } basic;

        struct {
            index_t block_row;
            index_t block_column;
        } coord;

        struct {
            int64_t task_queue_id;
            int64_t X_id;
            int64_t Y_id;
        } vector_task_simple;

        struct {
            int64_t task_queue_id;
            float alpha;
            int64_t X_id;
            int64_t Y_id;
        } vector_task_float;

        struct {
            int64_t task_queue_id;
            double alpha;
            int64_t X_id;
            int64_t Y_id;
        } vector_task_double;

        struct {
            int64_t task_queue_id;

            index_t KL, KU;

            UPLO Uplo;
            DIAG Diag;

            int64_t A_id;
            TRANSPOSE TransA;
            float alpha;

            int64_t X_id;
            float beta;

            int64_t Y_id;
        } mixed_task_float;

        struct {
            int64_t task_queue_id;

            index_t KL, KU;

            int64_t A_id;
            TRANSPOSE TransA;
            double alpha;

            int64_t X_id;
            double beta;

            int64_t Y_id;
        } mixed_task_double;

        struct {
            int64_t task_queue_id;

            int64_t A_id;
            TRANSPOSE TransA;
            float alpha;

            int64_t B_id;
            TRANSPOSE TransB;
            float beta;

            int64_t C_id;
        } matrix_task_float;

        struct {
            int64_t task_queue_id;

            int64_t A_id;
            TRANSPOSE TransA;
            double alpha;

            int64_t B_id;
            TRANSPOSE TransB;
            double beta;

            int64_t C_id;
        } matrix_task_double;
    };

};

enum response_type {
    R_NONE,
    R_SOME,
    R_INT64
};

struct response {
    response_type type;

    union {
        float result_float;
        double result_double;

        struct {
            index_t index;
            float value;
        } result_max_float_index;

        struct { 
            float first;
            float second;
        } result_float_pair;
        
        struct {
            double first;
            double second;
        } result_double_pair;

        struct {
            index_t index;
            double value;
        } result_max_double_index;

        struct {
            int64_t response;
        } simple;
    };
};

}