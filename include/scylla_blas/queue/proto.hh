#pragma once

#include "scylla_blas/utils/scylla_types.hh"

namespace scylla_blas::proto {

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

// This is the struct that will be sent trough the queue.
// We can freely modify it, to add different kinds of tasks.
// Instance of this struct will be cast to char array,
// stored as binary blob in the database,
// then "de-serialized" at the other end.
// This means it probably should not contain data pertaining to local memory,
// as storing such data is nearly useless.
struct task {

    task_type type;

    union {
        struct {
            int64_t data;
        } basic;

        struct {
            index_type block_row;
            index_type block_column;
        } coord;

        struct {
            int64_t task_queue_id;
            int64_t obj_id;
        } blas_auto;

        struct {
            int64_t task_queue_id;
            int64_t A_id;
            int64_t B_id;
        } blas_unary;

        struct {
            int64_t task_queue_id;
            int64_t A_id;
            int64_t B_id;
            int64_t C_id;
        } blas_binary;
    };

};

}