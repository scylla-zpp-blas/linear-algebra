#pragma once

class test_const {
public:
    const static inline scylla_blas::index_type float_matrix_AxB_id = 1;
    const static inline scylla_blas::index_type float_matrix_BxA_id = 2;
    const static inline scylla_blas::index_type float_matrix_BxB_id = 3;

    const static inline scylla_blas::index_type double_matrix_AxB_id = 11;
    const static inline scylla_blas::index_type double_matrix_BxA_id = 12;
    const static inline scylla_blas::index_type double_matrix_BxB_id = 13;

    const static inline scylla_blas::index_type test_vector_len = 2 * BLOCK_SIZE + 3;

    const static inline scylla_blas::index_type float_vector_1_id = 1;
    const static inline scylla_blas::index_type float_vector_2_id = 2;
    const static inline scylla_blas::index_type float_vector_3_id = 3;

    const static inline scylla_blas::index_type double_vector_1_id = 11;
    const static inline scylla_blas::index_type double_vector_2_id = 12;
    const static inline scylla_blas::index_type double_vector_3_id = 13;
};