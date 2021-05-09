#pragma once

class test_const {
public:
    const static inline scylla_blas::index_type float_matrix_AxB_id = 1000 + 1;
    const static inline scylla_blas::index_type float_matrix_BxA_id = 1000 + 2;
    const static inline scylla_blas::index_type float_matrix_BxB_id = 1000 + 3;

    const static inline scylla_blas::index_type double_matrix_AxB_id = 1000 + 11;
    const static inline scylla_blas::index_type double_matrix_BxA_id = 1000 + 12;
    const static inline scylla_blas::index_type double_matrix_BxB_id = 1000 + 13;

    /* Dimensions of test containers in fixtures */
    const static inline scylla_blas::index_type matrix_A = 2 * BLOCK_SIZE + 3;
    const static inline scylla_blas::index_type matrix_B = 2 * BLOCK_SIZE + 6;
    const static inline scylla_blas::index_type test_vector_len_A = matrix_A;
    const static inline scylla_blas::index_type test_vector_len_B = matrix_B;

    const static inline scylla_blas::index_type float_vector_1_id = 1000 + 1;
    const static inline scylla_blas::index_type float_vector_2_id = 1000 + 2;
    const static inline scylla_blas::index_type float_vector_3_id = 1000 + 3;
    const static inline scylla_blas::index_type float_vector_4_id = 1000 + 4;

    const static inline scylla_blas::index_type double_vector_1_id = 1000 + 11;
    const static inline scylla_blas::index_type double_vector_2_id = 1000 + 12;
    const static inline scylla_blas::index_type double_vector_3_id = 1000 + 13;
    const static inline scylla_blas::index_type double_vector_4_id = 1000 + 14;
};