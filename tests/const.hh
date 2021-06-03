#pragma once

struct vector_props {
    vector_props(scylla_blas::index_t id,
                 scylla_blas::index_t size) : id(id), size(size) {}

    scylla_blas::index_t id;
    scylla_blas::index_t size;
};

class test_const {
public:
    const static inline scylla_blas::index_t float_matrix_AxB_id = 1000 + 1;
    const static inline scylla_blas::index_t float_matrix_BxA_id = 1000 + 2;
    const static inline scylla_blas::index_t float_matrix_BxB_id = 1000 + 3;

    const static inline scylla_blas::index_t double_matrix_AxB_id = 1000 + 11;
    const static inline scylla_blas::index_t double_matrix_BxA_id = 1000 + 12;
    const static inline scylla_blas::index_t double_matrix_BxB_id = 1000 + 13;

    /* Dimensions of test containers in fixtures */
    const static inline scylla_blas::index_t matrix_A = 2 * DEFAULT_BLOCK_SIZE + 3;
    const static inline scylla_blas::index_t matrix_B = 2 * DEFAULT_BLOCK_SIZE + 6;
    const static inline scylla_blas::index_t test_vector_len_A = matrix_A;
    const static inline scylla_blas::index_t test_vector_len_B = matrix_B;

    const static inline scylla_blas::index_t float_vector_1_id = 1000 + 1;
    const static inline scylla_blas::index_t float_vector_2_id = 1000 + 2;
    const static inline scylla_blas::index_t float_vector_3_id = 1000 + 3;
    const static inline scylla_blas::index_t float_vector_4_id = 1000 + 4;

    const static inline scylla_blas::index_t double_vector_1_id = 1000 + 11;
    const static inline scylla_blas::index_t double_vector_2_id = 1000 + 12;
    const static inline scylla_blas::index_t double_vector_3_id = 1000 + 13;
    const static inline scylla_blas::index_t double_vector_4_id = 1000 + 14;

    const static inline vector_props float_vector_props[] = {
            vector_props(float_vector_1_id, test_vector_len_A),
            vector_props(float_vector_2_id, test_vector_len_A),
            vector_props(float_vector_3_id, test_vector_len_B),
            vector_props(float_vector_4_id, test_vector_len_B)
    };

    const static inline vector_props double_vector_props[] = {
            vector_props(double_vector_1_id, test_vector_len_A),
            vector_props(double_vector_2_id, test_vector_len_A),
            vector_props(double_vector_3_id, test_vector_len_B),
            vector_props(double_vector_4_id, test_vector_len_B)
    };

};