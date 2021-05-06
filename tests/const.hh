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

    const static inline scylla_blas::index_type float_vector_indexes[] = {
            1,
            2,
            3
    };
    const static inline std::size_t float_vector_indexes_size =
            std::size(float_vector_indexes);
    static scylla_blas::index_type getScyllaIndexOfFloatVector(std::size_t index) {
        if (index >= float_vector_indexes_size)
            throw std::runtime_error("Index out of bounds.");
        return float_vector_indexes[index];
    }

    const static inline scylla_blas::index_type double_vector_indexes[] = {
            11,
            12,
            13
    };
    const static inline std::size_t double_vector_indexes_size =
            std::size(double_vector_indexes);
    static scylla_blas::index_type getScyllaIndexOfDoubleVector(std::size_t index) {
        if (index >= double_vector_indexes_size)
            throw std::runtime_error("Index out of bounds.");
        return double_vector_indexes[index];
    }

};