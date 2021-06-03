#pragma once

#include "scylla_blas/utils/scylla_types.hh"

constexpr scylla_blas::index_t base_id = 0x42454e4348 << 24;
constexpr scylla_blas::index_t l_matrix_id = base_id + 1;
constexpr scylla_blas::index_t r_matrix_id = base_id + 2;
constexpr scylla_blas::index_t w_matrix_id = base_id + 3;
constexpr scylla_blas::index_t l_vector_id = base_id + 4;
constexpr scylla_blas::index_t r_vector_id = base_id + 5;
constexpr scylla_blas::index_t w_vector_id = base_id + 6;
constexpr int RANDOM_SEED = 0x1337;