#pragma once

#include "utils/scylla_types.hh"

constexpr uint16_t SCYLLA_DEFAULT_PORT = 9042;
constexpr scylla_blas::index_type MATRIX_BLOCK_SIZE = (1 << 8);