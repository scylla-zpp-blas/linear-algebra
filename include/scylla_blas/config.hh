#pragma once

#include "utils/scylla_types.hh"

constexpr uint16_t SCYLLA_DEFAULT_PORT = 9042;
constexpr scylla_blas::index_type BLOCK_SIZE = (1 << 5);

constexpr int64_t DEFAULT_WORKER_QUEUE_ID = 0;
constexpr int64_t WORKER_SLEEP_TIME_SECONDS = 10;
constexpr int64_t LIMIT_WORKER_CONCURRENCY = 4;

// TODO: allow for individual dimensions for each matrix
constexpr scylla_blas::index_type MATRIX_BLOCK_WIDTH = 5;
constexpr scylla_blas::index_type MATRIX_BLOCK_HEIGHT = 5;