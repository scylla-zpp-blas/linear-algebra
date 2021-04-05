#pragma once

#include "utils/scylla_types.hh"

constexpr uint16_t SCYLLA_DEFAULT_PORT = 9042;
constexpr scylla_blas::index_type BLOCK_SIZE = (1 << 2);

constexpr int64_t DEFAULT_WORKER_QUEUE_ID = 0;
constexpr int64_t WORKER_SLEEP_TIME_SECONDS = 1;
constexpr int64_t LIMIT_WORKER_CONCURRENCY = 4;
constexpr int64_t MAX_WORKER_RETRIES = 5;

/* Entries below this value won't be stored in our structures */
#define EPSILON (1e-7)