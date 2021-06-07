#pragma once

#include "utils/scylla_types.hh"

constexpr scylla_blas::index_t DEFAULT_BLOCK_SIZE = (1 << 2);
constexpr int64_t DEFAULT_WORKER_COUNT = 4;
constexpr int64_t DEFAULT_SCHEDULER_SLEEP_TIME_MICROSECONDS = 20000;

constexpr int64_t DEFAULT_WORKER_SLEEP_TIME_MICROSECONDS = 20000;
constexpr int64_t DEFAULT_MAX_WORKER_RETRIES = 5;

constexpr uint16_t SCYLLA_DEFAULT_PORT = 9042;
constexpr id_t HELPER_FLOAT_VECTOR_ID = 0;
constexpr id_t HELPER_DOUBLE_VECTOR_ID = 1;
constexpr id_t DEFAULT_WORKER_QUEUE_ID = 0;


/* Entries below this value preferably won't be stored in our structures */
#define EPSILON (1e-7)