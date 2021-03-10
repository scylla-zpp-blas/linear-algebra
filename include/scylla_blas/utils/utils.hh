#pragma once

#include <chrono>

namespace scylla_blas {

const double epsilon = 1e-9;

inline int64_t get_timestamp() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

}
