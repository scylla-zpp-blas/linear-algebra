#pragma once

#include <thread>
#include <chrono>

namespace scylla_blas {

const double epsilon = 1e-9;

inline int64_t get_timestamp() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline void wait_seconds(int64_t count) {
    std::this_thread::sleep_for(std::chrono::seconds(count));
}

}
