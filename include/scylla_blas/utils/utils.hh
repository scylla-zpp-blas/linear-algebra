#pragma once

#include <boost/thread/thread.hpp>
#include <chrono>

namespace scylla_blas {

const double epsilon = 1e-9;

inline int64_t get_timestamp() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline void wait_seconds(int64_t count) {
    boost::this_thread::sleep(boost::posix_time::seconds(count));
}

}
