#pragma once

#include <boost/core/demangle.hpp>
#include <cstdint>

namespace scylla_blas {

using index_type = int64_t;

template<class T>
static std::string get_type_name() {
    return boost::core::demangle(typeid(T).name());
}

}
