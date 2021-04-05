#pragma once

#include "scylla_blas/utils/scylla_types.hh"

namespace scylla_blas {

template<class T>
struct vector_value {
    index_type index;
    T value;

    constexpr vector_value(const index_type index, const T value) : index(index), value(value) { }
    constexpr explicit vector_value(const std::pair<index_type, T>& p) : index(p.first), value(p.second) { }
};

}
