#pragma once

#include "scylla_blas/utils/scylla_types.hh"

namespace scylla_blas {

template<class T>
struct vector_value {
    index_t index;
    T value;

    constexpr vector_value(const index_t index, const T value) : index(index), value(value) { }
    constexpr explicit vector_value(const std::pair<index_t, T>& p) : index(p.first), value(p.second) { }
};

}
