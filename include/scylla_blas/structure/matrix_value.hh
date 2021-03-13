#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

#include "scylla_blas/utils/scylla_types.hh"
#include "scylla_blas/utils/utils.hh"

namespace scylla_blas {

template <class V>
struct matrix_value {
    /* Top-left matrix cell coordinates are defined as (row_index, col_index) = (1, 1). */
    index_type row_index, col_index;
    V value;

    matrix_value(index_type i, index_type j, V val) : row_index(i), col_index(j), value(val) {}

    bool operator==(const matrix_value<V> &other) const {
        return row_index == other.row_index && col_index == other.col_index && value == other.value;
    }

    bool operator!=(const matrix_value<V> &other) const {
        return *this != other;
    }

    template <typename T>
    friend std::ostream& operator<<(std::ostream& out, const matrix_value<T>& value);
};

template<>
inline bool matrix_value<float>::operator==(const matrix_value<float> &other) const {
    return row_index == other.row_index && col_index == other.col_index && std::abs(value - other.value) < scylla_blas::epsilon;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& out, const matrix_value<T>& value) {
    out << std::string("(") << value.row_index << ", " << value.col_index << ") -> " << value.value;
    return out;
}

}