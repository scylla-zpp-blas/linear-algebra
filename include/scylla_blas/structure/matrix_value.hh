#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

#include "scylla_blas/config.hh"
#include "scylla_blas/utils/scylla_types.hh"
#include "scylla_blas/utils/utils.hh"

namespace scylla_blas {

template <class T>
struct matrix_value {
    /* Top-left matrix cell coordinates are defined as (row_index, col_index) = (1, 1). */
    index_type row_index, col_index;
    T value;

    matrix_value(index_type i, index_type j, T val) : row_index(i), col_index(j), value(val) {}

    bool operator==(const matrix_value &other) const {
        return row_index == other.row_index && col_index == other.col_index && (std::abs(value - other.value) < EPSILON);
    }

    bool operator!=(const matrix_value &other) const {
        return !(*this == other);
    }

    template <class U>
    friend std::ostream& operator<<(std::ostream& out, const matrix_value<U>& value);
};

}