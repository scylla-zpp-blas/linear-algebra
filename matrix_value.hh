#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

#include "utils/scylla_types.hh"
#include "utils/utils.hh"

namespace scylla_blas {

template <class V>
struct matrix_value {
    /* (i, j) = position of the cell containing the value in the matrix.
     * By default, (1, 1)-based indexing is used.
     * The first coordinate corresponds to the vertical (Y) axis,
     * the second – to the horizontal (X) axis.
     *
     * This convention corresponds to typical matrix indexing.
     */
    index_type i, j;
    V val;

    matrix_value(index_type i, index_type j, V val) : i(i), j(j), val(val) {}

    bool operator==(const matrix_value<V> &other) const {
        return i == other.i && j == other.j && val == other.val;
    }

    bool operator!=(const matrix_value<V> &other) const {
        return !(*this == other);
    }

    template <typename T>
    friend std::ostream& operator<<(std::ostream& out, const matrix_value<T>& value);
};

template<>
inline bool matrix_value<float>::operator==(const matrix_value<float> &other) const {
    return i == other.i && j == other.j && abs(val - other.val < scylla_blas::epsilon);
}

template <typename T>
inline std::ostream& operator<<(std::ostream& out, const matrix_value<T>& value) {
    out << std::string("(") << value.i << ", " << value.j << ") -> " << value.val;
    return out;
}

}