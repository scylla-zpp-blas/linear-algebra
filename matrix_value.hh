#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

#include "scylla_types.hh"

namespace scylla_blas {
    template <class V>
    struct matrix_value {
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
        return i == other.i && j == other.j && abs(val - other.val < 1e-9);
    }

    template <typename T>
    inline std::ostream& operator<<(std::ostream& out, const matrix_value<T>& value) {
        out << std::string("(") << value.i << ", " << value.j << ") -> " << value.val;
        return out;
    }
}