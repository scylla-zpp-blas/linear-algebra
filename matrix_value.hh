#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

template <class V>
struct matrix_value {
    size_t i, j;
    V val;

    matrix_value(size_t i, size_t j, V val) : i(i), j(j), val(val) {}

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
bool matrix_value<float>::operator==(const matrix_value<float> &other) const {
    return i == other.i && j == other.j && abs(val - other.val < 1e-9);
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const matrix_value<T>& value) {
    out << std::string("(") << value.i << ", " << value.j << ") -> " << value.val;
    return out;
}