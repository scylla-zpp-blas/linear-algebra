#pragma once

#include <bits/exception.h>
#include "scylla_matrix.hh"
#include "matrix_value_generator.hh"

class scylla_blas {
    template <class T>
    static scylla_matrix<T> multiply(matrix_value_generator<T> first, matrix_value_generator<T> second) {
        throw std::exception();
    };
};
