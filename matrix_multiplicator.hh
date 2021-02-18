#pragma once

#include <cstddef>
#include <utility>
#include "scylla_matrix.hh"

template <class T>
class matrix_multiplicator {
public:
    /* Multiplies two matrices loaded into Scylla in TODO representation. */
    virtual void multiply(size_t first_id, size_t second_id, size_t result_id) = 0;

    virtual scylla_matrix<T> multiply(scylla_matrix<T> first, scylla_matrix<T> second) = 0;

    virtual ~matrix_multiplicator() {};
};
