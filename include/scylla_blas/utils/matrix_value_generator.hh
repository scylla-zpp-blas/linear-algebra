#pragma once

#include "scylla_blas/structure/matrix_value.hh"

namespace scylla_blas {

template<class V>
class matrix_value_generator {
public:

    virtual bool has_next() = 0;

    virtual matrix_value<V> next() = 0;

    virtual size_t height() = 0;

    virtual size_t width() = 0;

    virtual ~matrix_value_generator() = default;
};

}