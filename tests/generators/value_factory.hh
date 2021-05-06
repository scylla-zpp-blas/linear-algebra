#pragma once

#include <random>

namespace scylla_blas {

template<typename T>
class value_factory {
public:
    virtual T next() = 0;
};

}