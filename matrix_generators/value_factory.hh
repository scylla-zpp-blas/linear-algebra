#pragma once

#include <random>

template<typename T>
class value_factory {
public:
    virtual T next() = 0;
};