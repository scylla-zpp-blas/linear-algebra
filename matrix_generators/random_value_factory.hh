#pragma once

#include <random>
#include "value_factory.hh"

template<typename T>
class random_value_factory : public value_factory<T> {
private:
    std::mt19937 _rng;
    T _min, _max;
    std::uniform_real_distribution<T> _dist;

public:
    random_value_factory(T min, T max, int seed) {
        this->_min = min;
        this->_max = max;
        this->_rng = std::mt19937(seed);
        this->_dist = std::uniform_real_distribution<T>(min, max);
    }

    T next() override {
        return _dist(_rng);
    }
};
