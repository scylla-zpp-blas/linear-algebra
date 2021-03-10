#pragma once

#include <random>

namespace scylla_blas {

template<typename T>
class value_factory {
private:
    std::mt19937 _rng;
    T _min, _max;
    std::uniform_real_distribution<T> _dist;

public:
    value_factory(T min, T max, int seed) {
        this->_min = min;
        this->_max = max;
        this->_rng = std::mt19937(seed);
        this->_dist = std::uniform_real_distribution<T>(min, max);
    }

    T next() {
        return _dist(_rng);
    }
};

}