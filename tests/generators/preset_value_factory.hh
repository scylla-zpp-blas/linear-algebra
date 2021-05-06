#pragma once

#include <random>
#include "value_factory.hh"

namespace scylla_blas {

template<typename T>
class preset_value_factory : public value_factory<T> {
private:
    std::vector<T> _values;
    int _iterator;
public:
    explicit preset_value_factory(std::vector<T> values): _values(values), _iterator(0) {}

    T next() override {
        T ret = _values[_iterator];
        _iterator = (_iterator + 1) % _values.size();
        return ret;
    }
};

}