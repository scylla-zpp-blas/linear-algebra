#pragma once

#include <random>
#include <memory>
#include "matrix_value_generator.hh"
#include "value_factory.hh"
#include "utils/int_math.hh"

class no_next_value_exception: public std::exception {
    const char* what() const noexcept override {
        return "no_next_value_exception";
    }
};

template <class V>
class sparse_matrix_value_generator : public matrix_value_generator<V> {
private:
    size_t _width, _height;
    size_t _suggested_max;
    std::mt19937 _rng;
    std::shared_ptr<value_factory<V>> _matrix_value_factory;
    size_t _last_pos, _next_pos, _currently_generated;

    void _calc_next_pos() {
        size_t _max_pos = height() * width();
        std::uniform_int_distribution<size_t>
                _dist(1, 1 + 2 * (_max_pos - _last_pos) / (_suggested_max - _currently_generated));
        _next_pos = _last_pos + _dist(_rng);
    }

public:
    sparse_matrix_value_generator(int height, int width, size_t suggested_number_of_values,
                                  int seed, std::shared_ptr<value_factory<V>> matrix_value_factory) {
        _height = height;
        _width = width;
        _suggested_max = suggested_number_of_values;
        this->_rng = std::mt19937(seed);
        this->_matrix_value_factory = matrix_value_factory;
        _last_pos = 0;
        _next_pos = 0;
        _currently_generated = 0;
        _calc_next_pos();
    }

    bool has_next() {
        return _next_pos <= height() * width() && _currently_generated < _suggested_max;
    }

    matrix_value<V> next() {
        if (!has_next()) {
            throw no_next_value_exception();
        }
        _currently_generated++;
        _last_pos = _next_pos;
        if (has_next()) {
            _calc_next_pos();
        }
        return matrix_value(IntMath::floor_div(_last_pos, width()), 1 + (_last_pos - 1) % width(), _matrix_value_factory->next());
    }

    size_t height() {
        return this->_height;
    }

    size_t width() {
        return this->_width;
    }
};
