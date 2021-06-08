#pragma once

#include <random>
#include <memory>

#include "matrix_value_generator.hh"

template<class V>
class preset_matrix_value_generator : public matrix_value_generator<V> {
private:
    size_t _width, _height;
    std::vector<scylla_blas::matrix_value<V>> _values;
    size_t _iterator;

public:
    preset_matrix_value_generator(size_t height, size_t width,
                                  std::vector<scylla_blas::matrix_value<V>> values) {
        _height = height;
        _width = width;
        _iterator = 0;
        _values = values;
    }

    bool has_next() {
        return _iterator < _values.size();
    }

    scylla_blas::matrix_value<V> next() {
        if (!has_next()) {
            throw no_next_value_exception();
        }
        scylla_blas::matrix_value<V> ret = _values[_iterator++];
        if (ret.row_index > height() || ret.col_index > width()) {
            throw std::runtime_error("Invalid preset value.");
        }
        return ret;
    }

    size_t height() {
        return this->_height;
    }

    size_t width() {
        return this->_width;
    }
};