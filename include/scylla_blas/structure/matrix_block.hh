#pragma once

#include <map>
#include <utility>
#include <vector>

#include "scylla_blas/structure/matrix_value.hh"
#include "scylla_blas/structure/vector_segment.hh"
#include "scylla_blas/utils/scylla_types.hh"


namespace scylla_blas {

template<class T>
class matrix_block {
    using row_map_t = std::map<scylla_blas::index_type, scylla_blas::vector_segment<T>>;
    using row_hashmap_t = std::unordered_map<scylla_blas::index_type, scylla_blas::vector_segment<T>>;
    using vector_of_values = std::vector<scylla_blas::matrix_value<T>>;

    /* TODO: this may be worth optimising by direct vector construction from values. */
    static row_map_t values_to_rows (const vector_of_values &vals) {
        row_hashmap_t helper;
        for (auto &val : vals)
            helper[val.row_index].emplace_back(val.col_index, val.value);

        row_map_t ret;
        for (auto &val : helper)
            ret.insert({val.first, std::move(val.second)});

        return ret;
    }

    /* TODO: std::vector instead of std::unordered map? */
    static row_map_t transpose_row_map (const row_map_t &rmap) {
        row_hashmap_t helper;
        for (auto &entry : rmap)
            for (auto &val : entry.second)
                helper[val.index].emplace_back(entry.first, val.value);

        row_map_t ret;
        for (auto &val : helper)
            ret.insert({val.first, std::move(val.second)});

        return ret;
    }

    static vector_of_values vectors_to_values(const row_map_t &rmap) {
        vector_of_values ret;

        for (auto &entry : rmap)
            for (auto &val : entry.second)
                ret.emplace_back(entry.first, val.index, val.value);

        return ret;
    }

    static vector_of_values transpose_values(const vector_of_values &vals) {
        return vectors_to_values(transpose_row_map(values_to_rows(vals)));
    }

    vector_of_values _values;

public:
    const int64_t matrix_id = 0;
    const index_type i;
    const index_type j;

    matrix_block(vector_of_values values) :
            _values(values), i(-1), j(-1) {}

    matrix_block(vector_of_values values, int64_t matrix_id, index_type i, index_type j, TRANSPOSE trans = NoTrans) :
            _values(trans == NoTrans ? values : transpose_values(values)), matrix_id(matrix_id), i(i), j(j) {}

    matrix_block &transpose(TRANSPOSE trans) {
        if (trans != NoTrans) {
            _values = transpose_values(_values);
        }

        return *this;
    }

    const vector_of_values& get_raw() const {
        return _values;
    }

    matrix_block &operator*=(const matrix_block &other) {
        auto row_map = values_to_rows(_values);
        auto transposed_other = transpose_row_map(values_to_rows(other._values));

        _values.clear();

        for (auto &left : row_map)
            for (auto &right : transposed_other)
                _values.emplace_back(left.first, right.first, left.second.dot_prod(right.second));

        return *this;
    }

    matrix_block &operator*=(const T arg) {
        for (auto &val : _values)
            val.value *= arg;

        return *this;
    }

    /* Returned block has undefined values of matrix_id, i, j */
    const matrix_block operator*(const matrix_block &other) const {
        matrix_block result = *this;
        result *= other;

        return result;
    }

    const matrix_block operator*(const T arg) const {
        matrix_block result = *this;
        result *= arg;

        return result;
    }

    matrix_block &operator+=(const matrix_block &other) {
        auto this_row_map = values_to_rows(_values);
        auto other_row_map = values_to_rows(other._values);

        _values.clear();

        for (auto &[row_id, row_values] : other_row_map) {
            this_row_map[row_id] += row_values;
        }

        _values = vectors_to_values(this_row_map);

        return *this;
    }

    /* Returned block has undefined values of matrix_id, i, j */
    const matrix_block operator+(const matrix_block &other) const {
        matrix_block result = *this;
        result += other;

        return result;
    }

    const vector_of_values &get_values_raw() const {
        return _values;
    }

    vector_segment<T> mult_vect(const vector_segment<T> &other) const {
        auto row_map = values_to_rows(_values);
        vector_segment<T> result;

        for (auto &[row_id, row_values] : row_map) {
            result.push_back(vector_value<T>(row_id, other.dot_prod(row_values)));
        }

        return result;
    }

    static matrix_block<T> outer_prod(const vector_segment<T> &X, const vector_segment<T> &Y) {
        vector_of_values vals;

        for (auto &[i, x] : X)
            for (auto &[j, y] : Y)
                vals.emplace_back(i, j, x * y);

        return matrix_block(vals);
    }
};

}