#pragma once

#include <map>
#include <utility>
#include <vector>

#include "scylla_blas/structure/matrix_value.hh"
#include "scylla_blas/structure/vector_segment.hh"
#include "scylla_blas/utils/scylla_types.hh"

namespace {

template<class T>
using list_of_vectors = std::map<scylla_blas::index_type, scylla_blas::vector_segment<T>>;

template<class T>
using fast_list_of_vectors = std::unordered_map<scylla_blas::index_type, scylla_blas::vector_segment<T>>;

template<class T>
using list_of_values = std::vector<scylla_blas::matrix_value<T>>;

/* TODO: this may be worth optimising by direct vector construction from values. */
template<class T>
list_of_vectors<T> values_to_vectors (const list_of_values<T> &lval) {
    fast_list_of_vectors<T> helper;
    for (auto &val : lval)
        helper[val.row_index].emplace_back(val.col_index, val.value);

    list_of_vectors<T> ret;
    for (auto &val : helper)
        ret.insert({val.first, std::move(val.second)});

    return ret;
}

/* TODO: std::vector instead of std::unordered map? */
template<class T>
list_of_vectors<T> transpose(const list_of_vectors<T> &love) {
    fast_list_of_vectors<T> helper;
    for (auto &entry : love)
        for (auto &val : entry.second)
            helper[val.index].emplace_back(entry.first, val.value);

    list_of_vectors<T> ret;
    for (auto &val : helper)
        ret.insert({val.first, std::move(val.second)});

    return ret;
}

template<class T>
list_of_values<T> vectors_to_values(const list_of_vectors<T> &love) {
    list_of_values<T> ret;

    for (auto &entry : love)
        for (auto &val : entry.second)
            ret.emplace_back(entry.first, val.index, val.value);

    return ret;
}

}

namespace scylla_blas {

template<class T>
class matrix_block {
    list_of_values<T> _values;

    explicit matrix_block(list_of_values<T> values) :
            _values(values), i(-1), j(-1) {}
public:
    const int64_t matrix_id = 0;
    const index_type i;
    const index_type j;

    matrix_block(std::vector<scylla_blas::matrix_value<T>> values, int64_t matrix_id, index_type i, index_type j) :
        _values(values), matrix_id(matrix_id), i(i), j(j) {}

    matrix_block operator*=(const matrix_block &other) {
        auto list_of_vectors = values_to_vectors(_values);
        auto transposed_other = transpose(values_to_vectors(other._values));

        _values.clear();

        for (auto &left : list_of_vectors)
            for (auto &right : transposed_other)
                _values.emplace_back(left.first, right.first, left.second.prod(right.second));

        return *this;
    }

    /* Returned block has undefined values of matrix_id, i, j */
    const matrix_block operator*(const matrix_block &other) const {
        matrix_block result = *this;
        result *= other;

        return result;
    }

    matrix_block operator+=(const matrix_block &other) {
        auto this_ListVec = values_to_vectors(_values);
        auto other_ListVec = values_to_vectors(other._values);

        _values.clear();

        for (auto &[row_id, row_values] : other_ListVec) {
            this_ListVec[row_id] += row_values;
        }

        _values = vectors_to_values(this_ListVec);

        return *this;
    }

    /* Returned block has undefined values of matrix_id, i, j */
    const matrix_block operator+(const matrix_block &other) const {
        matrix_block result = *this;
        result += other;

        return result;
    }

    const list_of_values<T> &get_values_raw() const {
        return _values;
    }
};

}