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
    using list_of_vectors = std::map<scylla_blas::index_type, scylla_blas::vector_segment<T>>;
    using fast_list_of_vectors = std::unordered_map<scylla_blas::index_type, scylla_blas::vector_segment<T>>;
    using list_of_values = std::vector<scylla_blas::matrix_value<T>>;

    /* TODO: this may be worth optimising by direct vector construction from values. */
    list_of_vectors values_to_vectors (const list_of_values &lval) {
        fast_list_of_vectors helper;
        for (auto &val : lval)
            helper[val.row_index].emplace_back(val.col_index, val.value);

        list_of_vectors ret;
        for (auto &val : helper)
            ret.insert({val.first, std::move(val.second)});

        return ret;
    }

    /* TODO: std::vector instead of std::unordered map? */
    list_of_vectors transpose_vectors(const list_of_vectors &love) {
        fast_list_of_vectors helper;
        for (auto &entry : love)
            for (auto &val : entry.second)
                helper[val.index].emplace_back(entry.first, val.value);

        list_of_vectors ret;
        for (auto &val : helper)
            ret.insert({val.first, std::move(val.second)});

        return ret;
    }

    list_of_values vectors_to_values(const list_of_vectors &love) {
        list_of_values ret;

        for (auto &entry : love)
            for (auto &val : entry.second)
                ret.emplace_back(entry.first, val.index, val.value);

        return ret;
    }

    list_of_values transpose_values(const list_of_values &lval) {
        return vectors_to_values(transpose_vectors(values_to_vectors(lval)));
    }

    list_of_values _values;

    explicit matrix_block(list_of_values values) :
            _values(values), i(-1), j(-1) {}
public:
    const int64_t matrix_id = 0;
    const index_type i;
    const index_type j;

    matrix_block(list_of_values values, int64_t matrix_id, index_type i, index_type j, TRANSPOSE trans = NoTrans) :
            _values(trans == NoTrans ? values : transpose_values(values)), matrix_id(matrix_id), i(i), j(j) {}

    matrix_block& transpose(TRANSPOSE trans) {
        if (trans != NoTrans) {
            _values = transpose_values(_values);
        }

        return *this;
    }

    matrix_block& operator*=(const matrix_block &other) {
        auto list_of_vectors = values_to_vectors(_values);
        auto transposed_other = transpose_vectors(values_to_vectors(other._values));

        _values.clear();

        for (auto &left : list_of_vectors)
            for (auto &right : transposed_other)
                _values.emplace_back(left.first, right.first, left.second.prod(right.second));

        return *this;
    }

    matrix_block& operator*=(const T arg) {
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

    matrix_block& operator+=(const matrix_block &other) {
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

    const list_of_values& get_values_raw() const {
        return _values;
    }
};

}