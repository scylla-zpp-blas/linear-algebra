#pragma once

#include <map>
#include <utility>
#include <vector>

#include "scylla_blas/structure/matrix_value.hh"
#include "scylla_blas/structure/vector_segment.hh"
#include "scylla_blas/utils/scylla_types.hh"

namespace {

/* TODO: subclass an abstract REPR<T> class in matrix_block? In scylla_blas? */
template<class T>
using LOVE = std::map<scylla_blas::index_type, scylla_blas::vector_segment<T>>;

template<class T>
using LVAL = std::vector<scylla_blas::matrix_value<T>>;

template<class T>
LOVE<T> to_list_of_vectors(const LVAL<T> &lval) {
    LOVE<T> ret;

    /* TODO : optimize to linear complexity
     * by inserting entire rows instead of separate values
     */
    for (auto &val : lval)
        ret[val.row_index].emplace_back(val.col_index, val.value);

    return ret;
}

template<class T>
LVAL<T> to_list(const LOVE<T> &love) {
    LVAL<T> ret;

    for (auto &entry : love)
        for (auto &val : entry.second)
            ret.emplace_back(entry.first, val.index, val.value);

    return ret;
}

template<class T>
LOVE<T> transpose(const LOVE<T> &love) {
    LOVE<T> ret;

    for (auto &entry : love)
        for (auto &val : entry.second)
            ret[val.index].emplace_back(entry.first, val.value);

    return ret;
}

}

namespace scylla_blas {

template<class T>
class matrix_block {
    LVAL<T> _values;

    explicit matrix_block(LVAL<T> values) :
            _values(values), i(-1), j(-1) {}
public:
    const int64_t matrix_id = 0;
    const index_type i;
    const index_type j;

    matrix_block(std::vector<scylla_blas::matrix_value<T>> values, int64_t matrix_id, index_type i, index_type j) :
        _values(values), matrix_id(matrix_id), i(i), j(j) {}

    matrix_block operator*=(const matrix_block &other) {
        _values = (*this * other)._values;

        return *this;
    }

    /* Returned block has undefined values of matrix_id, i, j */
    const matrix_block operator*(const matrix_block &other) const {
        auto list_of_vectors = to_list_of_vectors(_values);
        auto transposed_other = transpose(to_list_of_vectors(other._values));

        LVAL<T> result;

        for (auto &left : list_of_vectors)
            for (auto &right : transposed_other)
                result.emplace_back(left.first, right.first, left.second.prod(right.second));

        return matrix_block(result);
    }

    matrix_block operator+=(const matrix_block &other) {
        _values = (*this + other)._values;

        return *this;
    }

    /* Returned block has undefined values of matrix_id, i, j */
    const matrix_block operator+(const matrix_block &other) const {
        auto this_LOVE = to_list_of_vectors(_values);
        auto other_LOVE = to_list_of_vectors(other._values);

        for (auto &[row_id, row_values] : other_LOVE) {
            this_LOVE[row_id] += row_values;
        }

        return matrix_block(to_list(this_LOVE));
    }

    const LVAL<T> &get_values_raw() const {
        return _values;
    }
};

}