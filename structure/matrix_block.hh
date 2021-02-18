#pragma once

#include <map>
#include <vector>

#include <structure/matrix_value.hh>
#include <structure/vector.hh>
#include <utils/scylla_types.hh>

namespace {

/* TODO: subclass an abstract REPR<T> class in matrix_block? In scylla_blas? */
template<class T>
using LOVE = std::map<scylla_blas::index_type, scylla_blas::vector<T>>;

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
    using val_t = std::vector<matrix_value<T>>;
    val_t _values;

    explicit matrix_block(val_t values) :
            _values(values), i(-1), j(-1) {}
public:
    const std::string matrix_id;
    const index_type i;
    const index_type j;

    matrix_block(val_t values, const std::string &matrix_id, index_type i, index_type j) :
        _values(values), matrix_id(matrix_id), i(i), j(j) {}

    matrix_block operator*=(const matrix_block &other) {
        _values = (*this * other)._values;
    }

    /* Returned block has undefined values of matrix_id, i, j */
    const matrix_block operator*(const matrix_block &other) const {
        auto list_of_vectors = to_list_of_vectors(_values);
        auto transposed_other = transpose(to_list_of_vectors(other._values));

        val_t result;

        for (auto &left : list_of_vectors)
            for (auto &right : transposed_other)
                result.emplace_back(left.first, right.first, left.second.prod(right.second));

        return matrix_block(result);
    }

    const val_t &get_values_raw() const {
        return _values;
    }
};

}