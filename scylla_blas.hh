#pragma once

#include <bits/exception.h>
#include "scylla_types.hh"
#include "scylla_matrix.hh"
#include "matrix_value_generator.hh"

namespace {

template <class T>
scylla_blas::scylla_matrix<T> load_matrix_from_generator(std::shared_ptr<scmd::session> session,
                                                         scylla_blas::matrix_value_generator<T> &gen,
                                                         std::string id) {
    scylla_blas::scylla_matrix<T> result(session, id, true);
    scylla_blas::vector<T> next_row;
    scylla_blas::matrix_value<T> prev_val (-1, -1, 0);

    while(gen.has_next()) {
        scylla_blas::matrix_value<T> next_val = gen.next();

        if (prev_val.i != -1 && next_val.i != prev_val.i) {
            result.update_row(prev_val.i, next_row);
            next_row.clear();
        }

        next_row.emplace_back(next_val.j, next_val.val);
        prev_val = next_val;
    }

    if (prev_val.i != -1) {
        result.update_row(prev_val.i, next_row);
    }

    std::cerr << "Loaded a new matrix: " << id << " from a generator" << std::endl;
    return result;
}

}

namespace scylla_blas {

/** DEBUG **/
template<class T>
void print_matrix(scylla_matrix<T> matrix, index_type height, index_type width) {
    std::cout << "[" << matrix.get_id() << "]" << std::endl;
    for (index_type i = 1; i <= height; i++) {
        auto vec = matrix.get_row(i);
        auto it = vec.begin();
        for (index_type j = 1; j <= width; j++) {
            if (it->index == j) {
                std::cout << it->value << "\t";
                it++;
            } else {
                std::cout << 0 << "\t";
            }
        }
        std::cout << std::endl;
    }
}

template <class T>
scylla_matrix<T> multiply(std::shared_ptr<scmd::session> session, matrix_value_generator<T> &first, matrix_value_generator<T> &second) {
    auto A = load_matrix_from_generator(session, first, "A");
    auto B = load_matrix_from_generator(session, second, "B");
    auto AB = scylla_matrix<T>(session, "AB", true);

    print_matrix(A, first.height(), first.width());
    print_matrix(B, second.height(), second.width());

    for (index_type i = 1; i <= first.height(); i++) {
        /* FIXME? std::unordered_map could be more efficient for very large and very sparce matrices */
        std::vector<T> results(second.width() + 1, 0);

        scylla_blas::vector<T> left_vec = A.get_row(i);
        for (auto left_entry : left_vec) {
            scylla_blas::vector<T> right_vec = B.get_row(left_entry.index);
            right_vec *= left_entry.value;
            for (auto right_entry : right_vec) {
                results[right_entry.index] += right_entry.value;
            }
        }

        AB.update_row(i, results);
    }

    print_matrix(AB, first.height(), second.width());

    return AB;
};

}
