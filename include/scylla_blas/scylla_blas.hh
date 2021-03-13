#pragma once

#include "scylla_blas/matrix.hh"
#include "scylla_blas/utils/matrix_value_generator.hh"
#include "scylla_blas/utils/scylla_types.hh"
#include "scylla_blas/utils/utils.hh"

namespace {

template <class T>
scylla_blas::matrix<T> load_matrix_from_generator(const std::shared_ptr<scmd::session>& session,
                                                  scylla_blas::matrix_value_generator<T> &gen,
                                                  const std::string& id) {
    scylla_blas::matrix<T> result(session, id, true);
    scylla_blas::vector<T> next_row;
    scylla_blas::matrix_value<T> prev_val (-1, -1, 0);

    while(gen.has_next()) {
        scylla_blas::matrix_value<T> next_val = gen.next();

        if (prev_val.row_index != -1 && next_val.row_index != prev_val.row_index) {
            result.update_row(prev_val.row_index, next_row);
            next_row.clear();
        }

        next_row.emplace_back(next_val.col_index, next_val.value);
        prev_val = next_val;
    }

    if (prev_val.row_index != -1) {
        result.update_row(prev_val.row_index, next_row);
    }

    std::cerr << "Loaded a new matrix: " << id << " from a generator" << std::endl;
    return result;
}

}

namespace scylla_blas {

/** DEBUG **/
template<class T>
void print_matrix(matrix<T> &matrix, index_type height, index_type width) {
    std::cout << "[" << matrix.get_id() << "]" << std::endl;
    for (index_type i = 1; i <= height; i++) {
        auto vec = matrix.get_row(i);
        auto it = vec.begin();
        for (index_type j = 1; j <= width; j++) {
            if (it != vec.end() && it->index == j) {
                std::cout << it->value << "\t";
                it++;
            } else {
                std::cout << 0 << "\t";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <class T>
matrix<T> multiply(std::shared_ptr<scmd::session> session, matrix_value_generator<T> &first, matrix_value_generator<T> &second) {
    auto A = load_matrix_from_generator(session, first, "A");
    auto B = load_matrix_from_generator(session, second, "B");
    auto AB = matrix<T>(session, "AB", true);

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

        AB.update_row(i, scylla_blas::vector<T>(results));
    }

    print_matrix(AB, first.height(), second.width());

    return AB;
};

template <class T>
matrix<T> easy_multiply(std::shared_ptr<scmd::session> session, matrix_value_generator<T> &first, matrix_value_generator<T> &second) {
    /* Assume that A, B are small, i.e. their size < block_size */
    auto A = load_matrix_from_generator(session, first, "A_" + std::to_string(get_timestamp()));
    auto B = load_matrix_from_generator(session, second, "B_" + std::to_string(get_timestamp()));
    auto AB = matrix<T>(session, "AB_" + std::to_string(get_timestamp()), true);

    auto A_block = A.get_block(1, 1);
    auto B_block = B.get_block(1, 1);
    auto AB_block = A_block * B_block;

    AB.update_block(1, 1, AB_block);

    print_matrix(AB, first.height(), second.width());

    return AB;
};
}
