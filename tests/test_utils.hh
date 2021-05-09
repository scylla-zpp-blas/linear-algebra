#pragma once

#include <iomanip>

#include "scylla_blas/matrix.hh"
#include "scylla_blas/routines.hh"
#include "scylla_blas/utils/scylla_types.hh"
#include "scylla_blas/utils/utils.hh"

#include "generators/matrix_value_generator.hh"

template <class T>
void load_matrix_from_generator(const std::shared_ptr<scmd::session> &session,
                                matrix_value_generator<T> &gen,
                                scylla_blas::matrix<T> &matrix) {
    scylla_blas::vector_segment<T> next_row;
    scylla_blas::matrix_value<T> prev_val (-1, -1, 0);

    while(gen.has_next()) {
        scylla_blas::matrix_value<T> next_val = gen.next();

        if (prev_val.row_index != -1 && next_val.row_index != prev_val.row_index) {
            matrix.insert_row(prev_val.row_index, next_row);
            next_row.clear();
        }

        next_row.emplace_back(next_val.col_index, next_val.value);
        prev_val = next_val;
    }

    if (prev_val.row_index != -1) {
        matrix.insert_row(prev_val.row_index, next_row);
    }

    std::cerr << "Loaded a matrix: " << matrix.id << " from a generator" << std::endl;
}

/** DEBUG **/
template<class T>
void print_matrix(const scylla_blas::matrix<T> &matrix) {
    auto default_precision = std::cout.precision();

    std::cout << std::setprecision(4);
    std::cout << "Matrix " << matrix.id << ": " << std::endl;

    /* Show column numbering */
    std::cout << " \\ \t";
    for (scylla_blas::index_type j = 1; j <= matrix.column_count; j++)
        std::cout << std::setw(6) << j << " ";
    std::cout << std::endl;

    for (scylla_blas::index_type i = 1; i <= matrix.row_count; i++) {
        std::cout  << i << " ->\t";
        auto vec = matrix.get_row(i);
        auto it = vec.begin();
        for (scylla_blas::index_type j = 1; j <= matrix.column_count; j++) {
            if (it != vec.end() && it->index == j) {
                std::cout << std::setw(6) << it->value << " ";
                it++;
            } else {
                std::cout << std::setw(6) << 0 << " ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << std::setprecision(default_precision);
}

template<class T>
void print_vector(const scylla_blas::vector<T> &vec) {
    auto default_precision = std::cout.precision();
    auto whole = vec.get_whole();

    std::cout << std::setprecision(4);
    std::cout << "Vector " << vec.id << ": " << std::endl;

    scylla_blas::index_type expected = 1;
    for (auto entry : whole) {
        while (expected < entry.index) {
            /* Show empty rows */
            std::cout << expected << " ->\t" << 0 << std::endl;
            expected++;
        }

        std::cout << entry.index << " ->\t" << entry.value << std::endl;

        expected = entry.index + 1;
    }

    std::cout << std::endl;
    std::cout << std::setprecision(default_precision);
}
