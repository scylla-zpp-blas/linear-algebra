#pragma once
#include "scylla_blas/matrix.hh"
#include "scylla_blas/routines.hh"
#include "scylla_blas/utils/matrix_value_generator.hh"
#include "scylla_blas/utils/scylla_types.hh"
#include "scylla_blas/utils/utils.hh"

namespace scylla_blas {

template <class T>
scylla_blas::matrix<T> load_matrix_from_generator(const std::shared_ptr<scmd::session> &session,
                                                  scylla_blas::matrix_value_generator<T> &gen, int64_t id) {
    scylla_blas::matrix<T> result = scylla_blas::matrix<T>::init_and_return(session, id, gen.height(), gen.width(), true);
    scylla_blas::vector_segment<T> next_row;
    scylla_blas::matrix_value<T> prev_val (-1, -1, 0);

    while(gen.has_next()) {
        scylla_blas::matrix_value<T> next_val = gen.next();

        if (prev_val.row_index != -1 && next_val.row_index != prev_val.row_index) {
            result.insert_row(prev_val.row_index, next_row);
            next_row.clear();
        }

        next_row.emplace_back(next_val.col_index, next_val.value);
        prev_val = next_val;
    }

    if (prev_val.row_index != -1) {
        result.insert_row(prev_val.row_index, next_row);
    }

    std::cerr << "Loaded a new matrix: " << id << " from a generator" << std::endl;
    return result;
}

/** DEBUG **/
template<class T>
void print_matrix(scylla_blas::matrix<T> &matrix) {
    std::cout << "[" << matrix.id << "]" << std::endl;
    for (scylla_blas::index_type i = 1; i <= matrix.row_count; i++) {
        auto vec = matrix.get_row(i);
        auto it = vec.begin();
        for (scylla_blas::index_type j = 1; j <= matrix.column_count; j++) {
            if (it != vec.end() && it->index == j) {
                std::cout << it->value << " ";
                it++;
            } else {
                std::cout << 0 << "\t";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

}
