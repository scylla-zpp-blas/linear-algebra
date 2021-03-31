#pragma once
#include "scylla_blas/matrix.hh"
#include "scylla_blas/routines.hh"
#include "scylla_blas/utils/matrix_value_generator.hh"
#include "scylla_blas/utils/scylla_types.hh"
#include "scylla_blas/utils/utils.hh"

namespace {

template <class T>
scylla_blas::matrix<T> load_matrix_from_generator(const std::shared_ptr<scmd::session> &session,
                                                  scylla_blas::matrix_value_generator<T> &gen, int64_t id) {
    scylla_blas::matrix<T> result = scylla_blas::matrix<T>::init_and_return(session, id);
    scylla_blas::vector_segment<T> next_row;
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

/** DEBUG **/
template<class T>
void print_matrix(scylla_blas::matrix<T> &matrix,
                  scylla_blas::index_type height,
                  scylla_blas::index_type width) {
    std::cout << "[" << matrix.get_id() << "]" << std::endl;
    for (scylla_blas::index_type i = 1; i <= height; i++) {
        auto vec = matrix.get_row(i);
        auto it = vec.begin();
        for (scylla_blas::index_type j = 1; j <= width; j++) {
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

namespace scylla_blas {

template <class T>
matrix<T> naive_multiply(std::shared_ptr<scmd::session> session,
                         matrix_value_generator<T> &first,
                         matrix_value_generator<T> &second) {
    int64_t base_id = get_timestamp();

    int64_t A_id = base_id;
    int64_t B_id = base_id + 1;
    int64_t AB_id = base_id + 2;

    auto A = load_matrix_from_generator(session, first, A_id);
    auto B = load_matrix_from_generator(session, second, B_id);
    auto AB = matrix<T>::init_and_return(session, AB_id);

    print_matrix(A, first.height(), first.width());
    print_matrix(B, second.height(), second.width());

    for (index_type i = 1; i <= first.height(); i++) {
        /* FIXME? std::unordered_map could be more efficient for very large and very sparce matrices */
        std::vector<T> results(second.width() + 1, 0);

        scylla_blas::vector_segment<T> left_vec = A.get_row(i);
        for (auto left_entry : left_vec) {
            scylla_blas::vector_segment<T> right_vec = B.get_row(left_entry.index);
            right_vec *= left_entry.value;
            for (auto right_entry : right_vec) {
                results[right_entry.index] += right_entry.value;
            }
        }

        AB.update_row(i, scylla_blas::vector_segment<T>(results));
    }

    print_matrix(AB, first.height(), second.width());

    return AB;
};

template<class T>
matrix<T> blas_multiply(std::shared_ptr<scmd::session> session,
                        const matrix<T> &A, const matrix<T> &B);

template<>
matrix<float> blas_multiply(std::shared_ptr<scmd::session> session,
                            const matrix<float> &A, const matrix<float> &B) {
    return routine_factory(session).sgemm(RowMajor, NoTrans, NoTrans, 1, A, B, 1);
}

template<>
matrix<double> blas_multiply(std::shared_ptr<scmd::session> session,
                             const matrix<double> &A, const matrix<double> &B) {
    return routine_factory(session).dgemm(RowMajor, NoTrans, NoTrans, 1, A, B, 1);
}

/* 'Cannon' algorithm (actually little to do with the actual Cannon's algorithm) */
template <class T>
matrix<T> parallel_multiply(std::shared_ptr<scmd::session> session,
                            matrix_value_generator<T> &first,
                            matrix_value_generator<T> &second) {
    int64_t base_id = get_timestamp();
    int64_t A_id = base_id;
    int64_t B_id = base_id + 1;

    auto A = load_matrix_from_generator(session, first, A_id);
    auto B = load_matrix_from_generator(session, second, B_id);

    auto AB = blas_multiply<T>(session, A, B);

    print_matrix(A, first.height(), second.width());
    print_matrix(B, first.height(), second.width());
    print_matrix(AB, first.height(), second.width());

    return AB;
}

}
