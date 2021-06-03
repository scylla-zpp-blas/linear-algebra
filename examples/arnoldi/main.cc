#include <string>
#include <stdexcept>
#include <scylla_blas/config.hh>
#include <session.hh>
#include "arnoldi.hh"
#include <iostream>
#include <scylla_blas/matrix.hh>
#include <scylla_blas/vector.hh>
#include "sparse_matrix_value_generator.hh"
#include "random_value_factory.hh"
#include <iomanip>

template<class T>
void print_matrix_octave(const scylla_blas::matrix<T> &matrix);


template <class T>
void load_matrix_from_generator(const std::shared_ptr<scmd::session> &session,
                                matrix_value_generator<T> &gen,
                                scylla_blas::matrix<T> &matrix);

template<class T>
void init_matrix(std::shared_ptr<scmd::session> session,
                 std::shared_ptr<scylla_blas::matrix<T>>& matrix_ptr,
                 scylla_blas::index_t w,
                 scylla_blas::index_t h,
                 id_t id,
                 std::shared_ptr<value_factory<T>> value_factory = nullptr);

const id_t INITIAL_MATRIX_ID = 123456;

int main(int argc, char **argv) {
    if (argc <= 4) {
        throw std::runtime_error("You need to specify ip in the command line: " + std::string(argv[0]) + " scylla_ip scylla_port m n");
    }
    scylla_blas::index_t m, n;
    std::string scylla_ip = argv[1];
    std::string scylla_port = argv[2];
    m = std::stoi(argv[3]);
    n = std::stoi(argv[4]);
    if (n > m) {
        throw std::runtime_error("n cannot be greater than m");
    }

    auto session = std::make_shared<scmd::session>(scylla_ip, scylla_port);

    arnoldi::containers c = arnoldi::containers<float>(session, INITIAL_MATRIX_ID, m, n);

    // Initialize matrix A with random values.
    std::shared_ptr<value_factory<float>> f =
            std::make_shared<random_value_factory<float>>(0, 9, 142);
    init_matrix<float>(session, c.A, m, m, c.A_id, f);

    // Set vector to (1, 0, 0 ... 0).
    c.b->update_value(1, 1.0f);

    auto arnoldi_iteration = arnoldi(session);
    print_matrix_octave(*c.A);
    arnoldi_iteration.compute(c.A, c.b, n, c.h, c.Q, c.v, c.q, c.t);
    print_matrix_octave(*c.Q);
    print_matrix_octave(*c.h);
}

template<class T>
void print_matrix_octave(const scylla_blas::matrix<T> &matrix) {
    auto default_precision = std::cout.precision();

    std::cout << std::setprecision(4);
    std::cout << "Matrix " << matrix.get_id() << ": " << std::endl;

    std::cout << "[\n";

    for (scylla_blas::index_t i = 1; i <= matrix.get_row_count(); i++) {
        auto vec = matrix.get_row(i);
        auto it = vec.begin();
        for (scylla_blas::index_t j = 1; j <= matrix.get_column_count(); j++) {
            if (it != vec.end() && it->index == j) {
                std::cout << it->value << ", ";
                it++;
            } else {
                std::cout << 0 << ", ";
            }
        }
        std::cout << "\n";
    }

    std::cout << "]\n";
    std::cout << std::setprecision(default_precision);
}

template<class T>
void load_matrix_from_generator(const std::shared_ptr<scmd::session> &session, matrix_value_generator<T> &gen,
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

    std::cerr << "Loaded a matrix: " << matrix.get_id() << " from a generator" << std::endl;
}

size_t suggest_number_of_values(scylla_blas::index_t w, scylla_blas::index_t h) {
    size_t ret = w * h / 5; // 20% load
    return (ret > 0 ? ret : 1);
}

template<class T>
void init_matrix(std::shared_ptr<scmd::session> session,
                 std::shared_ptr<scylla_blas::matrix<T>> &matrix_ptr,
                 scylla_blas::index_t w,
                 scylla_blas::index_t h,
                 id_t id,
                 std::shared_ptr<value_factory <T>> value_factory) {
    //    matrix_ptr = std::make_shared<scylla_blas::matrix<T>>(session, id);
    matrix_ptr->clear_all();

    if (value_factory != nullptr) {
        sparse_matrix_value_generator<T> gen = sparse_matrix_value_generator<T>(w, h, suggest_number_of_values(w, h), id, value_factory);
        load_matrix_from_generator(session, gen, *matrix_ptr);
    }
}
