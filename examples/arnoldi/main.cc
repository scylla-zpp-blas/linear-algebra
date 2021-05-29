#include <string>
#include <stdexcept>
#include <scylla_blas/config.hh>
#include <session.hh>
#include "arnoldi.hh"
#include <iostream>
#include <scylla_blas/matrix.hh>
#include <scylla_blas/vector.hh>
#include "generators/sparse_matrix_value_generator.hh"
#include "generators/random_value_factory.hh"
#include <iostream>
#include <iomanip>

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
void print_matrix_octave(const scylla_blas::matrix<T> &matrix) {
    auto default_precision = std::cout.precision();

    std::cout << std::setprecision(4);
    std::cout << "Matrix " << matrix.id << ": " << std::endl;

    std::cout << "[\n";

    for (scylla_blas::index_type i = 1; i <= matrix.row_count; i++) {
        auto vec = matrix.get_row(i);
        auto it = vec.begin();
        for (scylla_blas::index_type j = 1; j <= matrix.column_count; j++) {
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

template<class T>
void init_vector(std::shared_ptr<scmd::session> session,
                 std::shared_ptr<scylla_blas::vector<T>> &vector_ptr,
                 scylla_blas::index_type len,
                 int64_t id,
                 std::shared_ptr<value_factory<T>> value_factory = nullptr) {
    scylla_blas::vector<T>::clear(session, id);
    vector_ptr = std::make_shared<scylla_blas::vector<T>>(session, id);

    if (value_factory != nullptr) {
        std::vector<scylla_blas::vector_value<T>> values;

        for (scylla_blas::index_type i = 1; i <= len; i++)
            values.emplace_back(i, value_factory->next());

        vector_ptr->update_values(values);
    }
}

template<class T>
void init_matrix(std::shared_ptr<scmd::session> session,
                 std::shared_ptr<scylla_blas::matrix<T>>& matrix_ptr,
                 scylla_blas::index_type w,
                 scylla_blas::index_type h,
                 int64_t id,
                 std::shared_ptr<value_factory<T>> value_factory = nullptr) {
//    matrix_ptr = std::make_shared<scylla_blas::matrix<T>>(session, id);
    matrix_ptr->clear_all();

    if (value_factory != nullptr) {
        sparse_matrix_value_generator<T> gen(w, h, w * h / 5, id, value_factory);
        load_matrix_from_generator(session, gen, *matrix_ptr);
    }
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        throw std::runtime_error("You need to specify ip in the command line: " + std::string(argv[0]) + " scylla_ip [scylla_port]");
    }
    std::string scylla_ip = argv[1];
    std::string scylla_port = argc > 2 ? argv[2] : std::to_string(SCYLLA_DEFAULT_PORT);

    auto session = std::make_shared<scmd::session>(scylla_ip, scylla_port);

    scylla_blas::index_type m = 10, n = 9;
    arnoldi::containers c = arnoldi::containers<float>(session, 123456, m, n);
    //sparse_matrix_value_generator<float> v = sparse_matrix_value_generator<float>(m, m, 300*10, 12345, random_value_factory<float>(0, 123, 123321));
    std::shared_ptr<value_factory<float>> f =
            std::make_shared<random_value_factory<float>>(0, 9, 142);
    init_matrix<float>(session, c.A, m, m, c.A_id, f);
    c.b->update_value(1, 1.0f);
    auto arnoldi_iteration = arnoldi(session);
    print_matrix_octave(*c.A);
    arnoldi_iteration.compute(c.A, c.b, n, c.h, c.Q, c.v, c.q, c.t);
    print_matrix_octave(*c.Q);
    print_matrix_octave(*c.h);

    auto test = c.Q->get_row(0);
    if (!test.empty()) std::cout << "oops";
}
