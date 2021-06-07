#include <string>
#include <iostream>
#include <stdexcept>
#include <iomanip>

#include <boost/program_options.hpp>

#include <session.hh>

#include <scylla_blas/matrix.hh>
#include <scylla_blas/vector.hh>
#include <scylla_blas/config.hh>

#include "sparse_matrix_value_generator.hh"
#include "random_value_factory.hh"

#include "arnoldi.hh"

namespace po = boost::program_options;

struct options {
    std::string host{};
    uint16_t port{};
    scylla_blas::index_t m;
    scylla_blas::index_t n;
    scylla_blas::index_t block_size;
    int64_t workers;
    int64_t scheduler_sleep_time;
    bool print_matrices = false;
};

void parse_arguments(int ac, char *av[], options &options) {
    po::options_description desc(fmt::format("Usage: {} [options]", av[0]));
    po::options_description opt("Options");
    opt.add_options()
            ("help", "Show program help")
            ("block_size,B", po::value<scylla_blas::index_t>(&options.block_size)->default_value(DEFAULT_BLOCK_SIZE), "Block size to use")
            ("workers,W", po::value<int64_t>(&options.workers)->default_value(DEFAULT_WORKER_COUNT), "How many workers can we use")
            ("sleep_time,S", po::value<int64_t>(&options.scheduler_sleep_time)->default_value(DEFAULT_SCHEDULER_SLEEP_TIME_MICROSECONDS), "How long should scheduler sleep waiting for task")
            ("print", "Print matrices")
            (",n", po::value<scylla_blas::index_t>(&options.n)->required(), "How many iterations of algorithm should be performed")
            (",m", po::value<scylla_blas::index_t>(&options.m)->required(), "Dimension of input matrix")
            ("host,H", po::value<std::string>(&options.host)->required(), "Address on which Scylla can be reached")
            ("port,P", po::value<uint16_t>(&options.port)->default_value(SCYLLA_DEFAULT_PORT), "port number on which Scylla can be reached");
    desc.add(opt);
    try {
        auto parsed = po::command_line_parser(ac, av)
                .options(desc)
                .run();
        po::variables_map vm;
        po::store(parsed, vm);
        if (vm.count("help") || ac == 1) {
            std::cout << desc << "\n";
            std::exit(0);
        }

        if (vm.count("print")) {
            options.print_matrices = true;
        }

        po::notify(vm);
    } catch (std::exception &e) {
        LogCritical("error: {}", e.what());
        std::exit(1);
    } catch (...) {
        LogCritical("Exception of unknown type!");
        std::exit(1);
    }
}


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
    struct options op;
    parse_arguments(argc, argv, op);

    if (op.n > op.m) {
        throw std::runtime_error("n cannot be greater than m");
    }

    auto session = std::make_shared<scmd::session>(op.host, std::to_string(op.port));

    arnoldi::containers c = arnoldi::containers<float>(session, INITIAL_MATRIX_ID, op.m, op.n, op.block_size);

    // Initialize matrix A with random values.
    std::shared_ptr<value_factory<float>> f =
            std::make_shared<random_value_factory<float>>(0, 9, 142);
    init_matrix<float>(session, c.A, op.m, op.m, c.A_id, f);

    // Set vector to (1, 0, 0 ... 0).
    c.b->update_value(1, 1.0f);

    auto arnoldi_iteration = arnoldi(session, op.workers, op.scheduler_sleep_time);
    if(op.print_matrices) print_matrix_octave(*c.A);
    arnoldi_iteration.compute(c.A, c.b, op.n, c.h, c.Q, c.v, c.q, c.t);
    if(op.print_matrices) print_matrix_octave(*c.Q);
    if(op.print_matrices) print_matrix_octave(*c.h);
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

    LogInfo("Loaded a matrix {} from a generator", matrix.get_id());
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
