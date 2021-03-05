#include <iostream>
#include <memory>

#include "value_factory.hh"
#include "sparse_matrix_value_generator.hh"
#include "scylla_blas.hh"

#include "session.hh"

int main(int argc, char **argv) {
    if (argc > 3) {
        std::cout << "Usage: " << argv[0] << " [IP address] [port]" << std::endl;
        exit(0);
    }

    std::string ip_address = argc > 1 ? argv[1] : "172.17.0.2"; // docker address = default
    std::string port = argc > 2 ? argv[2] : "9042";

    std::cerr << "Connecting to " << ip_address << ":" << port << "..." << std::endl;

    auto session = std::make_shared<scmd::session>(ip_address, port);

    {
        std::shared_ptr<scylla_blas::value_factory<float>> f = std::make_shared<scylla_blas::value_factory<float>>(0, 9,
                                                                                                                   1410);
        scylla_blas::sparse_matrix_value_generator<float> gen1(5, 5, 10, 11121, f);
        scylla_blas::sparse_matrix_value_generator<float> gen2(5, 5, 10, 22222, f);
        auto result = scylla_blas::multiply(session, gen1, gen2);
    }

    {
        std::shared_ptr<scylla_blas::value_factory<float>> f = std::make_shared<scylla_blas::value_factory<float>>(0, 9,
                                                                                                                   1410);
        scylla_blas::sparse_matrix_value_generator<float> gen1(5, 5, 10, 11121, f);
        scylla_blas::sparse_matrix_value_generator<float> gen2(5, 5, 10, 22222, f);
        auto result = scylla_blas::easy_multiply(session, gen1, gen2);
    }

    return 0;
}