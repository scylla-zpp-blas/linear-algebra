#include <iostream>
#include <memory>
#include "value_factory.hh"
#include "sparse_matrix_value_generator.hh"
#include "scylla_blas.hh"
#include "session.hh"

int main(int argc, char **argv) {
    if (argc > 2) {
        std::cout << "Usage: " << argv[0] << " [IP address] [port]" << std::endl;
        exit(0);
    }

    std::string ip_address = argc > 1 ? argv[1] : "172.17.0.2"; // docker address = default
    std::string port = argc > 2 ? argv[2] : "9042";

    std::cerr << "Connecting to " << ip_address << ":" << port << "..." << std::endl;

    auto session = std::make_shared<scmd::session>(ip_address, port);

    std::shared_ptr<value_factory<float>> f = std::make_shared<value_factory<float>>(0, 9, 42);
    scylla_blas::sparse_matrix_value_generator<float> gen1(1000, 1000, 10000, 1111, f);
    scylla_blas::sparse_matrix_value_generator<float> gen2(1000, 1000, 10000, 2222, f);
    auto result = scylla_blas::multiply(session, gen1, gen2);

    return 0;
}