#include <string>
#include <stdexcept>
#include <scylla_blas/config.hh>
#include <session.hh>
#include "arnoldi.hh"
#include <iostream>
#include <scylla_blas/matrix.hh>
#include <scylla_blas/vector.hh>


int main(int argc, char **argv) {
    if (argc <= 1) {
        throw std::runtime_error("You need to specify ip in the command line: " + std::string(argv[0]) + " scylla_ip [scylla_port]");
    }
    std::string scylla_ip = argv[1];
    std::string scylla_port = argc > 2 ? argv[2] : std::to_string(SCYLLA_DEFAULT_PORT);

    auto session = std::make_shared<scmd::session>(scylla_ip, scylla_port);
}