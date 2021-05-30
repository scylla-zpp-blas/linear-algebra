#pragma once

#include "scylla_blas/config.hh"
#include "const.hh"

class global_config {
public:
    static inline int argc = 0;
    static inline char** argv = nullptr;
    static inline std::string scylla_ip;
    static inline std::string scylla_port;

    static void init() {
        argc = boost::unit_test::framework::master_test_suite().argc;
        argv = boost::unit_test::framework::master_test_suite().argv;
        if (argc <= 1) {
            throw std::runtime_error("You need to specify ip in the command line: ./tests -- scylla_ip");
        }
        scylla_ip = argv[1];
        scylla_port = argc > 2 ? argv[2] : std::to_string(SCYLLA_DEFAULT_PORT);
    }
};