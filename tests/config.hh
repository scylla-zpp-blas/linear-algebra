#pragma once

class global_config {
public:
    static inline int argc = 0;
    static inline char** argv = nullptr;
    static inline std::string scylla_ip = "";
    static inline std::string scylla_port = "";

    static void init() {
        argc = boost::unit_test::framework::master_test_suite().argc;
        argv = boost::unit_test::framework::master_test_suite().argv;

        scylla_ip = argc > 1 ? argv[1] : "172.17.0.2";
        scylla_port = argc > 2 ? argv[2] : "9042";
    }
};