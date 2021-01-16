#include <iostream>
#include <string>

#include "structure/item_set.hh"
#include "scylla_matrix.hh"

#include "session.hh"

void test_matrices(std::shared_ptr<scmd::session> session) {
    auto matrix = scylla_blas::scylla_matrix<float>(session, "testowa");
    auto matrix_2 = scylla_blas::scylla_matrix<float>(session, "testowa");

    matrix.update_value(1, 0, M_PI);
    matrix.update_value(1, 1, 42);
    std::cout << "(1, 1): " << matrix.get_value(1, 1) << std::endl;
    matrix.update_value(1, 0, M_PI);
    std::cout << "(1, 0): " << matrix.get_value(1, 0) << std::endl;
    matrix.update_value(1, 1, 100);
    std::cout << "(1, 0): " << matrix.get_value(1, 0) << std::endl;
    std::cout << "(1, 1): " << matrix.get_value(1, 1) << std::endl;
}

void test_vectors() {
    auto vector_1 = scylla_blas::vector<float>();

    for (int i = 0; i < 10; i++) {
        vector_1.push_back({i, 10});
    }

    std::cout << "Vec_1: ";
    for (auto entry : vector_1) {
        std::cout << "(" << entry.index << ": " << entry.value << "), ";
    }
    std::cout << std::endl;

    auto vector_2 = scylla_blas::vector<float>();
    for (int i = 0; i < 5; i++) {
        vector_2.push_back({i, M_PI * i * i});
    }
    for (int i = 15; i < 20; i++) {
        vector_2.push_back({i, M_PI * i * i});
    }

    std::cout << "Vec_2: ";
    for (auto entry : vector_2) {
        std::cout << "(" << entry.index << ": " << entry.value << "), ";
    }
    std::cout << std::endl;

    vector_1 *= M_E;
    std::cout << "Vec_1 * e: ";
    for (auto entry : vector_1) {
        std::cout << "(" << entry.index << ": " << entry.value << "), ";
    }
    std::cout << std::endl;

    vector_1 += vector_2;
    std::cout << "Vec_1 * e + Vec_2: ";
    for (auto entry : vector_1) {
        std::cout << "(" << entry.index << ": " << entry.value << "), ";
    }
    std::cout << std::endl;
}

void test_item_sets(std::shared_ptr<scmd::session> session) {
    srand(time(NULL));
    std::vector<int> values = {0, 42, 1410, 1, 1999, 2021, 1000 * 1000 * 1000 + 7, 406};
    scylla_blas::item_set<int> s(session, values.begin(), values.end());

    try {
        for (int i = 0; ; i++) {
            int next_val = s.get_next();
            std::cout << i << ": " << next_val << std::endl;
        }
    } catch (const scylla_blas::empty_container_error &e) {
        return; // end of set
    }
}

int main(int argc, char **argv) {
    if (argc > 2) {
        std::cout << "Usage: " << argv[0] << " [IP address] [port]" << std::endl;
        exit(0);
    }

    std::string ip_address = argc > 1 ? argv[1] : "172.17.0.2"; // docker address = default
    std::string port = argc > 2 ? argv[2] : "9042";

    std::cerr << "Connecting to " << ip_address << ":" << port << "..." << std::endl;

    auto session = std::make_shared<scmd::session>(ip_address, port);

    /*
    test_matrices(session);
    test_vectors();
*/

    test_item_sets(session);

    return 0;
}

