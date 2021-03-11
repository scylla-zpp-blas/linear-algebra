#include <boost/test/unit_test.hpp>

#include "fixture.hh"

#include <session.hh>
#include <scylla_blas/queue/item_set.hh>
#include <scylla_blas/matrix.hh>

BOOST_FIXTURE_TEST_SUITE(structure_tests, scylla_fixture)

BOOST_AUTO_TEST_CASE(matrices)
{
    auto matrix = scylla_blas::matrix<float>(session, "testowa");
    auto matrix_2 = scylla_blas::matrix<float>(session, "testowa");

    matrix.update_value(1, 0, M_PI);
    matrix.update_value(1, 1, 42);
    std::cout << "(1, 1): " << matrix.get_value(1, 1) << std::endl;
    matrix.update_value(1, 0, M_PI);
    std::cout << "(1, 0): " << matrix.get_value(1, 0) << std::endl;
    matrix.update_value(1, 1, 100);
    std::cout << "(1, 0): " << matrix.get_value(1, 0) << std::endl;
    std::cout << "(1, 1): " << matrix.get_value(1, 1) << std::endl;
}


BOOST_AUTO_TEST_CASE(vectors)
{
    auto vector_1 = scylla_blas::vector<float>();

    for (int i = 0; i < 10; i++) {
        vector_1.emplace_back(i, 10);
    }

    std::cout << "Vec_1: ";
    for (auto entry : vector_1) {
        std::cout << "(" << entry.index << ": " << entry.value << "), ";
    }
    std::cout << std::endl;

    auto vector_2 = scylla_blas::vector<float>();
    for (int i = 0; i < 5; i++) {
        vector_2.emplace_back(i, (float)M_PI * i * i);
    }
    for (int i = 15; i < 20; i++) {
        vector_2.emplace_back(i, (float) M_PI * i * i);
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

BOOST_AUTO_TEST_CASE(item_set)
{
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

BOOST_AUTO_TEST_SUITE_END();

