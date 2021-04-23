#include <boost/test/unit_test.hpp>

#include "scylla_blas/queue/scylla_queue.hh"
#include "scylla_blas/matrix.hh"
#include "fixture.hh"

BOOST_FIXTURE_TEST_SUITE(structure_tests, scylla_fixture)

BOOST_AUTO_TEST_CASE(matrices)
{
    scylla_blas::matrix<float>::init(session, 0, 5, 4, true);

    auto matrix = scylla_blas::matrix<float>(session, 0);
    auto matrix_2 = scylla_blas::matrix<float>(session, 0);

    matrix.insert_value(1, 0, M_PI);
    matrix.insert_value(1, 1, 42);
    BOOST_REQUIRE_EQUAL(std::ceil(matrix.get_value(1, 0) * 10000), std::ceil(M_PI * 10000));
    BOOST_REQUIRE_EQUAL(matrix.get_value(1, 1), 42);

    matrix.insert_value(1, 0, M_PI);
    BOOST_REQUIRE_EQUAL(std::ceil(matrix.get_value(1, 0) * 10000), std::ceil(M_PI * 10000));

    matrix.insert_value(1, 1, 100);
    BOOST_REQUIRE_EQUAL(std::ceil(matrix.get_value(1, 0) * 10000), std::ceil(M_PI * 10000));
    BOOST_REQUIRE_EQUAL(matrix.get_value(1, 1), 100);

    BOOST_REQUIRE_EQUAL(matrix.row_count, 5);
    BOOST_REQUIRE_EQUAL(matrix.column_count, 4);

    BOOST_REQUIRE_EQUAL(matrix.row_count, matrix_2.row_count);
    BOOST_REQUIRE_EQUAL(matrix.column_count, matrix_2.column_count);
}


BOOST_AUTO_TEST_CASE(vectors)
{
    auto vector_1 = scylla_blas::vector_segment<float>();

    for (int i = 0; i < 10; i++) {
        vector_1.emplace_back(i, 10);
    }

    std::cout << "Vec_1: ";
    for (auto entry : vector_1) {
        std::cout << "(" << entry.index << ": " << entry.value << "), ";
    }
    std::cout << std::endl;

    auto vector_2 = scylla_blas::vector_segment<float>();
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

BOOST_AUTO_TEST_SUITE_END();
