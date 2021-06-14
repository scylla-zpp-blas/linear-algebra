#include <boost/test/unit_test.hpp>

#include "scylla_blas/queue/scylla_queue.hh"
#include "scylla_blas/matrix.hh"
#include "scylla_blas/vector.hh"
#include "scylla_blas/config.hh"
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

    BOOST_REQUIRE_EQUAL(matrix.get_row_count(), 5);
    BOOST_REQUIRE_EQUAL(matrix.get_column_count(), 4);

    BOOST_REQUIRE_EQUAL(matrix.get_row_count(), matrix_2.get_row_count());
    BOOST_REQUIRE_EQUAL(matrix.get_column_count(), matrix_2.get_column_count());
}

BOOST_AUTO_TEST_CASE(vector_segments)
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

BOOST_AUTO_TEST_CASE(vectors)
{
    /* init */
    auto vector_1 = scylla_blas::vector<float>::init_and_return(session, 0, 2*DEFAULT_BLOCK_SIZE+1);

    BOOST_REQUIRE_EQUAL(vector_1.get_segment_count(), 3);

    /* update_values */
    std::vector<scylla_blas::vector_value<float>> values_1;
    for (int i = 1; i <= vector_1.get_length(); i++) {
        values_1.emplace_back(i, M_PI);
    }
    vector_1.update_values(values_1);

    BOOST_REQUIRE_EQUAL(std::ceil(vector_1.get_value(1) * 10000), std::ceil(M_PI * 10000));
    BOOST_REQUIRE_EQUAL(std::ceil(vector_1.get_value(vector_1.get_length()) * 10000), std::ceil(M_PI * 10000));

    /* get_segment */
    auto seg_1 = vector_1.get_segment(2);

    BOOST_REQUIRE_EQUAL(std::ceil(seg_1[0].value * 10000), std::ceil(M_PI * 10000));
    BOOST_REQUIRE_EQUAL(std::ceil(seg_1[vector_1.get_block_size()-1].value * 10000), std::ceil(M_PI * 10000));

    BOOST_REQUIRE_EQUAL(seg_1[0].index, 1);
    BOOST_REQUIRE_EQUAL(seg_1[vector_1.get_block_size()-1].index, vector_1.get_block_size());

    BOOST_REQUIRE_EQUAL(seg_1.size(), vector_1.get_block_size());

    /* get_segment last */
    auto seg_end = vector_1.get_segment(3);

    BOOST_REQUIRE_EQUAL(seg_end.size(), 1);

    /* update_value */
    vector_1.update_value(1, M_E);
    vector_1.update_value(2, 0);

    BOOST_REQUIRE_EQUAL(std::ceil(vector_1.get_value(1) * 10000), std::ceil(M_E * 10000));
    BOOST_REQUIRE_EQUAL(vector_1.get_value(2), 0);

    /* update_segment */
    scylla_blas::vector_segment<float> seg_2;
    seg_2.emplace_back(1, M_E);
    seg_2.emplace_back(2, 0);
    vector_1.update_segment(2, seg_2);

    BOOST_REQUIRE_EQUAL(std::ceil(vector_1.get_value(vector_1.get_block_size()+1) * 10000), std::ceil(M_E * 10000));
    BOOST_REQUIRE_EQUAL(vector_1.get_value(vector_1.get_block_size()+2), 0);
    BOOST_REQUIRE_EQUAL(vector_1.get_value(vector_1.get_block_size()+3), 0);

    /* get_segment with zeros */
    auto seg_3 = vector_1.get_segment(2);

    BOOST_REQUIRE_EQUAL(std::ceil(seg_3[0].value * 10000), std::ceil(M_E * 10000));
    BOOST_REQUIRE_EQUAL(seg_3[0].index, 1);

    BOOST_REQUIRE_EQUAL(seg_3.size(), 1);

    /* update_values overwriting */
    std::vector<scylla_blas::vector_value<float>> values_2;
    values_2.emplace_back(1, 0);
    values_2.emplace_back(2, M_PI);
    values_2.emplace_back(3, M_E);
    vector_1.update_values(values_2);

    BOOST_REQUIRE_EQUAL(vector_1.get_value(1), 0);
    BOOST_REQUIRE_EQUAL(std::ceil(vector_1.get_value(2) * 10000), std::ceil(M_PI * 10000));
    BOOST_REQUIRE_EQUAL(std::ceil(vector_1.get_value(3) * 10000), std::ceil(M_E * 10000));

    /* clear */
    vector_1.clear_value(2);

    BOOST_REQUIRE_EQUAL(vector_1.get_value(2), 0);

    vector_1.clear_segment(1);

    BOOST_REQUIRE_EQUAL(vector_1.get_value(3), 0);
}

BOOST_AUTO_TEST_SUITE_END();
