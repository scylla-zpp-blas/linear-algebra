#include <boost/test/unit_test.hpp>

#include "../test_utils.hh"
#include "../fixture.hh"
#include "../vector_utils.hh"

BOOST_FIXTURE_TEST_CASE(vector_swap, vector_fixture)
{
    // Given two vectors with some values.
    std::vector<float> values1 = {4.0f, 3214.4243f, 290342.0f, 0.0f, 1.23456789f};
    std::vector<float> values2 = {4.5f, 324.4243f, 342.29f, 0.0f};
    auto vector1 = getScyllaVectorOf(test_const::float_vector_1_id, values1);
    auto vector2 = getScyllaVectorOf(test_const::float_vector_2_id, values2);

    // When performing sswap(v1, v2)
    scheduler->sswap(*vector1, *vector2);

    // Then values between vectors are swapped.
    std::optional<scylla_blas::vector_value<float>> difference1 = cmp_vector(*vector1, values2);
    BOOST_CHECK(!difference1.has_value());
    if (difference1.has_value()) {
        BOOST_ERROR(fmt::format("Difference at position {0}, {1} - {2}",
                                difference1->index,
                                difference1->value,
                                values2[difference1->index - 1]));
    }

    std::optional<scylla_blas::vector_value<float>> difference2 = cmp_vector(*vector2, values1);
    BOOST_CHECK(!difference2.has_value());
    if (difference2.has_value()) {
        BOOST_ERROR(fmt::format("Difference at position {0}, {1} - {2}",
                                difference2->index,
                                difference2->value,
                                values1[difference2->index - 1]));
    }
}


BOOST_FIXTURE_TEST_CASE(vector_swap_double, vector_fixture)
{
    // Given two vectors with some values.
    std::vector<double> values1 = {4.4327805450, 3214.4243, 290342.0, 0.0, 1.23456789};
    std::vector<double> values2 = {4.5, 324.4243, 342.29, 0.0, 23442.252526, 1.2346981958};
    auto vector1 = getScyllaVectorOf(test_const::double_vector_1_id, values1);
    auto vector2 = getScyllaVectorOf(test_const::double_vector_2_id, values2);

    // When performing sswap(v1, v2)
    scheduler->dswap(*vector1, *vector2);

    // Then values between vectors are swapped.
    std::optional<scylla_blas::vector_value<double>> difference1 = cmp_vector(*vector1, values2);
    BOOST_CHECK(!difference1.has_value());
    if (difference1.has_value()) {
        BOOST_ERROR(fmt::format("Difference at position {0}, {1} - {2}",
                                difference1->index,
                                difference1->value,
                                values2[difference1->index - 1]));
    }

    std::optional<scylla_blas::vector_value<double>> difference2 = cmp_vector(*vector2, values1);
    BOOST_CHECK(!difference2.has_value());
    if (difference2.has_value()) {
        BOOST_ERROR(fmt::format("Difference at position {0}, {1} - {2}",
                                difference2->index,
                                difference2->value,
                                values1[difference2->index - 1]));
    }
}
