#include <boost/test/unit_test.hpp>

#include "scylla_blas/scylla_blas.hh"
#include "../fixture.hh"
#include "vector_utilities.hh"

BOOST_FIXTURE_TEST_CASE(float_vector_copy, vector_fixture)
{
    // Given vector of five values and and vector with different id.
    std::vector<float> values = {4.234f, 3214.4243f, 290342.0f, 0.0f, 1.23456789f};
    auto vector1 = getScyllaVectorOf(values, 0);
    auto vector2 = getScyllaVector(1);

    // When performing copy from first vector to another.
    scheduler->scopy(*vector1, *vector2);

    // Then the second vector has the values of first.
    std::optional<scylla_blas::vector_value<float>> difference = cmp_vector(*vector2, values);
    BOOST_CHECK(!difference.has_value());
    if (difference.has_value()) {
        BOOST_ERROR(fmt::format("Difference at position {0}, {1} - {2}",
                                difference->index,
                                difference->value,
                                values[difference->index - 1]));
    }
}

BOOST_FIXTURE_TEST_CASE(double_vector_copy, vector_fixture)
{
    // Given vector of six values and and vector with different id.
    std::vector<double> values = {5355.939194952, 4.234, 3214.4243, 290342.0, 0.0, 1.23456789};
    auto vector1 = getScyllaVectorOf(values, 0);
    auto vector2 = getScyllaDoubleVector(1);

    // When performing copy from first vector to another.
    scheduler->dcopy(*vector1, *vector2);

    // Then the second vector has the values of first.
    std::optional<scylla_blas::vector_value<double>> difference = cmp_vector(*vector2, values);
    BOOST_CHECK(!difference.has_value());
    if (difference.has_value()) {
        BOOST_ERROR(fmt::format("Difference at position {0}, {1} - {2}",
                                difference->index,
                                difference->value,
                                values[difference->index - 1]));
    }
}

BOOST_FIXTURE_TEST_CASE(float_vector_alpha_sum_copy, vector_fixture)
{
    // Given two vectors with some values and alpha.
    std::vector<float> values1 = {4.0f, 3214.4243f, 290342.0f, 0.0f, 1.23456789f};
    std::vector<float> values2 = {4.5f, 324.4243f, 342.29f, 0.0f};
    auto vector1 = getScyllaVectorOf(values1, 0);
    auto vector2 = getScyllaVectorOf(values2, 1);
    float alpha = 1000.0f;

    // When performing saxpy from first vector to another.
    // saxpy - constant times a vector plus a vector.
    scheduler->saxpy(alpha, *vector1, *vector2);

    // Then the second vector has the result.
    for (int i = 0; i < values1.size(); i++) {
        values2[i] += values1[i] * alpha;
    }
    print_vector(*vector2);
    std::optional<scylla_blas::vector_value<float>> difference = cmp_vector(*vector2, values2);
    BOOST_CHECK(!difference.has_value());
    if (difference.has_value()) {
        BOOST_ERROR(fmt::format("Difference at position {0}, {1} - {2}",
                                difference->index,
                                difference->value,
                                values2[difference->index - 1]));
    }
//    print_vector(*float_B);
//    scheduler->saxpy(3.14, *float_A, *float_B);
//    print_vector(*float_A);
//    print_vector(*float_B);
//
//    print_vector(*double_B);
//    scheduler->daxpy(3.14, *double_A, *double_B);
//    print_vector(*double_A);
//    print_vector(*double_B);
}

BOOST_FIXTURE_TEST_CASE(double_vector_alpha_sum_copy, vector_fixture)
{
//    print_vector(*float_B);
//    scheduler->saxpy(3.14, *float_A, *float_B);
//    print_vector(*float_A);
//    print_vector(*float_B);
//
//    print_vector(*double_B);
//    scheduler->daxpy(3.14, *double_A, *double_B);
//    print_vector(*double_A);
//    print_vector(*double_B);
}