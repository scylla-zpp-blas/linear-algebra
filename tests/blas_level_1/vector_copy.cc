#include <boost/test/unit_test.hpp>

#include "../test_utils.hh"
#include "../fixture.hh"
#include "../vector_utils.hh"

BOOST_FIXTURE_TEST_CASE(float_vector_copy, vector_fixture)
{
    // Given vector of five values and and vector with different id.
    std::vector<float> values = {4.234f, 3214.4243f, 290342.0f, 0.0f, 1.23456789f};
    auto vector1 = getScyllaVectorOf(test_const::float_vector_1_id, values);
    auto vector2 = getScyllaVector(test_const::float_vector_2_id);

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

BOOST_FIXTURE_TEST_CASE(float_vector_copy_to_same, vector_fixture)
{
    // Given vector of five values and and vector with different id.
    std::vector<float> values = {4.234f, 3214.4243f, 290342.0f, 0.0f, 1.23456789f};
    auto vector1 = getScyllaVectorOf(test_const::float_vector_1_id, values);

    // When performing copy from first vector to another.
    scheduler->scopy(*vector1, *vector1);

    // Then the second vector has the values of first.
    std::optional<scylla_blas::vector_value<float>> difference = cmp_vector(*vector1, values);
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
    auto vector1 = getScyllaVectorOf(test_const::double_vector_1_id, values);
    auto vector2 = getScyllaDoubleVector(test_const::double_vector_2_id);

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
    auto vector1 = getScyllaVectorOf(test_const::float_vector_1_id, values1);
    auto vector2 = getScyllaVectorOf(test_const::float_vector_2_id, values2);
    float alpha = 1000.0f;

    // When performing saxpy from first vector to another.
    // saxpy - constant times a vector plus a vector.
    scheduler->saxpy(alpha, *vector1, *vector2);

    // Then the second vector has the result.
    while (values2.size() < values1.size()) {
        values2.push_back(0);
    }
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
}

BOOST_FIXTURE_TEST_CASE(double_vector_alpha_sum_copy, vector_fixture)
{
    // Given two vectors with some values and alpha.
    std::vector<double> values1 = {4.4327805450, 3214.4243, 290342.0, 0.0, 1.23456789};
    std::vector<double> values2 = {4.5, 324.4243, 342.29, 0.0, 23442.252526, 1.2346981958};
    auto vector1 = getScyllaVectorOf(test_const::double_vector_1_id, values1);
    auto vector2 = getScyllaVectorOf(test_const::double_vector_2_id, values2);
    float alpha = 1000.0f;

    // When performing daxpy from first vector to another.
    // daxpy - constant times a vector plus a vector.
    scheduler->daxpy(alpha, *vector1, *vector2);

    // Then the second vector has the result.
    while (values2.size() < values1.size()) {
        values2.push_back(0);
    }
    for (int i = 0; i < values1.size(); i++) {
        values2[i] += values1[i] * alpha;
    }

    print_vector(*vector2);
    std::optional<scylla_blas::vector_value<double>> difference = cmp_vector(*vector2, values2);
    BOOST_CHECK(!difference.has_value());
    if (difference.has_value()) {
        BOOST_ERROR(fmt::format("Difference at position {0}, {1} - {2}",
                                difference->index,
                                difference->value,
                                values2[difference->index - 1]));
    }
}


BOOST_FIXTURE_TEST_CASE(double_vector_alpha_sum_copy_to_same, vector_fixture)
{
    // Given two vectors with some values and alpha.
    std::vector<double> values1 = {4.4327805450, 3214.4243, 290342.0, 0.0, 1.23456789};
    auto vector1 = getScyllaVectorOf(test_const::double_vector_1_id, values1);
    float alpha = 1000.0f;

    // When performing daxpy from first vector to another.
    // daxpy - constant times a vector plus a vector.
    scheduler->daxpy(alpha, *vector1, *vector1);

    // Then the second vector has the result.

    for (int i = 0; i < values1.size(); i++) {
        values1[i] += values1[i] * alpha;
    }

    print_vector(*vector1);
    std::optional<scylla_blas::vector_value<double>> difference = cmp_vector(*vector1, values1);
    BOOST_CHECK(!difference.has_value());
    if (difference.has_value()) {
        BOOST_ERROR(fmt::format("Difference at position {0}, {1} - {2}",
                                difference->index,
                                difference->value,
                                values1[difference->index - 1]));
    }
}