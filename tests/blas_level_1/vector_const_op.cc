#include <cmath>
#include <boost/test/unit_test.hpp>

#include "../test_utils.hh"
#include "../fixture.hh"
#include "../vector_utils.hh"

BOOST_FIXTURE_TEST_CASE(vector_dot_float, vector_fixture)
{
    // Given two vector of five values.
    std::vector<float> values1 = {4.234f, 3214.4243f, 290342.0f, 0.0f, -1.0f};
    std::vector<float> values2 = {3.0f, 392.9001f, 0.005f, 5.0f, 29844.05325811f};
    auto vector1 = getScyllaVectorOf(test_const::float_vector_1_id, values1);
    auto vector2 = getScyllaVectorOf(test_const::float_vector_2_id, values2);

    // When performing dot product of these two vectors.
    float res = scheduler->sdot(*vector1, *vector2);

    float sum = 0;
    for (int i = 0; i < values1.size(); i++) {
        sum += values1[i] * values2[i];
    }

    print_vector(*vector1);
    print_vector(*vector2);
    std::cout << std::setprecision(20) << sum << "=sum\n";
    std::cout << std::setprecision(20) << res << "=res\n";

    // Then the dot product is correctly calculated and equal to sum.
    BOOST_CHECK(abs(sum - res) < scylla_blas::epsilon);
}

BOOST_FIXTURE_TEST_CASE(vector_dot_float_same_obj, vector_fixture)
{
    // Given one vector of five values.
    std::vector<float> values1 = {4.234f, 214.4243f, 342.0f, 0.0f, -1.0f};
    auto vector1 = getScyllaVectorOf(test_const::float_vector_1_id, values1);

    // When performing dot product of this vector x this vector.
    float res = scheduler->sdot(*vector1, *vector1);

    float sum = 0;
    for (float v : values1) {
        sum += v * v;
    }
    std::cout << std::setprecision(20) << sum << "=sum\n";
    std::cout << std::setprecision(20) << res << "=res\n";

    // Then the dot product is correctly calculated and equal to sum.
    BOOST_CHECK(abs(sum - res) < scylla_blas::epsilon);
}

BOOST_FIXTURE_TEST_CASE(vector_sdsdot_float, vector_fixture)
{
    // Given two vector of five values.
    std::vector<float> values1 = {4.234f, 3214.4243f, 290342.0f, 0.0f, -1.0f};
    std::vector<float> values2 = {3.0f, 392.9001f, 0.005f, 5.0f, 29844.05325811f};
    auto vector1 = getScyllaVectorOf(test_const::float_vector_1_id, values1);
    auto vector2 = getScyllaVectorOf(test_const::float_vector_2_id, values2);

    // When performing dot product of double precision plus value
    // of these two vectors.
    float res = scheduler->sdsdot(0.5f, *vector1, *vector2);

    double sum = 0.5f;
    for (int i = 0; i < values1.size(); i++) {
        sum += (double)values1[i] * (double)values2[i];
    }
    std::cout << std::setprecision(20) << sum << "=sum\n";
    std::cout << std::setprecision(20) << res << "=res\n";

    // Then the dot product is correctly calculated and equal to sum + value.
    BOOST_CHECK(abs((float)sum - res) < scylla_blas::epsilon);
}

BOOST_FIXTURE_TEST_CASE(vector_dsdot_float, vector_fixture)
{
    // Given two vector of five values.
    std::vector<float> values1 = {4.234f, 3214.4243f, 290342.0f, 0.0f, -1.0f};
    std::vector<float> values2 = {3.0f, 392.9001f, 0.005f, 5.0f, 29844.05325811f};
    auto vector1 = getScyllaVectorOf(test_const::float_vector_1_id, values1);
    auto vector2 = getScyllaVectorOf(test_const::float_vector_2_id, values2);

    // When performing dot product with double precision of these two vectors.
    double res = scheduler->dsdot(*vector1, *vector2);

    double sum = 0.0f;
    for (int i = 0; i < values1.size(); i++) {
        sum += (double)values1[i] * (double)values2[i];
    }
    std::cout << std::setprecision(20) << sum << "=sum\n";
    std::cout << std::setprecision(20) << res << "=res\n";

    // Then the dot product is correctly calculated and equal to sum.
    BOOST_CHECK(abs(sum - res) < scylla_blas::epsilon);
}

BOOST_FIXTURE_TEST_CASE(vector_dot_double, vector_fixture)
{
    // Given two vector of five values.
    std::vector<double> values1 = {4.234, 3214.4243, 290342.0, 0.0, -1.0};
    std::vector<double> values2 = {3.0, 392.9001, 0.005, 5.0, 29844.05325811};
    auto vector1 = getScyllaVectorOf(test_const::double_vector_1_id, values1);
    auto vector2 = getScyllaVectorOf(test_const::double_vector_2_id, values2);

    // When performing dot product of these two vectors.
    double res = scheduler->ddot(*vector1, *vector2);

    double sum = 0;
    for (int i = 0; i < values1.size(); i++) {
        sum += values1[i] * values2[i];
    }
    std::cout << std::setprecision(20) << sum << "=sum\n";
    std::cout << std::setprecision(20) << res << "=res\n";

    // Then the dot product is correctly calculated and equal to sum.
    BOOST_CHECK(abs(sum - res) < scylla_blas::epsilon);
}


BOOST_FIXTURE_TEST_CASE(vector_norm_float, vector_fixture)
{
    // Given vector of some values.
    std::vector<float> values1 = {0.00494931f, 0.119193f, 0.927604f, 0.354004f};
    auto vector1 = getScyllaVectorOf(test_const::float_vector_1_id, values1);

    // When performing vector euclidean norm.
    float res = scheduler->snrm2(*vector1);

    float nrm = 0;
    for (float v : values1) {
        nrm += v * v;
    }
    nrm = std::sqrt(nrm);
    std::cout << std::setprecision(20) << nrm << "=sum\n";
    std::cout << std::setprecision(20) << res << "=res\n";

    // Then the norm is correctly calculated and equal to nrm.
    BOOST_CHECK(abs(nrm - res) < scylla_blas::epsilon);
}

BOOST_FIXTURE_TEST_CASE(vector_sasum_float, vector_fixture)
{
    // Given vector of some values.
    std::vector<float> values1 = {0.00494931f, 0.119193f, -0.927604f, 0.354004f};
    auto vector1 = getScyllaVectorOf(test_const::float_vector_1_id, values1);

    // When performing sum of absolute values in this vector.
    float res = scheduler->sasum(*vector1);

    float abs_sum = 0;
    for (float v : values1) {
        abs_sum += abs(v);
    }
    std::cout << std::setprecision(20) << abs_sum << "=sum\n";
    std::cout << std::setprecision(20) << res << "=res\n";

    // Then it is correctly calculated and equal to sum.
    BOOST_CHECK(abs(abs_sum - res) < scylla_blas::epsilon);
}

BOOST_FIXTURE_TEST_CASE(vector_isamax_float, vector_fixture)
{
    // Given vector of some values.
    std::vector<float> values1 = {0.00494931f, 0.119193f, -0.927604f, 0.354004f};
    auto vector1 = getScyllaVectorOf(test_const::float_vector_1_id, values1);

    // When performing max fetch on the abs values in this vector.
    scylla_blas::index_type res = scheduler->isamax(*vector1);

    scylla_blas::index_type max_index = 0;
    for (scylla_blas::index_type i = 1; i < values1.size(); i++) {
        if (abs(values1[i]) > abs(values1[max_index])) {
            max_index = i;
        }
    }
    std::cout << std::setprecision(20) << values1[max_index] << "=sum\n";
    std::cout << res << "=res index\n";

    // Then the found index corresponds to the index of largest absolute value's index.
    BOOST_CHECK(max_index + 1 == res);
}

BOOST_FIXTURE_TEST_CASE(vector_norm_double, vector_fixture)
{
    // Given vector of some values.
    std::vector<double> values1 = {0.00494931, 0.119193, 0.927604, 0.354004};
    auto vector1 = getScyllaVectorOf(test_const::double_vector_1_id, values1);

    // When performing vector euclidean norm.
    double res = scheduler->dnrm2(*vector1);

    double nrm = 0;
    for (double i : values1) {
        nrm += i * i;
    }
    nrm = sqrt(nrm);
    std::cout << std::setprecision(20) << nrm << "=sum\n";
    std::cout << std::setprecision(20) << res << "=res\n";

    // Then the norm is correctly calculated and equal to nrm.
    BOOST_CHECK(abs(nrm - res) < scylla_blas::epsilon);
}

BOOST_FIXTURE_TEST_CASE(vector_dasum_double, vector_fixture)
{
    // Given vector of some values.
    std::vector<double> values1 = {0.00494931, 0.119193, -0.927604, 0.354004};
    auto vector1 = getScyllaVectorOf(test_const::double_vector_1_id, values1);

    // When performing sum of absolute values in this vector.
    double res = scheduler->dasum(*vector1);

    double abs_sum = 0;
    for (double v : values1) {
        abs_sum += abs(v);
    }
    std::cout << std::setprecision(20) << abs_sum << "=sum\n";
    std::cout << std::setprecision(20) << res << "=res\n";

    // Then it is correctly calculated and equal to sum.
    BOOST_CHECK(abs(abs_sum - res) < scylla_blas::epsilon);
}

BOOST_FIXTURE_TEST_CASE(vector_idamax_double, vector_fixture)
{
    // Given vector of some values.
    std::vector<double> values1 = {0.00494931, 0.119193, -0.927604, 0.354004};
    auto vector1 = getScyllaVectorOf(test_const::double_vector_1_id, values1);

    // When performing max fetch on the abs values in this vector.
    scylla_blas::index_type res = scheduler->idamax(*vector1);

    scylla_blas::index_type max_index = 0;
    for (scylla_blas::index_type i = 1; i < values1.size(); i++) {
        if (abs(values1[i]) > abs(values1[max_index])) {
            max_index = i;
        }
    }
    std::cout << std::setprecision(20) << values1[max_index] << "=sum\n";
    std::cout << res << "=res index\n";

    // Then the found index corresponds to the index of largest absolute value's index.
    BOOST_CHECK(max_index + 1 == res);
}
