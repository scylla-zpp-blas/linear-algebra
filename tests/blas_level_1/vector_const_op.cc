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
    std::cout << std::setprecision(20) << sum << "=sum\n";
    std::cout << std::setprecision(20) << res << "=res\n";

    // Then the dot product is correctly calculated and equal to sum.
    BOOST_CHECK(abs(sum - res) < scylla_blas::epsilon);
//    /* DOT */
//    std::cout << "sdot(float_A, float_B) = " << scheduler->sdot(*float_A, *float_A2) << std::endl;
//    std::cout << "ddot(double_A, double_B) = " << scheduler->ddot(*double_A, *double_A2) << std::endl;
//    std::cout << "sdsdot(0.5, float_A, float_B) = " << scheduler->sdsdot(0.5, *float_A, *float_A2) << std::endl;
//    std::cout << "dsdot(float_A, float_B) = " << scheduler->dsdot(*float_A, *float_A2) << std::endl;
//
//    /* One-vector-ops */
//    std::cout << "Norm2 of float_A: " << scheduler->snrm2(*float_A) << std::endl;
//    std::cout << "Sum of modules of float_A: " << scheduler->sasum(*float_A) << std::endl;
//    std::cout << "Max value index of float_A: " << scheduler->isamax(*float_A) << std::endl;
//
//    std::cout << "Norm2 of double_A: " << scheduler->dnrm2(*double_A) << std::endl;
//    std::cout << "Sum of modules of double_A: " << scheduler->dasum(*double_A) << std::endl;
//    std::cout << "Max value index of double_A: " << scheduler->idamax(*double_A) << std::endl;
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
    // Given two vector of five values.
    std::vector<float> values1 = {0.00494931f, 0.119193f, 0.927604f, 0.354004f};
    auto vector1 = getScyllaVectorOf(test_const::float_vector_1_id, values1);

    // When performing dot product of these two vectors.
    float res = scheduler->snrm2(*vector1);

    float nrm = 0;
    for (int i = 0; i < values1.size(); i++) {
        nrm += values1[i] * values1[i];
    }
    nrm = sqrt(nrm);
    std::cout << std::setprecision(20) << nrm << "=sum\n";
    std::cout << std::setprecision(20) << res << "=res\n";

    // Then the dot product is correctly calculated and equal to sum.
    BOOST_CHECK(abs(nrm - res) < scylla_blas::epsilon);
}

BOOST_FIXTURE_TEST_CASE(vector_norm_double, vector_fixture)
{
    // Given two vector of five values.
    std::vector<double> values1 = {0.00494931, 0.119193, 0.927604, 0.354004};
    auto vector1 = getScyllaVectorOf(test_const::double_vector_1_id, values1);

    // When performing dot product of these two vectors.
    double res = scheduler->dnrm2(*vector1);

    double nrm = 0;
    for (int i = 0; i < values1.size(); i++) {
        nrm += values1[i] * values1[i];
    }
    nrm = sqrt(nrm);
    std::cout << std::setprecision(20) << nrm << "=sum\n";
    std::cout << std::setprecision(20) << res << "=res\n";

    // Then the dot product is correctly calculated and equal to sum.
    BOOST_CHECK(abs(nrm - res) < scylla_blas::epsilon);
}