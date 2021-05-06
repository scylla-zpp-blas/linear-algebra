#include <boost/test/unit_test.hpp>

#include "../test_utils.hh"
#include "../fixture.hh"
#include "scylla_blas/queue/worker_proc.hh"
#include "../vector_utils.hh"

BOOST_FIXTURE_TEST_CASE(float_vector_scale_IT, vector_fixture)
{
    // Given vector of 4 floats
    std::vector<float> vals = {1.6f, 2.9999f, 3.0f, 0.0f};
    auto vector = getScyllaVectorOf(vals, 0);

    // When performing scaling by 2
    scheduler->sscal(2, *vector);

    // Then result vector is scaled by 2.
    std::vector<float> vals2 = {1.6f * 2, 2.9999f * 2, 3.0f * 2, 0.0f * 2};
    std::optional<scylla_blas::vector_value<float>> difference = cmp_vector(*vector, vals2);
    BOOST_CHECK(!difference.has_value());
    if (difference.has_value()) {
        BOOST_ERROR(fmt::format("Difference at position {0}, {1} - {2}",
                                difference->index,
                                difference->value,
                                vals2[difference->index - 1]));
    }
}

BOOST_FIXTURE_TEST_CASE(double_vector_scale_IT, vector_fixture)
{
    // Given vector of 4 doubles
    std::vector<double> vals = {1.6, 2.999999, 3.0, 3.141592653589793238462643383};
    auto vector = getScyllaVectorOf(vals, 0);

    // When performing scaling by 59.49
    const double alpha = 59.05;
    scheduler->dscal(alpha, *vector);

    // Then result vector is scaled by 59.49.
    std::vector<double> vals2 = {
            1.6 * alpha,
            2.999999 * alpha,
            3.0 * alpha,
            3.141592653589793238462643383 * alpha};
    std::optional<scylla_blas::vector_value<double>> difference = cmp_vector(*vector, vals2);
    BOOST_CHECK(!difference.has_value());
    if (difference.has_value()) {
        BOOST_ERROR(fmt::format("Difference at position {0}, {1} - {2}",
                                difference->index,
                                difference->value,
                                vals2[difference->index - 1]));
    }
}
