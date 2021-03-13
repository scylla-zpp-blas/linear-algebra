#include <boost/test/unit_test.hpp>

#include <scmd.hh>

#include "generators/value_factory.hh"
#include "generators/sparse_matrix_value_generator.hh"
#include "scylla_blas/scylla_blas.hh"
#include "fixture.hh"


BOOST_FIXTURE_TEST_SUITE(multiply_tests, scylla_fixture)

BOOST_AUTO_TEST_CASE(multiply)
{
    std::shared_ptr<scylla_blas::value_factory<float>> f = std::make_shared<scylla_blas::value_factory<float>>(0, 9,
                                                                                                               1410);
    scylla_blas::sparse_matrix_value_generator<float> gen1(5, 5, 10, 11121, f);
    scylla_blas::sparse_matrix_value_generator<float> gen2(5, 5, 10, 22222, f);
    auto result = scylla_blas::multiply(session, gen1, gen2);
}

BOOST_AUTO_TEST_CASE(easy_multiply)
{
    std::shared_ptr<scylla_blas::value_factory<float>> f = std::make_shared<scylla_blas::value_factory<float>>(0, 9,
                                                                                                               1410);
    scylla_blas::sparse_matrix_value_generator<float> gen1(5, 5, 10, 11121, f);
    scylla_blas::sparse_matrix_value_generator<float> gen2(5, 5, 10, 22222, f);
    auto result = scylla_blas::easy_multiply(session, gen1, gen2);
}

BOOST_AUTO_TEST_SUITE_END();