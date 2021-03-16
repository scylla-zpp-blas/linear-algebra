#include <boost/test/unit_test.hpp>

#include <scmd.hh>

#include "generators/value_factory.hh"
#include "generators/sparse_matrix_value_generator.hh"
#include "scylla_blas/scylla_blas.hh"
#include "fixture.hh"


BOOST_FIXTURE_TEST_SUITE(multiply_tests, scylla_fixture)

BOOST_AUTO_TEST_CASE(naive_multiply)
{
    std::shared_ptr<scylla_blas::value_factory<float>> f =
            std::make_shared<scylla_blas::value_factory<float>>(0, 9,1111);

    scylla_blas::sparse_matrix_value_generator<float> gen1(5, 5, 10, 42, f);
    scylla_blas::sparse_matrix_value_generator<float> gen2(5, 5, 10, 44, f);
    auto result = scylla_blas::naive_multiply(session, gen1, gen2);
}

BOOST_AUTO_TEST_CASE(parallel_multiply)
{
    std::shared_ptr <scylla_blas::value_factory<float>> f =
            std::make_shared < scylla_blas::value_factory < float >> (0, 9, 1111);

    scylla_blas::sparse_matrix_value_generator<float> gen1(5, 5, 10, 42, f);
    scylla_blas::sparse_matrix_value_generator<float> gen2(5, 5, 10, 44, f);
    auto result = scylla_blas::parallel_multiply(session, gen1, gen2);
}

BOOST_AUTO_TEST_SUITE_END();