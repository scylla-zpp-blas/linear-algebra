#include <boost/test/unit_test.hpp>

#include <scmd.hh>

#include "value_factory.hh"
#include "sparse_matrix_value_generator.hh"
#include "../test_utils.hh"
#include "../fixture.hh"

BOOST_FIXTURE_TEST_SUITE(multiply_tests, matrix_fixture)

BOOST_AUTO_TEST_CASE(float_mm)
{
    using namespace scylla_blas;

    // print_matrix(*float_6x5);
    // print_matrix(*float_5x6);

    std::cerr << "Multiply" << std::endl;
    float_BxB->clear_all();
    scheduler->sgemm(NoTrans, NoTrans, 1, *float_BxA, *float_AxB, 1, *float_BxB);
    // print_matrix(*float_6x6);

    std::cerr << "Multiply with coeff (2.5)" << std::endl;
    float_BxB->clear_all();
    scheduler->sgemm(NoTrans, NoTrans, 2.5, *float_BxA, *float_AxB, 1, *float_BxB);
    // print_matrix(*float_6x6);

    std::cerr << "Multiply and add 3 times result" << std::endl;
    float_BxB->clear_all();
    scheduler->sgemm(NoTrans, NoTrans, 1, *float_BxA, *float_AxB, 1, *float_BxB);
    scheduler->sgemm(NoTrans, NoTrans, 1, *float_BxA, *float_AxB, 3, *float_BxB);
    // print_matrix(*float_6x6);

    std::cerr << "Multiply transA" << std::endl;
    float_BxB->clear_all();
    scheduler->sgemm(Trans, NoTrans, 1, *float_AxB, *float_AxB, 1, *float_BxB);
    // print_matrix(*float_6x6);

    std::cerr << "Multiply transB" << std::endl;
    float_BxB->clear_all();
    scheduler->sgemm(NoTrans, Trans, 1, *float_BxA, *float_BxA, 1, *float_BxB);
    // print_matrix(*float_6x6);

    std::cerr << "Multiply transAB" << std::endl;
    float_BxB->clear_all();
    scheduler->sgemm(Trans, Trans, 1, *float_AxB, *float_BxA, 1, *float_BxB);
    // print_matrix(*float_6x6);
}

BOOST_AUTO_TEST_SUITE_END();