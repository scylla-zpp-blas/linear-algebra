#include <boost/test/unit_test.hpp>

#include "scylla_blas/scylla_blas.hh"
#include "../fixture.hh"

BOOST_FIXTURE_TEST_CASE(vector_constant_operations, vector_fixture)
{
    /* DOT */
    std::cout << "sdot(float_A, float_B) = " << scheduler->sdot(*float_A, *float_B) << std::endl;
    std::cout << "ddot(double_A, double_B) = " << scheduler->ddot(*double_A, *double_B) << std::endl;
    std::cout << "sdsdot(0.5, float_A, float_B) = " << scheduler->sdsdot(0.5, *float_A, *float_B) << std::endl;
    std::cout << "dsdot(float_A, float_B) = " << scheduler->dsdot(*float_A, *float_B) << std::endl;

    /* One-vector-ops */
    std::cout << "Norm2 of float_A: " << scheduler->snrm2(*float_A) << std::endl;
    std::cout << "Sum of modules of float_A: " << scheduler->sasum(*float_A) << std::endl;
    std::cout << "Max value index of float_A: " << scheduler->isamax(*float_A) << std::endl;
    
    std::cout << "Norm2 of double_A: " << scheduler->dnrm2(*double_A) << std::endl;
    std::cout << "Sum of modules of double_A: " << scheduler->dasum(*double_A) << std::endl;
    std::cout << "Max value index of double_A: " << scheduler->idamax(*double_A) << std::endl;
}