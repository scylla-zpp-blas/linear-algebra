#include <boost/test/unit_test.hpp>

#include "../test_utils.hh"
#include "../fixture.hh"

BOOST_FIXTURE_TEST_CASE(vector_copy, vector_fixture)
{
    print_vector(*float_A2);
    scheduler->scopy(*float_A, *float_A2);
    print_vector(*float_A);
    print_vector(*float_A2);

    std::cout << "Auto copy shouldn't modify anything" << std::endl;
    scheduler->scopy(*float_A, *float_A);
    print_vector(*float_A);

    print_vector(*double_A2);
    scheduler->dcopy(*double_A, *double_A2);
    print_vector(*double_A);
    print_vector(*double_A2);

    std::cout << "Auto copy shouldn't modify anything" << std::endl;
    scheduler->dcopy(*double_A, *double_A);
    print_vector(*double_A);
}

BOOST_FIXTURE_TEST_CASE(vector_alpha_sum_copy, vector_fixture)
{
    print_vector(*float_A2);
    scheduler->saxpy(3.14, *float_A, *float_A2);
    print_vector(*float_A);
    print_vector(*float_A2);

    print_vector(*double_A);
    scheduler->daxpy(3.14, *double_A, *double_A2);
    print_vector(*double_A);
    print_vector(*double_A2);
}