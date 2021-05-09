#include <boost/test/unit_test.hpp>

#include "../test_utils.hh"
#include "../fixture.hh"

BOOST_FIXTURE_TEST_CASE(vector_scale, vector_fixture)
{
    scheduler->sscal(2, *float_A);
    print_vector(*float_A);

    scheduler->dscal(2, *double_A);
    print_vector(*double_A);
}