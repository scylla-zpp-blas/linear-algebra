#include <boost/test/unit_test.hpp>

#include "scylla_blas/scylla_blas.hh"
#include "../fixture.hh"

namespace {
template<class T>
void print_vector(const scylla_blas::vector<T>& vec) {
    auto whole = vec.get_whole();

    std::cout << "Vector " << vec.id << ": ";
    for (auto entry : whole) {
        std::cout << "(" << entry.index << "-> " << entry.value << "), ";
    }
    std::cout << std::endl;
}
}

BOOST_FIXTURE_TEST_CASE(vector_copy, vector_fixture)
{
    print_vector(*float_B);
    scheduler->scopy(*float_A, *float_B);
    print_vector(*float_A);
    print_vector(*float_B);

    print_vector(*double_B);
    scheduler->dcopy(*double_A, *double_B);
    print_vector(*double_A);
    print_vector(*double_B);
}

BOOST_FIXTURE_TEST_CASE(vector_alpha_sum_copy, vector_fixture)
{
    print_vector(*float_B);
    scheduler->saxpy(3.14, *float_A, *float_B);
    print_vector(*float_A);
    print_vector(*float_B);

    print_vector(*double_B);
    scheduler->daxpy(3.14, *double_A, *double_B);
    print_vector(*double_A);
    print_vector(*double_B);
}