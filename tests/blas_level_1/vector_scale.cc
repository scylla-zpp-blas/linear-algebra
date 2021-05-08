#include <boost/test/unit_test.hpp>

#include "../fixture.hh"
#include "scylla_blas/queue/worker_proc.hh"

namespace {
template<class T>
void print_vector(const scylla_blas::vector<T> &vec) {
    auto whole = vec.get_whole();

    std::cout << "Vector " << vec.id << ": ";
    for (auto entry : whole) {
        std::cout << "(" << entry.index << "-> " << entry.value << "), ";
    }
    std::cout << std::endl;
}

template <class T>
inline void assert_zeros(std::vector<T> &values, int start, int end) {
    for (int i = start; i < end; i++) {
        BOOST_CHECK(abs(values[i]) < scylla_blas::epsilon);
    }
}

template<class T>
void cmp_vector(const scylla_blas::vector<T> &vec, std::vector<T> values) {
    auto whole = vec.get_whole();

    int last = -1;
    for (auto entry : whole) {
        if (last + 1 != entry.index - 1) {
            assert_zeros(values, last + 1, entry.index - 1);
        }
        BOOST_CHECK(abs(entry.value - values[entry.index - 1]) < scylla_blas::epsilon);
        last = entry.index - 1;
    }
}


BOOST_FIXTURE_TEST_CASE(float_vector_scale_IT, vector_fixture)
{
    // Given vector of 3 floats
    std::vector<float> vals = {1.6f, 2.9999f, 3.0f};
    auto vector = getScyllaVectorOf(vals);

    // When performing scaling by 2
    scheduler->sscal(2, *vector);

    // Then result vector is scaled by 2.
    std::vector<float> valsx2 = {1.6f * 2, 2.9999f * 2, 3.0f * 2};
    cmp_vector(*vector, valsx2);
}

BOOST_FIXTURE_TEST_CASE(double_vector_scale_IT, vector_fixture)
{
    // Given vector of 4 doubles
    std::vector<double> vals = {1.6, 2.999999, 3.0, 3.141592653589793238462643383};
    auto vector = getScyllaVectorOf(vals);

    // When performing scaling by 59.49
    const double alpha = 59.05;
    scheduler->dscal(alpha, *vector);

    // Then result vector is scaled by 59.49.
    std::vector<double> valsx2 = {
            1.6 * alpha,
            2.999999 * alpha,
            3.0 * alpha,
            3.141592653589793238462643383 * alpha};
    cmp_vector(*vector, valsx2);
}


}
