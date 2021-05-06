#include <boost/test/unit_test.hpp>

#include "scylla_blas/scylla_blas.hh"
#include "../fixture.hh"
#include "scylla_blas/queue/worker_proc.hh"
#include "../generators/preset_value_factory.hh"

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
inline bool is_zeros(std::vector<T> &values, int start, int end) {
    bool is_zeros = true;
    for (int i = start; i < end; i++) {
        if (abs(values[i]) > scylla_blas::epsilon)
            is_zeros = false;
    }
    return is_zeros;
}

template<class T>
bool cmp_vector(const scylla_blas::vector<T> &vec, std::vector<T> values) {
    auto whole = vec.get_whole();

    int last = -1;
    for (auto entry : whole) {
        if (last + 1 != entry.index - 1 && !is_zeros(values, last + 1, entry.index - 1)) {
            return false;
        }
        if (abs(entry.value - values[entry.index - 1]) > scylla_blas::epsilon) {
            return false;
        }
        last = entry.index - 1;
    }
    return true;
}


BOOST_FIXTURE_TEST_CASE(vector_scale, vector_fixture)
{
    // Given vector of 3 floats
    std::vector<float> vals = {1.6f, 2.9999f, 3.0f};
    auto vector = getScyllaVectorOf(vals);

    // When performing scaling by 2
    scheduler->sscal(2, *vector);

    // Then result vector is scaled by 2.
    std::vector<float> valsx2 = {1.6f * 2, 2.9999f * 2, 3.0f * 2};
    BOOST_CHECK(cmp_vector(*vector, valsx2));
}

}
