#include <boost/test/unit_test.hpp>

#include "../test_utils.hh"
#include "../fixture.hh"

namespace {

template<class T>
void trim_to_banded(auto matrix, scylla_blas::index_type KL, scylla_blas::index_type KU) {
    using namespace scylla_blas;

    for (index_type i = 1; i <= matrix->row_count; i++) {
        scylla_blas::vector_segment<T> old_row = matrix->get_row(i);
        scylla_blas::vector_segment<T> new_row;
        for (auto &val : old_row) {
            auto new_val = val;

            /* Skip values that are too far (we have to insert 0 due to our block clearing limitations) */
            if (val.index > i + KU) new_val.value = 0;
            if (val.index < i - KL) new_val.value = 0;

            new_row.emplace_back(new_val);
        }
        matrix->update_row(i, new_row);
    }
}

}

BOOST_FIXTURE_TEST_CASE(mixed_operations, mixed_fixture)
{
    using namespace scylla_blas;
    {
        /* FLOAT */

        print_matrix(*float_BxA);
        print_vector(*float_A);
        std::cout << "sgemv(float_BxA, float_A) = ";
        float_B->clear_all();
        print_vector(scheduler->sgemv(scylla_blas::NoTrans, 1, *float_BxA, *float_A, 0, *float_B));

        float_B->clear_all();
        print_matrix(*float_AxB);
        std::cout << "sgemv(float_AxB^T, float_A) = ";
        print_vector(scheduler->sgemv(scylla_blas::Trans, 1, *float_AxB, *float_A, 0, *float_B));

        float_B->clear_all();
        index_type KL = 1, KU = 2;
        trim_to_banded<float>(float_BxA, KL, KU);
        print_matrix(*float_BxA);
        std::cout << "banded sgbmv(float_BxA, float_A) = ";
        print_vector(scheduler->sgbmv(scylla_blas::NoTrans, KL, KU, 1, *float_BxA, *float_A, 0, *float_B));
        
        float_B->clear_all();
        trim_to_banded<float>(float_AxB, KL, KU);
        print_matrix(*float_AxB);
        std::cout << "banded sgbmv(float_AxB^T, float_A) = ";
        print_vector(scheduler->sgbmv(scylla_blas::Trans, KL, KU, 1, *float_AxB, *float_A, 0, *float_B));
    }

    {
        /* DOUBLE */

        print_matrix(*double_BxA);
        print_vector(*double_A);
        std::cout << "dgemv(double_BxA, double_A) = ";
        double_B->clear_all();
        print_vector(scheduler->dgemv(scylla_blas::NoTrans, 1, *double_BxA, *double_A, 0, *double_B));

        double_B->clear_all();
        print_matrix(*double_AxB);
        std::cout << "dgemv(double_AxB^T, double_A) = ";
        print_vector(scheduler->dgemv(scylla_blas::Trans, 1, *double_AxB, *double_A, 0, *double_B));

        index_type KL = 1, KU = 2;

        double_B->clear_all();
        trim_to_banded<double>(double_BxA, KL, KU);
        print_matrix(*double_BxA);
        std::cout << "banded sgbmv(double_BxA, double_A) = ";
        print_vector(scheduler->dgbmv(scylla_blas::NoTrans, KL, KU, 1, *double_BxA, *double_A, 0, *double_B));

        double_B->clear_all();
        trim_to_banded<double>(double_AxB, KL, KU);
        print_matrix(*double_AxB);
        std::cout << "banded sgbmv(double_AxB^T, double_A) = ";
        print_vector(scheduler->dgbmv(scylla_blas::Trans, KL, KU, 1, *double_AxB, *double_A, 0, *double_B));
    }
    
    print_vector(*float_A);
    print_vector(*float_B);
    float_AxB->clear_all();
    std::cout  << "float_A^T * float_B = ";
    print_matrix(scheduler->sger(1, *float_A, *float_B, *float_AxB));
    
    print_vector(*double_A);
    print_vector(*double_B);
    double_AxB->clear_all();
    std::cout  << "double_A^T * double_B = ";
    print_matrix(scheduler->dger(1, *double_A, *double_B, *double_AxB));
}

