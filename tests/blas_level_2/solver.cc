#include <boost/test/unit_test.hpp>

#include "../test_utils.hh"
#include "../fixture.hh"
#include "preset_matrix_value_generator.hh"

namespace {

template<class T>
void trim_to_triangular(auto matrix, scylla_blas::index_t K = -1) {
    using namespace scylla_blas;

    for (index_t i = 1; i <= matrix->get_row_count(); i++) {
        scylla_blas::vector_segment<T> old_row = matrix->get_row(i);
        scylla_blas::vector_segment<T> new_row;
        for (auto &val : old_row) {
            auto new_val = val;

            /* Skip unnecessary values */
            if (K != -1 && val.index > i + K) new_val.value = 0;
            if (val.index < i) new_val.value = 0;

            if (val.index == i) new_val.value = 0; /* We will insert those later */
            new_row.emplace_back(new_val);
        }
        matrix->update_row(i, new_row);
    }

    for (index_t i = 1; i <= matrix->get_row_count(); i++) {
        matrix->insert_value(i, i, 10);
    }
}
}

BOOST_FIXTURE_TEST_CASE(triangular_solver, mixed_fixture)
{
    using namespace scylla_blas;

    {
        auto float_B = getScyllaVector(test_const::float_vector_3_id);
        auto float_B2 = getScyllaVector(test_const::float_vector_4_id);
        auto old_vec = float_B->get_whole();

        trim_to_triangular<float>(float_BxB);
        scheduler->strsv(Upper, NoTrans, NonUnit, *float_BxB, *float_B);
        scheduler->sgemv(NoTrans, 1,  *float_BxB, *float_B, 0, *float_B2);

        auto new_vec = float_B2->get_whole();
        LogDebug("value {0}", (old_vec + new_vec * (-1)).nrminf() / new_vec.nrminf());
        BOOST_REQUIRE_LE((old_vec + new_vec * (-1)).nrminf() / new_vec.nrminf(), EPSILON);
    }

    {
        auto double_B = getScyllaDoubleVector(test_const::double_vector_3_id);
        auto double_B2 = getScyllaDoubleVector(test_const::double_vector_4_id);
        auto old_vec = double_B->get_whole();

        trim_to_triangular<double>(double_BxB);
        scheduler->dtrsv(Upper, NoTrans, NonUnit, *double_BxB, *double_B);
        scheduler->dgemv(NoTrans, 1,  *double_BxB, *double_B, 0, *double_B2);

        auto new_vec = double_B2->get_whole();
        BOOST_REQUIRE_LE((old_vec + new_vec * (-1)).nrminf() / new_vec.nrminf(), EPSILON);
    }
}


BOOST_FIXTURE_TEST_CASE(triangular_solver_preset, mixed_fixture)
{
    using namespace scylla_blas;

    {
        auto B = test_const::matrix_B;
        std::vector<float> values;
        for (scylla_blas::index_t i = 1; i <= B; i++) {
            values.emplace_back(1.0f);
        }
        auto float_B = getScyllaVectorOf(test_const::float_vector_3_id, values);
        auto float_B2 = getScyllaVector(test_const::float_vector_4_id);
        auto old_vec = float_B->get_whole();

        // generate diagonal
        float_BxB->clear_all();
        std::vector<scylla_blas::matrix_value<float>> diag;
        for (scylla_blas::index_t i = 1; i <= B; i++) {
            diag.emplace_back(i, i, 1.0f);
        }
        preset_matrix_value_generator<float> gen(B, B, diag);
        load_matrix_from_generator(session, gen, *float_BxB);

//        trim_to_triangular<float>(float_BxB);

        print_matrix_octave(*float_BxB);
        scheduler->strsv(Upper, NoTrans, NonUnit, *float_BxB, *float_B);
        scheduler->sgemv(NoTrans, 1,  *float_BxB, *float_B, 0, *float_B2);

        auto new_vec = float_B2->get_whole();
        LogDebug("value {0}", (old_vec + new_vec * (-1)).nrminf() / new_vec.nrminf());
        BOOST_REQUIRE_LE((old_vec + new_vec * (-1)).nrminf() / new_vec.nrminf(), EPSILON);
    }
}

BOOST_FIXTURE_TEST_CASE(triangular_banded_solver, mixed_fixture)
{
    using namespace scylla_blas;
    index_t K = 2;

    {
        auto float_B = getScyllaVector(test_const::float_vector_3_id);
        auto float_B2 = getScyllaVector(test_const::float_vector_4_id);
        auto old_vec = float_B->get_whole();

        trim_to_triangular<float>(float_BxB, K);
        scheduler->stbsv(Upper, NoTrans, NonUnit, K, *float_BxB, *float_B);
        scheduler->sgbmv(NoTrans, 0, K, 1, *float_BxB, *float_B, 0, *float_B2);

        auto new_vec = float_B2->get_whole();
        BOOST_REQUIRE_LE((old_vec + new_vec * (-1)).nrminf() / new_vec.nrminf(), EPSILON);
    }

    {
        auto double_B = getScyllaDoubleVector(test_const::double_vector_3_id);
        auto double_B2 = getScyllaDoubleVector(test_const::double_vector_4_id);
        auto old_vec = double_B->get_whole();

        trim_to_triangular<double>(double_BxB, K);
        scheduler->dtbsv(Upper, NoTrans, NonUnit, K, *double_BxB, *double_B);
        scheduler->dgbmv(NoTrans, 0, K, 1, *double_BxB, *double_B, 0, *double_B2);

        auto new_vec = double_B2->get_whole();
        BOOST_REQUIRE_LE((old_vec + new_vec * (-1)).nrminf() / new_vec.nrminf(), EPSILON);
    }
}

