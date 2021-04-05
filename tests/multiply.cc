#include <boost/test/unit_test.hpp>

#include <scmd.hh>

#include "generators/value_factory.hh"
#include "generators/sparse_matrix_value_generator.hh"
#include "scylla_blas/scylla_blas.hh"
#include "fixture.hh"

#if false
template <class T>
scylla_blas::matrix<T> naive_multiply(const std::shared_ptr<scmd::session> &session,
                         const scylla_blas::matrix<T> &A, const scylla_blas::matrix<T> &B,
                         scylla_blas::matrix<T> &AB) {
    for (scylla_blas::index_type i = 1; i <= A.row_count; i++) {
        std::vector<T> results(B.column_count + 1, 0);

        scylla_blas::vector_segment<T> left_vec = A.get_row(i);
        for (auto left_entry : left_vec) {
            scylla_blas::vector_segment<T> right_vec = B.get_row(left_entry.index);
            right_vec *= left_entry.value;
            for (auto right_entry : right_vec) {
                results[right_entry.index] += right_entry.value;
            }
        }

        AB.insert_row(i, scylla_blas::vector_segment<T>(results));
    }

    return AB;
};

template<class T>
void test_multiply_with_routine(const std::shared_ptr<scmd::session> &session,
                                int64_t A_id, scylla_blas::index_type width_A, scylla_blas::index_type height_A,
                                int64_t B_id, scylla_blas::index_type width_B, scylla_blas::index_type height_B,
                                auto fun) {
    BOOST_REQUIRE_EQUAL(C_1.row_count, C_2.row_count);
    BOOST_REQUIRE_EQUAL(C_1.column_count, C_2.column_count);

    BOOST_REQUIRE_EQUAL(C_2.row_count, A.row_count);
    BOOST_REQUIRE_EQUAL(C_2.column_count, B.column_count);

    /* Map to tuples – we don't want to implement operators for matrix/vector values */
    std::vector<scylla_blas::matrix_value<T>> naive_tuples;
    for (scylla_blas::index_type i = 1; i <= C_1.row_count; i++) {
        auto naive_vals = C_1.get_row(i);
        for (auto &val : naive_vals)
            naive_tuples.emplace_back(i, val.index, val.value);
    }

    std::vector<scylla_blas::matrix_value<T>> blas_tuples;
    for (scylla_blas::index_type i = 1; i <= C_2.row_count; i++) {
        auto blas_vals = C_2.get_row(i);
        for (auto &val : blas_vals)
            blas_tuples.emplace_back(i, val.index, val.value);
    }

    /* TODO: Use boost macros? */
    auto it_1 = naive_tuples.begin();
    auto it_2 = blas_tuples.begin();
    for (int i = 0; it_1 != naive_tuples.end() && it_2 != blas_tuples.end(); i++, it_1++, it_2++) {
        if (*it_1 != *it_2) {
            std::cerr << "Value mismatch at position " << i << "!" << std::endl;
            std::cerr << "(" << it_1->row_index << ", " << it_1->col_index << ") -> " << it_1->value << " != " <<
                         "(" << it_2->row_index << ", " << it_2->col_index << ") -> " << it_2->value << std::endl;
            std::cerr << "Aborting...";
            BOOST_CHECK(false);
        }
    }
}
#endif

BOOST_FIXTURE_TEST_SUITE(multiply_tests, matrix_fixture)

BOOST_AUTO_TEST_CASE(float_mm)
{
    using namespace scylla_blas;

    print_matrix(*float_6x5);
    print_matrix(*float_5x6);

    std::cerr << "Multiply" << std::endl;
    matrix<float>::clear(session, float_6x6->id);
    routine_scheduler(session).sgemm( NoTrans, NoTrans, 1, *float_6x5, *float_5x6, 1, *float_6x6);
    print_matrix(*float_6x6);

    std::cerr << "Multiply with coeff (2.5)" << std::endl;
    matrix<float>::clear(session, float_6x6->id);
    routine_scheduler(session).sgemm( NoTrans, NoTrans, 2.5, *float_6x5, *float_5x6, 1, *float_6x6);
    print_matrix(*float_6x6);

    std::cerr << "Multiply and add 3 times result" << std::endl;
    matrix<float>::clear(session, float_6x6->id);
    routine_scheduler(session).sgemm( NoTrans, NoTrans, 1, *float_6x5, *float_5x6, 1, *float_6x6);
    routine_scheduler(session).sgemm( NoTrans, NoTrans, 1, *float_6x5, *float_5x6, 3, *float_6x6);
    print_matrix(*float_6x6);

    std::cerr << "Multiply transA" << std::endl;
    matrix<float>::clear(session, float_6x6->id);
    routine_scheduler(session).sgemm( Trans, NoTrans, 1, *float_5x6, *float_5x6, 1, *float_6x6);
    print_matrix(*float_6x6);

    std::cerr << "Multiply transB" << std::endl;
    matrix<float>::clear(session, float_6x6->id);
    routine_scheduler(session).sgemm( NoTrans, Trans, 1, *float_6x5, *float_6x5, 1, *float_6x6);
    print_matrix(*float_6x6);

    std::cerr << "Multiply transAB" << std::endl;
    matrix<float>::clear(session, float_6x6->id);
    routine_scheduler(session).sgemm( Trans, Trans, 1, *float_5x6, *float_6x5, 1, *float_6x6);
    print_matrix(*float_6x6);
}

BOOST_AUTO_TEST_SUITE_END();