#include <boost/test/unit_test.hpp>

#include <scmd.hh>

#include "generators/value_factory.hh"
#include "generators/sparse_matrix_value_generator.hh"
#include "scylla_blas/scylla_blas.hh"
#include "fixture.hh"

template <class T>
scylla_blas::matrix<T> naive_multiply(const std::shared_ptr<scmd::session> &session,
                         const scylla_blas::matrix<T> &A, const scylla_blas::matrix<T> &B) {
    int64_t base_id = scylla_blas::get_timestamp();

    int64_t A_id = base_id;
    int64_t B_id = base_id + 1;
    int64_t AB_id = base_id + 2;

    auto AB = scylla_blas::matrix<T>::init_and_return(session, AB_id, A.rows, B.columns);

    for (scylla_blas::index_type i = 1; i <= A.rows; i++) {
        /* FIXME? std::unordered_map could be more efficient for very large and very sparse matrices */
        std::vector<T> results(B.columns + 1, 0);

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
                                scylla_blas::index_type A_id, scylla_blas::index_type B_id, auto fun) {
    std::shared_ptr<scylla_blas::value_factory<T>> f =
            std::make_shared<scylla_blas::value_factory<T>>(0, 9, 1111);

    scylla_blas::sparse_matrix_value_generator<T> gen1(4, 6, 10, 42, f);
    scylla_blas::sparse_matrix_value_generator<T> gen2(6, 8, 10, 44, f);

    scylla_blas::matrix<T> A = scylla_blas::load_matrix_from_generator(session, gen1, A_id);
    scylla_blas::matrix<T> B = scylla_blas::load_matrix_from_generator(session, gen2, B_id);
    print_matrix(A);
    print_matrix(B);

    auto C_1 = naive_multiply(session, A, B);
    print_matrix(C_1);

    auto C_2 = fun(session, A, B);
    print_matrix(C_2);

    BOOST_REQUIRE_EQUAL(C_1.rows, C_2.rows);
    BOOST_REQUIRE_EQUAL(C_1.columns, C_2.columns);

    BOOST_REQUIRE_EQUAL(C_2.rows, A.rows);
    BOOST_REQUIRE_EQUAL(C_2.columns, B.columns);

    /* Map to tuples – we don't want to implement operators for matrix/vector values */
    std::vector<scylla_blas::matrix_value<T>> naive_tuples;
    for (scylla_blas::index_type i = 1; i <= C_1.rows; i++) {
        auto naive_vals = C_1.get_row(i);
        for (auto &val : naive_vals)
            naive_tuples.emplace_back(i, val.index, val.value);
    }

    std::vector<scylla_blas::matrix_value<T>> blas_tuples;
    for (scylla_blas::index_type i = 1; i <= C_2.rows; i++) {
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

BOOST_FIXTURE_TEST_SUITE(multiply_tests, scylla_fixture)

BOOST_AUTO_TEST_CASE(multiply_float_mm)
{
    using namespace scylla_blas;
    test_multiply_with_routine<float>(
            session, 1000, 1001,
            [](const auto &session, const auto &A, const auto &B) {
                return routine_factory(session).sgemm(RowMajor, NoTrans, NoTrans, 1, A, B, 1);
            }
    );
}

BOOST_AUTO_TEST_CASE(multiply_double_mm)
{
    using namespace scylla_blas;
    test_multiply_with_routine<double>(
            session, 2000, 2001,
            [](const auto &session, const auto &A, const auto &B) {
                return routine_factory(session).dgemm(RowMajor, NoTrans, NoTrans, 1, A, B, 1);
            }
    );
}

BOOST_AUTO_TEST_SUITE_END();