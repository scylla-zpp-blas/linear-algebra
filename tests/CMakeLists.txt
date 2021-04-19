set(TEST_SRC
        config.hh
        fixture.hh
        main.cc

        multiply.cc
        queue.cc
        structure_test.cc
        blas_level_1/vector_copy.cc
        blas_level_1/vector_swap.cc
        blas_level_1/vector_scale.cc

        generators/sparse_matrix_value_generator.hh
        generators/value_factory.hh
)

add_executable(scylla_blas_tests "${TEST_SRC}")
target_link_libraries(scylla_blas_tests scylla_blas ${Boost_THREAD_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
target_compile_definitions(scylla_blas_tests PUBLIC -DBOOST_TEST_DYN_LINK)
add_test(scylla_blas_tests scylla_blas_tests)