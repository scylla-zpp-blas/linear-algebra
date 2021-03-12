add_executable(arnoldi
        main.cc
        arnoldi.cc
        arnoldi.hh
        generators/matrix_value_generator.hh
        generators/preset_value_factory.hh
        generators/random_value_factory.hh
        generators/sparse_matrix_value_generator.hh
        generators/value_factory.hh
        )
target_link_libraries(arnoldi scylla_blas ${Boost_THREAD_LIBRARY})