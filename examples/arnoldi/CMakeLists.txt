add_executable(arnoldi
        main.cc
        arnoldi.cc
        arnoldi.hh
        )
target_link_libraries(arnoldi scylla_blas ${Boost_THREAD_LIBRARY})
target_include_directories(arnoldi PRIVATE "${CMAKE_SOURCE_DIR}/matrix_generators")