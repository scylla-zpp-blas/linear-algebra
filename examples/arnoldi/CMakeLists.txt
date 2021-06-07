add_executable(arnoldi
        main.cc
        arnoldi.cc
        arnoldi.hh
        )
target_link_libraries(arnoldi scylla_blas_static "${Boost_LIBRARIES}")
target_include_directories(arnoldi PRIVATE "${Boost_INCLUDE_DIRS}" "${CMAKE_SOURCE_DIR}/matrix_generators")