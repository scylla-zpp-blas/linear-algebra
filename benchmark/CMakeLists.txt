set(BENCHMARK_SRC main.cc benchmark.hh benchmark.cc const.hh)

add_executable(scylla_blas_benchmark "${BENCHMARK_SRC}")
target_include_directories(scylla_blas_benchmark PUBLIC "${Boost_INCLUDE_DIRS}")
target_link_libraries(scylla_blas_benchmark PUBLIC scylla_blas_static "${Boost_LIBRARIES}")
target_include_directories(scylla_blas_benchmark PRIVATE "${CMAKE_SOURCE_DIR}/matrix_generators")

add_executable(insert_benchmark insert_benchmark.cc)
target_link_libraries(insert_benchmark PUBLIC scylla_blas_static)