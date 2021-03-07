cmake_minimum_required(VERSION 3.15)
project(linear_algebra)
enable_testing()

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(lib/driver EXCLUDE_FROM_ALL)
add_subdirectory(lib/fmt EXCLUDE_FROM_ALL)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/scylla_blas)
set(SRC
        ${SRC_DIR}/structure/item_set.cc
        ${SRC_DIR}/structure/vector.cc

        ${SRC_DIR}/utils/int_math.cc )

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/scylla_blas)
set(INCLUDE
        ${INCLUDE_DIR}/matrix.hh
        ${INCLUDE_DIR}/scylla_blas.hh

        ${INCLUDE_DIR}/utils/value_factory.hh
        ${INCLUDE_DIR}/utils/matrix_value_generator.hh
        ${INCLUDE_DIR}/utils/sparse_matrix_value_generator.hh

        ${INCLUDE_DIR}/structure/vector.hh
        ${INCLUDE_DIR}/structure/item_set.hh
        ${INCLUDE_DIR}/structure/matrix_block.hh
        ${INCLUDE_DIR}/structure/matrix_value.hh

        ${INCLUDE_DIR}/utils/int_math.hh
        ${INCLUDE_DIR}/utils/scylla_types.hh
        ${INCLUDE_DIR}/utils/utils.hh)

set(LIB
        scylla_modern_cpp_driver
        fmt::fmt)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib)

# Use once after starting a new Scylla cluster to initialize all necessary namespaces
add_executable(init src/init.cc "${SRC}")
target_link_libraries(init "${LIB}")

# Test data structures: item set, vectors, matrices
add_executable(struct tests/structure_test.cc "${SRC}")
target_link_libraries(struct "${LIB}")
add_test(struct struct)

# Test matrix multiplication
add_executable(multiply tests/multiply.cc "${SRC}")
target_link_libraries(multiply "${LIB}")
add_test(multiply multiply)
