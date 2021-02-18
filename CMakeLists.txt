cmake_minimum_required(VERSION 3.15)
project(linear_algebra)

set(CMAKE_CXX_STANDARD 20)

set(SRC
        value_factory.hh
        matrix_value_generator.hh
        sparse_matrix_value_generator.hh

        matrix_value.hh
        scylla_matrix.cc scylla_matrix.hh

        scylla_blas.cc scylla_blas.hh

        structure/vector.cc structure/vector.hh
        structure/item_set.cc structure/item_set.hh
        structure/matrix_block.hh

        utils/int_math.cc utils/int_math.hh
        utils/scylla_types.hh
        utils/utils.hh)

set(LIB
        scylla_modern_cpp_driver
        fmt::fmt)

add_subdirectory(lib/driver EXCLUDE_FROM_ALL)
add_subdirectory(lib/fmt EXCLUDE_FROM_ALL)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib)

# Use once after starting a new Scylla cluster to initialize all necessary namespaces
add_executable(init init.cc "${SRC}")
target_link_libraries(init "${LIB}")

# Test data structures: item set, vectors, matrices
add_executable(struct structure_test.cc "${SRC}")
target_link_libraries(struct "${LIB}")

# Test matrix multiplication
add_executable(multiply multiply.cc "${SRC}")
target_link_libraries(multiply "${LIB}")

# Probably obsolete?
add_executable(linear_algebra main.cc "${SRC}")
target_link_libraries(linear_algebra "${LIB}")