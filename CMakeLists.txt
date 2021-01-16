cmake_minimum_required(VERSION 3.15)
project(linear_algebra)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(lib/driver EXCLUDE_FROM_ALL)
add_subdirectory(lib/fmt EXCLUDE_FROM_ALL)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib)

add_executable(init init.cc
        structure/vector.cc structure/vector.hh
        scylla_matrix.cc scylla_matrix.hh
        scylla_blas.cc scylla_blas.hh scylla_types.hh)
target_link_libraries(init scylla_modern_cpp_driver fmt::fmt)

add_executable(linear_algebra
        main.cpp
        value_factory.hh
        matrix_value.hh
        matrix_value_generator.hh
        sparse_matrix_value_generator.hh
        utils/int_math.cc utils/int_math.hh
        scylla_blas.cc
        scylla_blas.hh
        matrix_multiplicator.hh
        scylla_matrix.cc scylla_matrix.hh)
target_link_libraries(linear_algebra scylla_modern_cpp_driver fmt::fmt)

add_executable(matrix matrix_test.cc
        structure/vector.cc structure/vector.hh
        scylla_matrix.cc scylla_matrix.hh
        structure/item_set.cc structure/item_set.hh)
target_link_libraries(matrix scylla_modern_cpp_driver fmt::fmt)

add_executable(multiply multiply.cc utils/int_math.cc)
target_link_libraries(multiply scylla_modern_cpp_driver fmt::fmt)