cmake_minimum_required(VERSION 3.15)
project(linear_algebra)

set(CMAKE_CXX_STANDARD 20)

add_executable(linear_algebra main.cpp value_factory.hh matrix_value.hh matrix_value_generator.hh sparse_matrix_value_generator.hh utils/int_math.cc utils/int_math.hh scylla_blas.cc scylla_blas.hh matrix_multiplicator.hh scylla_matrix.cc scylla_matrix.hh)