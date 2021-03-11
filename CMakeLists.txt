cmake_minimum_required(VERSION 3.15)
project(linear_algebra)
enable_testing()

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(lib/driver EXCLUDE_FROM_ALL)

# Required to correctly link with fmt
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
add_subdirectory(lib/fmt EXCLUDE_FROM_ALL)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/scylla_blas)
set(BLAS_SRC
        ${SRC_DIR}/queue/item_set.cc
        ${SRC_DIR}/structure/vector.cc

        ${SRC_DIR}/utils/int_math.cc
        src/main.cc)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/scylla_blas)
set(BLAS_INCLUDE
        ${INCLUDE_DIR}/matrix.hh
        ${INCLUDE_DIR}/scylla_blas.hh

        ${INCLUDE_DIR}/utils/value_factory.hh
        ${INCLUDE_DIR}/utils/matrix_value_generator.hh
        ${INCLUDE_DIR}/utils/sparse_matrix_value_generator.hh

        ${INCLUDE_DIR}/structure/vector.hh
        ${INCLUDE_DIR}/queue/item_set.hh
        ${INCLUDE_DIR}/structure/matrix_block.hh
        ${INCLUDE_DIR}/structure/matrix_value.hh

        ${INCLUDE_DIR}/utils/int_math.hh
        ${INCLUDE_DIR}/utils/scylla_types.hh
        ${INCLUDE_DIR}/utils/utils.hh)

add_library(scylla_blas SHARED "${BLAS_SRC}" "${BLAS_INCLUDE}")
target_include_directories(scylla_blas PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(scylla_blas PUBLIC scylla_modern_cpp_driver fmt::fmt)

# We are building as standalone project - build tests
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    add_subdirectory(tests)
endif ()
