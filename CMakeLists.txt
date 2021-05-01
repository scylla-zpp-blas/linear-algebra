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
        ${SRC_DIR}/blas_generic.cc
        ${SRC_DIR}/blas_level_1.cc
        ${SRC_DIR}/blas_level_3.cc
        ${SRC_DIR}/matrix.cc
        ${SRC_DIR}/vector.cc
        ${SRC_DIR}/queue/scylla_queue.cc
        ${SRC_DIR}/queue/worker_proc.cc
        )

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/scylla_blas)
set(BLAS_INCLUDE
        ${INCLUDE_DIR}/config.hh
        ${INCLUDE_DIR}/matrix.hh
        ${INCLUDE_DIR}/routines.hh
        ${INCLUDE_DIR}/scylla_blas.hh
        ${INCLUDE_DIR}/vector.hh

        ${INCLUDE_DIR}/queue/proto.hh
        ${INCLUDE_DIR}/queue/scylla_queue.hh
        ${INCLUDE_DIR}/queue/worker_proc.hh

        ${INCLUDE_DIR}/structure/vector_segment.hh
        ${INCLUDE_DIR}/structure/vector_value.hh
        ${INCLUDE_DIR}/structure/matrix_block.hh
        ${INCLUDE_DIR}/structure/matrix_value.hh

        ${INCLUDE_DIR}/utils/matrix_value_generator.hh
        ${INCLUDE_DIR}/utils/scylla_types.hh
        ${INCLUDE_DIR}/utils/utils.hh
        src/scylla_blas/blas_generic.cc)

add_library(scylla_blas SHARED "${BLAS_SRC}" "${BLAS_INCLUDE}")
target_include_directories(scylla_blas PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(scylla_blas PUBLIC scylla_modern_cpp_driver fmt::fmt)

set(EXEC_SRC
        src/main.cc)

find_package(Boost 1.60.0 COMPONENTS system thread filesystem unit_test_framework program_options REQUIRED )
add_executable(scylla_blas_worker "${EXEC_SRC}")
target_include_directories(scylla_blas_worker PUBLIC "${Boost_INCLUDE_DIRS}")
target_link_libraries(scylla_blas_worker PUBLIC scylla_blas "${Boost_LIBRARIES}")

# We are building as standalone project - build tests
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    add_subdirectory(tests)
endif ()
