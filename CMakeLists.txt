cmake_minimum_required(VERSION 3.15)
project(linear_algebra)

# Dependencies
add_subdirectory(lib/driver EXCLUDE_FROM_ALL)
# Required to correctly link with fmt
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
add_subdirectory(lib/fmt EXCLUDE_FROM_ALL)
find_package(Boost 1.60.0 COMPONENTS system filesystem unit_test_framework program_options REQUIRED )

enable_testing()
# Enable generation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(BUILD_EXAMPLES "Should examples be built" ON)
option(BUILD_BENCHMARK "Should benchmarks be built" ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    option(BUILD_TESTS "Should the tests be built" ON)
    set(SCYLLA_BLAS_LOGLEVEL "DEBUG" CACHE STRING "Scylla blas loglevel. Possible values: TRACE DEBUG INFO WARN ERROR CRITICAL")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
else() # Release mode
    option(BUILD_TESTS "Should the tests be built" OFF)
    set(SCYLLA_BLAS_LOGLEVEL "INFO" CACHE STRING "Scylla blas loglevel. Possible values: TRACE DEBUG INFO WARN ERROR CRITICAL")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

message("Scylla blas: build type ${CMAKE_BUILD_TYPE}")
message("Scylla blas configuration:")
message("BUILD_EXAMPLES ${BUILD_EXAMPLES}")
message("BUILD_BENCHMARK ${BUILD_BENCHMARK}")
message("BUILD_TESTS ${BUILD_TESTS}")
message("SCYLLA_BLAS_LOGLEVEL ${SCYLLA_BLAS_LOGLEVEL}")
message("CMAKE_INTERPROCEDURAL_OPTIMIZATION ${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")
message("CMAKE_EXPORT_COMPILE_COMMANDS ${CMAKE_EXPORT_COMPILE_COMMANDS}")
message("CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE}")

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/scylla_blas)
set(BLAS_SRC
        ${SRC_DIR}/blas_generic.cc
        ${SRC_DIR}/blas_level_1.cc
        ${SRC_DIR}/blas_level_2.cc
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
        ${INCLUDE_DIR}/vector.hh

        ${INCLUDE_DIR}/queue/proto.hh
        ${INCLUDE_DIR}/queue/scylla_queue.hh
        ${INCLUDE_DIR}/queue/worker_proc.hh

        ${INCLUDE_DIR}/structure/vector_segment.hh
        ${INCLUDE_DIR}/structure/vector_value.hh
        ${INCLUDE_DIR}/structure/matrix_block.hh
        ${INCLUDE_DIR}/structure/matrix_value.hh

        ${INCLUDE_DIR}/logging/logging.hh
        ${INCLUDE_DIR}/utils/scylla_types.hh
        ${INCLUDE_DIR}/utils/utils.hh)

add_library(scylla_blas SHARED "${BLAS_SRC}" "${BLAS_INCLUDE}")
target_include_directories(scylla_blas PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(scylla_blas PUBLIC scylla_modern_cpp_driver fmt::fmt)
target_compile_features(scylla_blas PUBLIC cxx_std_20)
target_compile_definitions(scylla_blas PUBLIC SCYLLA_BLAS_LOGLEVEL=${SCYLLA_BLAS_LOGLEVEL})

set(EXEC_SRC
        src/main.cc)

add_executable(scylla_blas_worker "${EXEC_SRC}")
target_include_directories(scylla_blas_worker PUBLIC "${Boost_INCLUDE_DIRS}")
target_link_libraries(scylla_blas_worker PUBLIC scylla_blas "${Boost_LIBRARIES}")

# We are building as standalone project - build tests
if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif ()
