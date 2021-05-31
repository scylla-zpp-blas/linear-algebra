find_package(Boost 1.60.0 COMPONENTS system thread filesystem unit_test_framework program_options REQUIRED )
add_executable(example example.cc jacobi_solver.cc jacobi_solver.hh)
target_link_libraries(example PUBLIC scylla_blas "${Boost_LIBRARIES}")