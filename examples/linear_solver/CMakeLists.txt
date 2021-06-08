add_executable(solver example.cc jacobi_solver.cc jacobi_solver.hh)
target_link_libraries(solver PUBLIC scylla_blas)