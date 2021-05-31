#include <iostream>

#include "jacobi_solver.hh"
#include "scylla_blas/utils/scylla_types.hh"
#include "scylla_blas/structure/matrix_block.hh"
#include <scylla_blas/routines.hh>
#include <scylla_blas/matrix.hh>

int main(int argc, char **argv) {
    if (argc <= 2) {
        throw std::runtime_error("You need to specify ip in the command line: " + std::string(argv[0]) + " scylla_ip scylla_port");
    }
    std::string scylla_ip = argv[1];
    std::string scylla_port = argv[2];

    auto session = std::make_shared<scmd::session>(scylla_ip, scylla_port);
    std::shared_ptr<scylla_blas::routine_scheduler> scheduler = std::make_shared<scylla_blas::routine_scheduler>(session);

    scylla_blas::index_type N = 100;

    /* A diagonally dominant matrix */
    scylla_blas::matrix<double> m = scylla_blas::matrix<double>::init_and_return(session, 1, N, N);
    for (scylla_blas::index_type i = 1; i <= N; i++) {
        for (scylla_blas::index_type j = 1; j <= N; j++) {
            if (i == j) {
                m.insert_value(i, j, 4);
            }
            if (i == j + 1 || i == j - 1) {
                m.insert_value(i, j, 1);
            }
        }
    }
    scylla_blas::vector<double> actual_solution = scylla_blas::vector<double>::init_and_return(session, 1, N);
    for (scylla_blas::index_type i = 1; i <= N; i++) {
        actual_solution.update_value(i, i);
    }

    scylla_blas::vector<double> b = scylla_blas::vector<double>::init_and_return(session, 2, N);
    scheduler->dgemv(scylla_blas::NoTrans, 1, m, actual_solution, 0, b);

    jacobi_solver solver = jacobi_solver(session, m, 101);

    scylla_blas::vector<double> x = scylla_blas::vector<double>::init_and_return(session, 3, N);
    solver.solve(x, b);

    for (scylla_blas::index_type i = 1; i <= N; i++) { // should print numbers from 1 to N
        std::cout << x.get_value(i) << std::endl;
    }
}