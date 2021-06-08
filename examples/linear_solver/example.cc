#include <iostream>

#include <scylla_blas/utils/scylla_types.hh>
#include <scylla_blas/structure/matrix_block.hh>
#include <scylla_blas/routines.hh>
#include <scylla_blas/matrix.hh>
#include <scylla_blas/config.hh>

#include "jacobi_solver.hh"

/* Dimensions of the example */
constexpr scylla_blas::index_t N = 10;

int main(int argc, char **argv) {
    if (argc <= 1) {
        throw std::runtime_error("You need to specify ip in the command line: " + std::string(argv[0]) + " scylla_ip");
    }
    std::string scylla_ip = argv[1];
    std::string scylla_port;
    if (argc >= 3) {
        scylla_port = argv[2];
    } else {
        scylla_port = std::to_string(SCYLLA_DEFAULT_PORT);
    }

    auto session = std::make_shared<scmd::session>(scylla_ip, scylla_port);
    std::shared_ptr<scylla_blas::routine_scheduler> scheduler = std::make_shared<scylla_blas::routine_scheduler>(session);

    /* An example diagonally dominant matrix */
    scylla_blas::matrix<double> m = scylla_blas::matrix<double>::init_and_return(session, 1, N, N);

    std::vector<scylla_blas::matrix_value<double>> matrix_values;
    for (scylla_blas::index_t i = 1; i <= N; i++) {
        for (scylla_blas::index_t j = 1; j <= N; j++) {
            if (i == j) {
                matrix_values.emplace_back(i, j, 4);
            }
            if (i == j + 1 || i == j - 1) {
                matrix_values.emplace_back(i, j, 1);
            }
        }
    }
    m.insert_values(matrix_values);

    scylla_blas::vector<double> actual_solution = scylla_blas::vector<double>::init_and_return(session, 1, N);

    std::vector<scylla_blas::vector_value<double>> actual_solution_values;
    for (scylla_blas::index_t i = 1; i <= N; i++) {
        actual_solution_values.emplace_back(i, i);
    }
    actual_solution.update_values(actual_solution_values);

    scylla_blas::vector<double> b = scylla_blas::vector<double>::init_and_return(session, 2, N);
    scheduler->dgemv(scylla_blas::NoTrans, 1, m, actual_solution, 0, b);

    jacobi_solver solver = jacobi_solver(session, m, 101);

    scylla_blas::vector<double> x = scylla_blas::vector<double>::init_and_return(session, 3, N);
    solver.solve(x, b);

    for (scylla_blas::index_t i = 1; i <= N; i++) { // should print numbers from 1 to N
        std::cout << x.get_value(i) << std::endl;
    }
}