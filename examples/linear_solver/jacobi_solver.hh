#pragma once

#include <exception>
#include <iostream>
#include <memory>

#include "scylla_blas/utils/scylla_types.hh"
#include "scylla_blas/structure/matrix_block.hh"
#include <scylla_blas/routines.hh>
#include <scylla_blas/matrix.hh>
#include <scylla_blas/vector.hh>

class jacobi_solver {
private:
    static constexpr double CONVERGENCE_THRESHOLD = 1e-6;
    static constexpr size_t NUM_OF_ITERATIONS = 1000;

    std::shared_ptr<scmd::session> _session;
    std::shared_ptr<scylla_blas::routine_scheduler> _scheduler;
    std::shared_ptr<scylla_blas::matrix<double>> _mat_A;
    std::shared_ptr<scylla_blas::matrix<double>> _mat_D_inverted;
    std::shared_ptr<scylla_blas::matrix<double>> _mat_L_plus_U;
    std::shared_ptr<scylla_blas::vector<double>> _aux_vector;
    scylla_blas::index_t _dimensions;

    void init_auxiliaries(scylla_blas::index_t initial_id);
    void build_matrices();

    bool check_convergence(scylla_blas::vector<double> &x, scylla_blas::vector<double> &b, double threshold);
    void jacobi_iteration(scylla_blas::vector<double> &x, scylla_blas::vector<double> &b);

public:
    /* Initializes solver used for solving systems of linear equations with matrix A.
     * When initialized the solver constructs matrices L+U and D^-1 necessary for the algorithm.
     */
    jacobi_solver(const std::shared_ptr<scmd::session> &session, scylla_blas::matrix<double> &A, scylla_blas::index_t initial_id);

    /* Solves the equation Ax = b using Jacobi method.
     * @param x - initial guess, result vector
     * @param b - RHS of the equation
     * @param num_of_iterations - maximal number of iterations before terminating
     * @param threshold - convergence threshold
     */
    void solve(scylla_blas::vector<double> &x, scylla_blas::vector<double> &b,
               size_t num_of_iterations = NUM_OF_ITERATIONS, double threshold = CONVERGENCE_THRESHOLD);
};
