#include "jacobi_solver.hh"

void jacobi_solver::init_auxiliaries(scylla_blas::index_t initial_id) {
    scylla_blas::vector<double>::init(_session, initial_id, _dimensions);
    _aux_vector = std::make_shared<scylla_blas::vector<double>>(_session, initial_id++);

    scylla_blas::matrix<double>::init(_session, initial_id, _dimensions, _dimensions);
    _mat_D_inverted = std::make_shared<scylla_blas::matrix<double>>(_session, initial_id++);

    scylla_blas::matrix<double>::init(_session, initial_id, _dimensions, _dimensions);
    _mat_L_plus_U = std::make_shared<scylla_blas::matrix<double>>(_session, initial_id++);
}

void jacobi_solver::build_matrices() {
    scylla_blas::index_t blocks_dimensions = _mat_A->get_blocks_height();

    for (scylla_blas::index_t i = 1; i <= blocks_dimensions; i++) {
        for (scylla_blas::index_t j = 1; j <= blocks_dimensions; j++) {
            scylla_blas::matrix_block<double> block = _mat_A->get_block(i, j);
            if (i == j) {
                std::vector<scylla_blas::matrix_value<double>> values = block.get_values_raw();
                std::vector<scylla_blas::matrix_value<double>> l_plus_u_values;
                std::vector<scylla_blas::matrix_value<double>> d_inverted_values;
                for (auto &val : values) {
                    if (val.row_index == val.col_index) {
                        val.value = 1 / val.value;
                        d_inverted_values.push_back(val);
                    } else {
                        l_plus_u_values.push_back(val);
                    }
                }
                scylla_blas::matrix_block<double> l_plus_u_block(l_plus_u_values);
                scylla_blas::matrix_block<double> d_inverted_block(d_inverted_values);
                _mat_L_plus_U->insert_block(i, j, l_plus_u_block);
                _mat_D_inverted->insert_block(i, j, d_inverted_block);
            } else {
                _mat_L_plus_U->insert_block(i, j, block);
            }
        }
    }
}

/* We use a simple convergence rule and check whether ||b - Ax||_inf < threshold.
 * A more sophisticated stopping rule might be better for practical applications */
bool jacobi_solver::check_convergence(scylla_blas::vector<double> &x, scylla_blas::vector<double> &b, double threshold) {
    _scheduler->dcopy(b, *_aux_vector);
    _scheduler->dgemv(scylla_blas::NoTrans, -1, *_mat_A, x, 1, *_aux_vector);
    scylla_blas::index_t id_max = _scheduler->idamax(*_aux_vector);
    double val = _aux_vector->get_value(id_max);
    std::cerr << "Error of current solution: " << val << std::endl;
    return std::abs(val) < threshold;
}

void jacobi_solver::jacobi_iteration(scylla_blas::vector<double> &x, scylla_blas::vector<double> &b) {
    _scheduler->dcopy(b, *_aux_vector);
    _scheduler->dgemv(scylla_blas::NoTrans, -1, *_mat_L_plus_U, x, 1, *_aux_vector);
    _scheduler->dgemv(scylla_blas::NoTrans, 1, *_mat_D_inverted, *_aux_vector, 0, x);
}

jacobi_solver::jacobi_solver(const std::shared_ptr<scmd::session> &session, scylla_blas::matrix<double> &A, scylla_blas::index_t initial_id) :
        _session(session),
        _scheduler(nullptr),
        _mat_A(nullptr),
        _mat_D_inverted(nullptr),
        _mat_L_plus_U(nullptr),
        _aux_vector(nullptr) {
    if (A.get_column_count() != A.get_row_count()) {
        throw std::runtime_error(fmt::format("Matrix {0} is not a square matrix",
                                             A.get_id()));
    }
    _dimensions = A.get_column_count();

    _scheduler = std::make_shared<scylla_blas::routine_scheduler>(session);
    _mat_A = std::make_shared<scylla_blas::matrix<double>>(session, A.get_id());

    init_auxiliaries(initial_id);
    build_matrices();
}

void jacobi_solver::solve(scylla_blas::vector<double> &x, scylla_blas::vector<double> &b, size_t num_of_iterations,
                          double threshold) {
    if (x.get_length() != _dimensions) {
        throw std::runtime_error(fmt::format("Vector {0} of length {1} incompatible with Jacobi solver",
                           x.get_id(), x.get_length()));
    }

    if (b.get_length() != _dimensions) {
        throw std::runtime_error(fmt::format("Vector {0} of length {1} incompatible with Jacobi solver",
                                             b.get_id(), b.get_length()));
    }

    for (size_t i = 1; i <= num_of_iterations; i++) {
        std::cerr << std::endl << "Begin iteration " << i << std::endl;
        jacobi_iteration(x, b);
        if (check_convergence(x, b, threshold)) {
            return;
        }
    }

    throw std::runtime_error(fmt::format("Convergence not reached after {0} iterations",
                                         num_of_iterations));
}
