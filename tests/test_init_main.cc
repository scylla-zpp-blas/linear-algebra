#include <memory>
#include <scmd.hh>

#include "scylla_blas/matrix.hh"
#include "scylla_blas/vector.hh"

#include "const.hh"

void init_matrices(const std::shared_ptr<scmd::session> &session) {
    std::cerr << "Initializing containers for test matrices..." << std::endl;

    scylla_blas::index_type A = test_const::matrix_A;
    scylla_blas::index_type B = test_const::matrix_B;
    
    scylla_blas::matrix<float>::init(session, test_const::float_matrix_AxB_id, A, B);
    scylla_blas::matrix<float>::init(session, test_const::float_matrix_BxA_id, B, A);
    scylla_blas::matrix<float>::init(session, test_const::float_matrix_BxB_id, B, B);

    scylla_blas::matrix<double>::init(session, test_const::double_matrix_AxB_id, A, B);
    scylla_blas::matrix<double>::init(session, test_const::double_matrix_BxA_id, B, A);
    scylla_blas::matrix<double>::init(session, test_const::double_matrix_BxB_id, B, B);

    std::cerr << "Containers for test matrices initialized!" << std::endl;
}

void init_vectors(const std::shared_ptr<scmd::session> &session) {
    std::cerr << "Initializing containers for test vectors..." << std::endl;
    
    scylla_blas::index_type len_A = test_const::test_vector_len_A;
    scylla_blas::index_type len_B = test_const::test_vector_len_B;

    scylla_blas::vector<float>::init(session, test_const::float_vector_1_id, len_A);
    scylla_blas::vector<float>::init(session, test_const::float_vector_2_id, len_A);

    for (scylla_blas::index_type vector_id : test_const::float_vector_indexes) {
        scylla_blas::vector<float>::init(session, vector_id, len);
    }
    for (scylla_blas::index_type vector_id : test_const::double_vector_indexes) {
        scylla_blas::vector<float>::init(session, vector_id, len);
    }

    std::cerr << "Containers for test vectors initialized!" << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        throw std::runtime_error("You need to specify ip in the command line: " + std::string(argv[0]) + " scylla_ip [scylla_port]");
    }
    std::string scylla_ip = argv[1];
    std::string scylla_port = argc > 2 ? argv[2] : std::to_string(SCYLLA_DEFAULT_PORT);
    
    auto session = std::make_shared<scmd::session>(scylla_ip, scylla_port);
    init_matrices(session);
    init_vectors(session);
}