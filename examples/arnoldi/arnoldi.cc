#include "arnoldi.hh"

void arnoldi::transfer_row_to_vector(std::shared_ptr<scylla_blas::matrix<float>> mat, scylla_blas::index_type row_index,
                                     std::shared_ptr<scylla_blas::vector<float>> vec) {
    std::cout << "Transfering row to vector.\n";
    scylla_blas::vector_segment<float> row = mat->get_row(row_index);
    std::cout << "Clear all\n";
    vec->clear_all();
    std::cout << "Update\n";
    vec->update_values(row);
    std::cout << "Transfering row to vector DONE.\n";
}

void arnoldi::transfer_vector_to_row(std::shared_ptr<scylla_blas::matrix<float>> mat, scylla_blas::index_type row_index,
                                     std::shared_ptr<scylla_blas::vector<float>> vec) {
    std::cout << "Transfering vector to row.\n";
    scylla_blas::vector_segment<float> row = vec->get_whole();
    mat->update_row(row_index, row);
    std::cout << "Transfering vector to row DONE.\n";
}

arnoldi::arnoldi(std::shared_ptr<scmd::session> session) : _session(session), _scheduler(session) {}

void arnoldi::compute(std::shared_ptr<scylla_blas::matrix<float>> A,
                      std::shared_ptr<scylla_blas::vector<float>> b,
                      scylla_blas::index_type n,
                      std::shared_ptr<scylla_blas::matrix<float>> h,
                      std::shared_ptr<scylla_blas::matrix<float>> Q,
                      std::shared_ptr<scylla_blas::vector<float>> v,
                      std::shared_ptr<scylla_blas::vector<float>> q,
                      std::shared_ptr<scylla_blas::vector<float>> t) {
    h->clear_all();
    Q->clear_all();
    std::cout << "Initial phase\n";
    _scheduler.scopy(*b, *q);
    _scheduler.sscal(1.0f / _scheduler.snrm2(*q), *q);
    transfer_vector_to_row(Q, 1, q);
    std::cout << "INITIAL DONE, ENTERING LOOPS\n";
    for (scylla_blas::index_type k = 1; k <= n; k++) {
        std::cout << "MATRIX MULTIPLY\n";
        _scheduler.sgemv(scylla_blas::NoTrans, 1.0f, *A, *q, 0, *v);
        std::cout << "MATRIX MULTIPLY DONE\n";
        std::cout << "INNER LOOP:\n";
        for (scylla_blas::index_type j = 1; j <= k; j++) {
            transfer_row_to_vector(Q, j, t);
            std::cout << "Value insert\n";
            h->insert_value(j, k, _scheduler.sdot(*t, *v));
            std::cout << "Value insert done\n";
            std::cout << "Saxpy\n";
            _scheduler.saxpy(-h->get_value(j, k), *t, *v); // v = v - h[j, k] * Q[:, j]
            std::cout << "Saxpy DONE\n";
        }
        std::cout << "INNER LOOP DONE\n";
        std::cout << "Norm\n";
        h->insert_value(k + 1, k, _scheduler.snrm2(*v));
        std::cout << "Norm done\n";
        const float eps = 1e-12;
        if (h->get_value(k + 1, k) > eps) {
            std::cout << "IF statement\n";
            _scheduler.scopy(*v, *q);
            _scheduler.sscal(1.0f / h->get_value(k + 1, k), *q);
            transfer_vector_to_row(Q, k + 1, q);
        }
        else {
            return; // Q, h;
        }
    }
    return; // Q, h;
}

