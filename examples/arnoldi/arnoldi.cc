#include "arnoldi.hh"

void arnoldi::transfer_row_to_vector(std::shared_ptr<scylla_blas::matrix<float>> mat, scylla_blas::index_type row_index,
                                     std::shared_ptr<scylla_blas::vector<float>> vec) {
    LogTrace("Transfering row to vector.");
    scylla_blas::vector_segment<float> row = mat->get_row(row_index);
    LogTrace("Clear all");
    vec->clear_all();
    LogTrace("Update");
    vec->update_values(row);
    LogTrace("Transfering row to vector DONE.");
}

void arnoldi::transfer_vector_to_row(std::shared_ptr<scylla_blas::matrix<float>> mat, scylla_blas::index_type row_index,
                                     std::shared_ptr<scylla_blas::vector<float>> vec) {
    LogTrace("Transfering vector to row.");
    scylla_blas::vector_segment<float> row = vec->get_whole();
    mat->update_row(row_index, row);
    LogTrace("Transfering vector to row DONE.");
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
    LogTrace("Initial phase");
    _scheduler.scopy(*b, *q);
    _scheduler.sscal(1.0f / _scheduler.snrm2(*q), *q);
    transfer_vector_to_row(Q, 1, q);
    LogTrace("INITIAL DONE, ENTERING LOOPS");
    for (scylla_blas::index_type k = 1; k <= n; k++) {
        LogTrace("MATRIX MULTIPLY");
        _scheduler.sgemv(scylla_blas::NoTrans, 1.0f, *A, *q, 0, *v);
        LogTrace("MATRIX MULTIPLY DONE");
        LogTrace("INNER LOOP:");
        for (scylla_blas::index_type j = 1; j <= k; j++) {
            transfer_row_to_vector(Q, j, t);
            LogTrace("\tValue insert");
            h->insert_value(j, k, _scheduler.sdot(*t, *v));
            LogTrace("\tValue insert done");
            LogTrace("\tSaxpy");
            _scheduler.saxpy(-h->get_value(j, k), *t, *v); // v = v - h[j, k] * Q[:, j]
            LogTrace("\tSaxpy DONE");
        }
        LogTrace("INNER LOOP DONE");
        LogTrace("Norm");
        h->insert_value(k + 1, k, _scheduler.snrm2(*v));
        LogTrace("Norm done");
        const float eps = 1e-12;
        if (h->get_value(k + 1, k) > eps) {
            LogTrace("IF statement");
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

