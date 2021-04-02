#include "scylla_blas/matrix.hh"

void scylla_blas::basic_matrix::init_meta(const std::shared_ptr<scmd::session> &session) {
    scmd::statement init_meta(R"(CREATE TABLE IF NOT EXISTS blas.matrix_meta (
                                                id      BIGINT PRIMARY KEY,
                                                rows    BIGINT,
                                                columns BIGINT);)");
    session->execute(init_meta.set_timeout(0));
}

std::pair<scylla_blas::index_type, scylla_blas::index_type> scylla_blas::basic_matrix::get_dimensions(
        const std::shared_ptr<scmd::session> &session, int64_t id) {
    std::cerr << "Obtaining dimensions for " << id  << "..." << std::endl;
    scmd::query_result result = session->execute(fmt::format("SELECT * FROM blas.matrix_meta WHERE id = {0};", id));

    result.next_row(); // there must be exactly one!
    std::cerr << "Obtaining dimensions!" << std::endl;
    return {result.get_column<index_type>("rows"),
            result.get_column<index_type>("columns")};
}

scylla_blas::basic_matrix::basic_matrix(const std::shared_ptr<scmd::session>& session, int64_t id) :
        _session(session),
        id(id),
        rows(get_dimensions(session, id).first),
        columns(get_dimensions(session, id).second),
#define PREPARE(x, args...) x(_session->prepare(fmt::format(args)))
        PREPARE(_get_value_prepared,
                "SELECT * FROM blas.matrix_{} WHERE block_x = ? AND block_y = ? AND id_x = ? AND id_y = ? ALLOW FILTERING;", id),
        PREPARE(_get_row_prepared,
                "SELECT * FROM blas.matrix_{} WHERE block_x = ? AND id_x = ?;", id),
        PREPARE(_get_block_prepared,
                "SELECT * FROM blas.matrix_{} WHERE block_x = ? AND block_y = ? ALLOW FILTERING;", id),
        PREPARE(_insert_value_prepared,
                "INSERT INTO blas.matrix_{} (block_x, block_y, id_x, id_y, value) VALUES (?, ?, ?, ?, ?);", id)
#undef PREPARE
{ }