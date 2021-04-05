#include "scylla_blas/matrix.hh"

void scylla_blas::basic_matrix::init_meta(const std::shared_ptr<scmd::session> &session) {
    scmd::statement init_meta(R"(CREATE TABLE IF NOT EXISTS blas.matrix_meta (
                                                id           BIGINT PRIMARY KEY,
                                                row_count    BIGINT,
                                                column_count BIGINT);)");
    session->execute(init_meta.set_timeout(0));
}

std::pair<scylla_blas::index_type, scylla_blas::index_type> scylla_blas::basic_matrix::get_dimensions() const {
    scmd::query_result result = _session->execute(_get_meta_prepared);

    if (!result.next_row()) {
        throw std::runtime_error(fmt::format("Meta info for matrix {} not found in the database.", id));
    }

    return {result.get_column<index_type>("row_count"),
            result.get_column<index_type>("column_count")};
}

scylla_blas::basic_matrix::basic_matrix(const std::shared_ptr<scmd::session> &session, int64_t id) :
        _session(session),
        id(id),
        row_count(get_dimensions().first),
        column_count(get_dimensions().second),
#define PREPARE(x, args...) x(_session->prepare(fmt::format(args)))
        PREPARE(_get_meta_prepared,
                "SELECT * FROM blas.matrix_meta WHERE id = {};", id),
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

void scylla_blas::basic_matrix::clear(const std::shared_ptr<scmd::session> &session, int64_t id) {
    scmd::statement drop_table(fmt::format("TRUNCATE blas.matrix_{0};", id));
    session->execute(drop_table.set_timeout(0));
}

void scylla_blas::basic_matrix::resize(const std::shared_ptr<scmd::session> &session,
                                       int64_t id, int64_t new_row_count, int64_t new_column_count) {
    session->execute(fmt::format(R"(
            UPDATE blas.matrix_meta
                SET     row_count      = {1},
                        column_count   = {2}
                WHERE   id             = {0};
        )", id, new_row_count, new_column_count));
}