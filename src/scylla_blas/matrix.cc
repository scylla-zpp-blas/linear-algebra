#include "scylla_blas/matrix.hh"

void scylla_blas::basic_matrix::update_meta() {
    scmd::query_result result = _session->execute(_get_meta_prepared, id);

    if (!result.next_row()) {
        throw std::runtime_error(fmt::format("Meta info for matrix {} not found in the database.", id));
    }
    row_count = result.get_column<index_t>("row_count");
    column_count = result.get_column<index_t>("column_count");
    block_size = result.get_column<index_t>("block_size");
}

void scylla_blas::basic_matrix::clear(const std::shared_ptr<scmd::session> &session, int64_t id) {
    scmd::statement truncate(fmt::format("TRUNCATE blas.matrix_{};", id));
    session->execute(truncate.set_timeout(0));
}

void scylla_blas::basic_matrix::resize(const std::shared_ptr<scmd::session> &session,
                                       int64_t id, int64_t new_row_count, int64_t new_column_count) {
    session->execute(R"(
            UPDATE blas.matrix_meta
                SET     row_count      = ?,
                        column_count   = ?
                WHERE   id             = ?;
        )", new_row_count, new_column_count, id);
}

void scylla_blas::basic_matrix::set_block_size(const std::shared_ptr<scmd::session> &session, int64_t id, int64_t new_block_size) {
    session->execute(R"(
            UPDATE blas.matrix_meta
                SET     block_size     = ?
                WHERE   id             = ?;
        )", new_block_size, id);
}

void scylla_blas::basic_matrix::drop(const std::shared_ptr<scmd::session> &session, int64_t id) {
    session->execute(fmt::format(R"(DROP TABLE blas.matrix_{})", id));
    session->execute(R"(DELETE FROM blas.matrix_meta WHERE id = ?)", id);
}

void scylla_blas::basic_matrix::init_meta(const std::shared_ptr<scmd::session> &session) {
    scmd::statement init_meta(R"(CREATE TABLE IF NOT EXISTS blas.matrix_meta (
                                                id           BIGINT PRIMARY KEY,
                                                row_count    BIGINT,
                                                column_count BIGINT,
                                                block_size   BIGINT);)");
    session->execute(init_meta.set_timeout(0));
}

scylla_blas::basic_matrix::basic_matrix(const std::shared_ptr<scmd::session> &session, int64_t id) :
        _session(session),
        id(id),
        row_count(0), column_count(0), block_size(0), // Updated in constructor body in update_meta
#define PREPARE(x, args...) x(_session->prepare(fmt::format(args)))
        PREPARE(_get_meta_prepared,
                "SELECT row_count, column_count, block_size FROM blas.matrix_meta WHERE id = ?;"),
        PREPARE(_get_value_prepared,
                "SELECT id_x, id_y, value FROM blas.matrix_{} WHERE block_x = ? AND block_y = ? AND id_x = ? AND id_y = ?;", id),
        PREPARE(_get_row_prepared,
                "SELECT id_x, id_y, value FROM blas.matrix_{} WHERE block_x = ? AND id_x = ? ALLOW FILTERING;", id),
        PREPARE(_get_block_prepared,
                "SELECT id_x, id_y, value FROM blas.matrix_{} WHERE block_x = ? AND block_y = ?;", id),
        PREPARE(_insert_value_prepared,
                "INSERT INTO blas.matrix_{} (block_x, block_y, id_x, id_y, value) VALUES (?, ?, ?, ?, ?);", id),
        PREPARE(_clear_all_prepared,
                "TRUNCATE blas.matrix_{};", id),
        PREPARE(_clear_block_row_prepared,
                "DELETE FROM blas.matrix_{} WHERE block_x = ? AND block_y = ? AND id_x = ?;", id),
        PREPARE(_resize_prepared,
                "UPDATE blas.matrix_meta SET row_count = ?, column_count = ? WHERE id = ?;"),
        PREPARE(_set_block_size_prepared,
                "UPDATE blas.matrix_meta SET block_size = ? WHERE id = ?;")
#undef PREPARE
{
    update_meta();
}

void scylla_blas::basic_matrix::clear_all() {
    _session->execute(_clear_all_prepared.get_statement().set_timeout(0));
}

void scylla_blas::basic_matrix::clear_row(index_t row) {
    std::vector<scmd::future> scheduled;
    scylla_blas::index_t blocks_count = get_blocks_width();
    for (scylla_blas::index_t block_idx = 1; block_idx <= blocks_count; block_idx++) {
        auto stmt = _clear_block_row_prepared.get_statement();
        stmt.set_timeout(0);
        scheduled.push_back(_session->execute_async(stmt, get_block_row(row), block_idx, row));
    }
    for (auto &future : scheduled) {
        future.wait();
    }
}

void scylla_blas::basic_matrix::resize(int64_t new_row_count, int64_t new_column_count) {
    _session->execute(_resize_prepared, new_row_count, new_column_count, id);
    this->row_count = new_row_count;
    this->column_count = new_column_count;
}

void scylla_blas::basic_matrix::set_block_size(int64_t new_block_size) {
    _session->execute(_set_block_size_prepared, new_block_size, id);
    this->block_size = new_block_size;
}
