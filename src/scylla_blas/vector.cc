#include "scylla_blas/vector.hh"

void scylla_blas::basic_vector::get_meta_from_database() {
    scmd::query_result result = _session->execute(_get_meta_prepared);

    if (!result.next_row()) {
        throw std::runtime_error(fmt::format("Meta info for vector {} not found in the database.", id));
    }

    this->length =  result.get_column<index_t>("length");
    this->block_size =  result.get_column<index_t>("block_size");
}

void scylla_blas::basic_vector::clear(const std::shared_ptr<scmd::session> &session, int64_t id) {
    scmd::statement drop_table(fmt::format("TRUNCATE blas.vector_{0};", id));
    session->execute(drop_table.set_timeout(0));
}

void scylla_blas::basic_vector::resize(const std::shared_ptr<scmd::session> &session,
                                       int64_t id, int64_t new_length) {
    session->execute(R"(
            UPDATE blas.vector_meta
                SET     length     = ?
                WHERE   id         = ?;
        )", new_length, id);
}

void scylla_blas::basic_vector::set_block_size(const std::shared_ptr<scmd::session> &session, scylla_blas::id_t id,
                                               scylla_blas::index_t new_block_size) {
    session->execute(R"(
            UPDATE blas.vector_meta
                SET     block_size  = ?
                WHERE   id          = ?;
        )", new_block_size, id);
}

void scylla_blas::basic_vector::drop(const std::shared_ptr<scmd::session> &session, int64_t id) {
    session->execute(fmt::format(R"(DROP TABLE blas.vector_{})", id));
    session->execute(R"(DELETE FROM blas.vector_meta WHERE id = ?)", id);
}

void scylla_blas::basic_vector::init_meta(const std::shared_ptr<scmd::session> &session) {
    scmd::statement init_meta(R"(CREATE TABLE IF NOT EXISTS blas.vector_meta (
                                                id         BIGINT PRIMARY KEY,
                                                length     BIGINT,
                                                block_size BIGINT);)");
    session->execute(init_meta.set_timeout(0));
}

scylla_blas::basic_vector::basic_vector(const std::shared_ptr<scmd::session> &session, int64_t id) :
        _session(session),
        id(id),
        length(0), block_size(0), // Updated in constructor body in update_meta
#define PREPARE(x, args...) x(_session->prepare(fmt::format(args)))
        PREPARE(_get_meta_prepared,
                "SELECT * FROM blas.vector_meta WHERE id = {};", id),
        PREPARE(_get_value_prepared,
                "SELECT * FROM blas.vector_{} WHERE segment = ? AND idx = ?;", id),
        PREPARE(_get_segment_prepared,
                "SELECT * FROM blas.vector_{} WHERE segment = ?;", id),
        PREPARE(_get_vector_prepared,
                "SELECT * FROM blas.vector_{};", id),
        PREPARE(_insert_value_prepared,
                "INSERT INTO blas.vector_{} (segment, idx, value) VALUES (?, ?, ?);", id),
        PREPARE(_clear_value_prepared,
                "DELETE FROM blas.vector_{} WHERE segment = ? AND idx = ?;", id),
        PREPARE(_clear_segment_prepared,
                "DELETE FROM blas.vector_{} WHERE segment = ?;", id),
        PREPARE(_resize_prepared,
                "UPDATE blas.vector_meta SET length = ? WHERE id = ?;"),
        PREPARE(_set_block_size_prepared,
                "UPDATE blas.vector_meta SET block_size = ? WHERE id = ?;")
#undef PREPARE
{
    get_meta_from_database();
}

void scylla_blas::basic_vector::resize(scylla_blas::index_t new_length) {
    _session->execute(_resize_prepared, new_length, id);
    this->length = new_length;
}

void scylla_blas::basic_vector::set_block_size(scylla_blas::index_t new_block_size) {
    _session->execute(_set_block_size_prepared, new_block_size, id);
    this->block_size = new_block_size;
}

