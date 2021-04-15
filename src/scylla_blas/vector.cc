#include "scylla_blas/vector.hh"

void scylla_blas::basic_vector::init_meta(const std::shared_ptr<scmd::session> &session) {
    scmd::statement init_meta(R"(CREATE TABLE IF NOT EXISTS blas.vector_meta (
                                                id           BIGINT PRIMARY KEY,
                                                length   BIGINT);)");
    session->execute(init_meta.set_timeout(0));
}

scylla_blas::index_type scylla_blas::basic_vector::get_length() const {
    scmd::query_result result = _session->execute(_get_meta_prepared);

    if (!result.next_row()) {
        throw std::runtime_error(fmt::format("Meta info for vector {} not found in the database.", id));
    }

    return result.get_column<index_type>("length");
}

scylla_blas::basic_vector::basic_vector(const std::shared_ptr<scmd::session> &session, int64_t id) :
        _session(session),
        id(id),
        length(get_length()),
#define PREPARE(x, args...) x(_session->prepare(fmt::format(args)))
        PREPARE(_get_meta_prepared,
                "SELECT * FROM blas.vector_meta WHERE id = {};", id),
        PREPARE(_get_value_prepared,
                "SELECT * FROM blas.vector_{} WHERE segment = ? AND idx = ? ALLOW FILTERING;", id),
        PREPARE(_get_segment_prepared,
                "SELECT * FROM blas.vector_{} WHERE segment = ? ALLOW FILTERING;", id),
        PREPARE(_get_vector_prepared,
                "SELECT * FROM blas.vector_{};", id),
        PREPARE(_insert_value_prepared,
                "INSERT INTO blas.vector_{} (segment, idx, value) VALUES (?, ?, ?);", id),
        PREPARE(_clear_value_prepared,
                "DELETE FROM blas.vector_{} WHERE segment = ? AND idx = ?;", id),
        PREPARE(_clear_segment_prepared,
                "DELETE FROM blas.vector_{} WHERE segment = ?;", id)
#undef PREPARE
{ }

void scylla_blas::basic_vector::clear(const std::shared_ptr<scmd::session> &session, int64_t id) {
    scmd::statement drop_table(fmt::format("TRUNCATE blas.vector_{0};", id));
    session->execute(drop_table.set_timeout(0));
}

void scylla_blas::basic_vector::resize(const std::shared_ptr<scmd::session> &session,
                                       int64_t id, int64_t new_length) {
    session->execute(fmt::format(R"(
            UPDATE blas.vector_meta
                SET           length     = {1}
                WHERE   id             = {0};
        )", id, new_length));
}
