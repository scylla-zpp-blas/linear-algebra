#pragma once

#include <cstddef>
#include <iostream>
#include <list>
#include <memory>

#include <query_result.hh>
#include <session.hh>

#include <scylla_blas/structure/matrix_block.hh>
#include <scylla_blas/structure/matrix_value.hh>
#include <scylla_blas/structure/vector.hh>
#include <scylla_blas/utils/scylla_types.hh>


namespace scylla_blas {

/*
* Matrix class giving access to matrix loaded into Scylla.
*/
template<class T>
class matrix {
private:
    static const index_type block_size = (1 << 8);

    static constexpr index_type get_block_col(index_type j) {
        return (j - 1) / block_size + 1;
    }

    static constexpr index_type get_block_row(index_type i) {
        return (i - 1) / block_size + 1;
    }

    static std::shared_ptr<scmd::session>
    prepare_session(std::shared_ptr<scmd::session> session, const std::string &id, bool force_new) {
        if (force_new) {
            /* Maybe truncate instead? */
            scmd::statement drop_table(fmt::format("DROP TABLE IF EXISTS blas.matrix_{0};", id));

            /* FIXME: Same problem as with creation; look below */
            cass_statement_set_request_timeout((CassStatement *)drop_table.get_statement(), 0);
            session->execute(drop_table);
        }

        /* TODO: consider a better key */
        scmd::statement create_table(fmt::format(R"(
        CREATE TABLE IF NOT EXISTS blas.matrix_{0} (
                    block_x bigint,
                    block_y bigint,
                    id_x bigint,
                    id_y bigint,
                    value {1},
                    PRIMARY KEY (block_x, id_x, id_y));
            )", id, get_type_name<T>()));

        /* FIXME: HACK!!!
         * We use this to avoid timeout errors.
         * Note the casting used to get rid of the const qualifier.
         * TODO: Implement this as a cass_statement method in the driver
         */
        cass_statement_set_request_timeout((CassStatement *)create_table.get_statement(), 0);
        session->execute(create_table);

        return session;
    }

    std::string _id;
    std::shared_ptr<scmd::session> _session;

    scmd::prepared_query _get_value_prepared;
    scmd::prepared_query _get_row_prepared;
    scmd::prepared_query _get_block_prepared;
    scmd::prepared_query _update_value_prepared;

    template<class... Args>
    std::vector<matrix_value<T>> get_vals_for_query(scmd::prepared_query &query, Args... args) {
        scmd::query_result result = _session->execute(query.get_statement().bind(args...));

        std::vector<matrix_value<T>> result_vector;
        while (result.next_row()) {
            result_vector.emplace_back(
                    result.get_column<int64_t>("id_x"),
                    result.get_column<int64_t>("id_y"),
                    result.get_column<T>("value")
            );
        }

        return result_vector;
    }

public:
    std::string get_id() {
        return _id;
    }

    matrix(std::shared_ptr<scmd::session> session, const std::string &id, bool force_new = false) :
            _id(id),
            _session(prepare_session(session, _id, force_new)),
#define PREPARE(x, args...) x(_session->prepare(fmt::format(args)))
            PREPARE(_get_value_prepared,
                    "SELECT * FROM blas.matrix_{} WHERE block_x = ? AND block_y = ? AND id_x = ? AND id_y = ? ALLOW FILTERING;",
                    _id),
            PREPARE(_get_row_prepared,
                    "SELECT * FROM blas.matrix_{} WHERE block_x = ? AND id_x = ?;", _id),
            PREPARE(_get_block_prepared,
                    "SELECT * FROM blas.matrix_{} WHERE block_x = ? AND block_y = ? ALLOW FILTERING;", _id),
            PREPARE(_update_value_prepared,
                    "INSERT INTO blas.matrix_{} (block_x, block_y, id_x, id_y, value) VALUES (?, ?, ?, ?, ?);", _id)
#undef PREPARE
    { std::cerr << "Initialized matrix " << id << std::endl; }

    T get_value(index_type x, index_type y) {
        auto ans_vec = get_vals_for_query(_get_value_prepared, get_block_row(x), get_block_col(y), x, y);

        if (!ans_vec.empty()) {
            return ans_vec[0].value;
        } else {
            return 0;
        }
    }

    vector<T> get_row(index_type x) {
        auto row_full = get_vals_for_query(_get_row_prepared, get_block_row(x), x);
        vector<T> answer;

        for (matrix_value<T> &v : row_full) {
            answer.emplace_back(v.col_index, v.value);
        }

        return answer;
    }

    matrix_block<T> get_block(index_type x, index_type y) {
        auto block_values = get_vals_for_query(_get_block_prepared, x, y);

        /* Move by offset – a block is an independent unit.
         * E.g. if we have a block sized 2x2, then for such a matrix:
         * ---------
         * |1 0 0 0|
         * |1 1 0 0|
         * |0 0 1 0|
         * |0 0 1 1|
         * ---------
         * both blocks (1, 1) and (2, 2) will have identical sets of coordinates for all values.
         * This should make further operations on abstract blocks easier by a bit.
         */
        index_type offset_x = (x - 1) * block_size;
        index_type offset_y = (y - 1) * block_size;

        for (auto &val : block_values) {
            val.row_index -= offset_x;
            val.col_index -= offset_y;
        }

        return scylla_blas::matrix_block(block_values, _id, x, y);
    }

    void update_value(index_type x, index_type y, T value) {
        _session->execute(_update_value_prepared.get_statement()
                                  .bind(get_block_row(x), get_block_col(y), x, y, value));
    }

    void update_value(index_type block_x, index_type block_y, index_type x, index_type y, T value) {
        _session->execute(_update_value_prepared.get_statement()
                                  .bind(block_x, block_y, x, y, value));
    }

    /* TODO: make a better implementation of the update methods below */
    void update_values(std::vector<matrix_value<T>> values) {
        for (auto &val: values)
            update_value(val.row_index, val.col_index, val.value);
    }

    void update_row(index_type x, vector<T> row_data) {
        for (auto &val : row_data)
            update_value(x, val.index, val.value);
    }

    void update_block(index_type x, index_type y, const matrix_block<T> &block) {
        index_type offset_x = (x - 1) * block_size;
        index_type offset_y = (y - 1) * block_size;

        for (auto &val : block.get_values_raw())
            update_value(x, y, offset_x + val.row_index, offset_y + val.col_index, val.value);
    }
};

}