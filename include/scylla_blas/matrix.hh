#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>

#include <fmt/format.h>
#include <scmd.hh>

#include "scylla_blas/structure/matrix_block.hh"
#include "scylla_blas/structure/matrix_value.hh"
#include "scylla_blas/structure/vector_segment.hh"
#include "scylla_blas/utils/scylla_types.hh"
#include "config.hh"

namespace scylla_blas {

/* Matrix classes giving access to a matrix loaded into Scylla.
 * base_matrix: generic matrix operations
 * matrix<T>: matrix operations templated by matrix value type
 */
class basic_matrix {
protected:
    inline static constexpr index_type ceil_div (index_type a, index_type b) { return 1 + (a - 1) / b; }
    inline static constexpr index_type get_block_col(index_type j) { return ceil_div(j, BLOCK_SIZE); }
    inline static constexpr index_type get_block_row(index_type i) { return ceil_div(i, BLOCK_SIZE); }

    std::shared_ptr<scmd::session> _session;

    scmd::prepared_query _get_value_prepared;
    scmd::prepared_query _get_row_prepared;
    scmd::prepared_query _get_block_prepared;
    scmd::prepared_query _update_value_prepared;

    static std::pair<index_type, index_type> get_dimensions(const std::shared_ptr<scmd::session> &session, int64_t id);

public:
    // Should we make these private, with accessors?
    const index_type id;
    const index_type rows;
    const index_type columns;

    /* Height/width measured in blocks is equal to the block index of terminal blocks.
     * E.g. in a matrix that is 2 blocks wide the rightmost column belongs to the block number 2.
     */
    index_type get_blocks_width() const { return get_block_col(columns); }
    index_type get_blocks_height() const { return get_block_row(rows); }

    static void init_meta(const std::shared_ptr<scmd::session> &session);

    basic_matrix(const std::shared_ptr<scmd::session> &session, int64_t id);
};

template<class T>
class matrix : public basic_matrix {
    template<class... Args>
    std::vector<matrix_value<T>> get_vals_for_query(const scmd::prepared_query &query, Args... args) const {
        scmd::query_result result = _session->execute(query.get_statement().bind(args...));

        std::vector<matrix_value<T>> result_vector;
        while (result.next_row()) {
            result_vector.emplace_back(
                    result.get_column<index_type>("id_x"),
                    result.get_column<index_type>("id_y"),
                    result.get_column<T>("value")
            );
        }

        return result_vector;
    }
public:
    matrix(const std::shared_ptr<scmd::session> &session, int64_t id) : basic_matrix(session, id)
        { std::cerr << "A handle created to matrix " << id << std::endl; }

    /* We don't want to implicitly initialize a handle (somewhat costly) if it is discarded by the user.
     * Instead, let's have a version of init that does it explicitly, and a version that doesn't do it at all.
     * TODO: Can we do the same with one function and attributes for the compiler?
     */
    static void init(const std::shared_ptr<scmd::session>& session,
                     int64_t id, index_type rows, index_type columns, bool force_new = true) {
        std::cerr << "initializing matrix " << id << "..." << std::endl;

        scmd::statement create_table(fmt::format(R"(
            CREATE TABLE IF NOT EXISTS blas.matrix_{0} (
                block_x BIGINT,
                block_y BIGINT,
                id_x    BIGINT,
                id_y    BIGINT,
                value   {1},
                PRIMARY KEY (block_x, id_x, id_y));
        )", id, get_type_name<T>()));

        session->execute(create_table.set_timeout(0));
        std::cerr << "created matrix table for id = " << id << "..." << std::endl;

        if (force_new) {
            std::cerr << "Truncating..." << std::endl;
            scmd::statement drop_table(fmt::format("TRUNCATE blas.matrix_{0};", id));
            session->execute(drop_table.set_timeout(0));
            std::cerr << "Truncated!" << std::endl;
        }

        session->execute(fmt::format(R"(
            UPDATE blas.matrix_meta
            SET     rows    = {1},
                    columns = {2}
            WHERE   id      = {0};
        )", id, rows, columns));

        std::cerr << "initialized matrix " << id << std::endl;
    }

    static matrix init_and_return(const std::shared_ptr<scmd::session>& session,
                                  int64_t id, index_type rows, index_type columns, bool force_new = true) {
        init(session, id, rows, columns, force_new);
        return matrix<T>(session, id);
    }

    T get_value(index_type x, index_type y) const {
        auto ans_vec = get_vals_for_query(_get_value_prepared, get_block_row(x), get_block_col(y), x, y);

        if (!ans_vec.empty()) {
            return ans_vec[0].value;
        } else {
            return 0;
        }
    }

    vector_segment<T> get_row(index_type x) const {
        auto row_full = get_vals_for_query(_get_row_prepared, get_block_row(x), x);
        vector_segment<T> answer;

        for (matrix_value<T> &v : row_full) {
            answer.emplace_back(v.col_index, v.value);
        }

        return answer;
    }

    matrix_block<T> get_block(index_type x, index_type y) const {
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
        index_type offset_x = (x - 1) * BLOCK_SIZE;
        index_type offset_y = (y - 1) * BLOCK_SIZE;

        for (auto &val : block_values) {
            val.row_index -= offset_x;
            val.col_index -= offset_y;
        }

        return scylla_blas::matrix_block(block_values, id, x, y);
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

    void update_row(index_type x, vector_segment<T> row_data) {
        for (auto &val : row_data)
            update_value(x, val.index, val.value);
    }

    void update_block(index_type x, index_type y, const matrix_block<T> &block) {
        index_type offset_x = (x - 1) * BLOCK_SIZE;
        index_type offset_y = (y - 1) * BLOCK_SIZE;

        for (auto &val : block.get_values_raw())
            update_value(x, y, offset_x + val.row_index, offset_y + val.col_index, val.value);
    }
};

}