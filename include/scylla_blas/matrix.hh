#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>

#include <fmt/format.h>
#include <scmd.hh>

#include "scylla_blas/logging/logging.hh"
#include "scylla_blas/structure/matrix_block.hh"
#include "scylla_blas/structure/matrix_value.hh"
#include "scylla_blas/structure/vector_segment.hh"
#include "scylla_blas/utils/scylla_types.hh"
#include "config.hh"

namespace scylla_blas {

/* Matrix classes giving access to a matrix loaded into Scylla.
 * basic_matrix: generic matrix operations
 * matrix<T>: matrix operations templated by matrix value type
 */
class basic_matrix {
protected:
    std::shared_ptr<scmd::session> _session;

    scmd::prepared_query _get_meta_prepared;
    scmd::prepared_query _get_value_prepared;
    scmd::prepared_query _get_row_prepared;
    scmd::prepared_query _get_block_prepared;
    scmd::prepared_query _insert_value_prepared;
    scmd::prepared_query _clear_all_prepared;
    scmd::prepared_query _clear_row_prepared;
    scmd::prepared_query _resize_prepared;
    scmd::prepared_query _set_block_size_prepared;

    id_t id;
    index_t row_count;
    index_t column_count;
    index_t block_size;

    inline static constexpr index_t ceil_div (index_t a, index_t b) { return 1 + (a - 1) / b; }

    index_t get_block_col(index_t j) const { return ceil_div(j, block_size); }
    index_t get_block_row(index_t i) const { return ceil_div(i, block_size); }

    void update_meta();

public:
    /* Removes all values inserted into the matrix up to the point of execution.
     * Doesn't remove the matrix itself or modify its metadata, so it doesn't need
     * to be reinitialized for further usage.
     *
     * This static private method works same as the object-specific, public method "clear_all".
     * The only difference is that a separate object needs not be initialized.
     */
    static void clear(const std::shared_ptr<scmd::session> &session, id_t id);

    /* Changes matrices' dimensions in the database.
     * Those can be used by the user for custom assertions, although the class itself
     * doesn't make ANY validity checks concerning dimensions of stored or returned data.
     */
    static void resize(const std::shared_ptr<scmd::session> &session,
                       id_t id, index_t new_row_count, index_t new_column_count);

    /*
     * Sets matrix block size. Should only be called when matrix is empty - it does not
     * change matrix data to fit new block size. Calling on non-empty matrix WILL cause
     * wrong results later on.
     */
    static void set_block_size(const std::shared_ptr<scmd::session> &session, id_t id, index_t new_block_size);

    /* Deletes matrix and all of its data.
     */
    static void drop(const std::shared_ptr<scmd::session> &session, id_t id);

    static void init_meta(const std::shared_ptr<scmd::session> &session);

    basic_matrix(const std::shared_ptr<scmd::session> &session, id_t id);
    basic_matrix(const basic_matrix& other) = delete;
    basic_matrix& operator=(const basic_matrix &other) = delete;
    basic_matrix(basic_matrix&& other) noexcept = default;
    basic_matrix& operator=(basic_matrix&& other) noexcept = default;

    bool operator==(const basic_matrix &other) const {
        return this->id == other.id;
    }

    id_t get_id() const {
        return this->id;
    }

    index_t get_block_size() const {
        return this->block_size;
    }

    index_t get_column_count(TRANSPOSE trans = NoTrans) const {
        if (trans != NoTrans) return get_row_count();
        return column_count;
    }
    index_t get_row_count(TRANSPOSE trans = NoTrans) const {
        if (trans != NoTrans) return get_column_count();
        return row_count;
    }

    /* Height/width measured in blocks is equal to the block index of terminal blocks.
     * E.g. in a matrix that is 2 blocks wide the rightmost column belongs to the block number 2.
     */
    index_t get_blocks_width(TRANSPOSE trans = NoTrans) const {
        if (trans != NoTrans) return get_blocks_height();

        return get_block_col(column_count);
    }
    index_t get_blocks_height(TRANSPOSE trans = NoTrans) const {
        if (trans != NoTrans) return get_blocks_width();

        return get_block_row(row_count);
    }

    /* Returns the column range of possibly non-zero blocks for given block_row,
     * assuming that the matrix is banded, with given KL and KU parameters
     */
    std::pair<index_t, index_t> get_banded_block_limits_for_row(index_t block_row, index_t KL,
                                                                index_t KU, TRANSPOSE trans = NoTrans) const {
        index_t start = std::max(block_row - ceil_div(KL, block_size), index_t(0));
        index_t end = std::min(get_blocks_width(trans), block_row + ceil_div(KU, block_size));

        return {start, end};
    }

    /* Returns the row range of possibly non-zero blocks for given block_column,
     * assuming that the matrix is banded, with given KL and KU parameters
     */
    std::pair<index_t, index_t> get_banded_block_limits_for_column(index_t block_column, index_t KL,
                                                                   index_t KU, TRANSPOSE trans = NoTrans) const {
        index_t start = std::max(block_column - ((KU - 1) / block_size + 1), index_t(0));
        index_t end = std::min(get_blocks_height(trans), block_column + KL / block_size);

        return {start, end};
    }

    void clear_row(index_t x);
    void clear_all();
    void resize(index_t new_row_count, index_t new_column_count);
    void set_block_size(index_t new_block_size);
};

template<class T>
class matrix : public basic_matrix {
public:
    /* We don't want to implicitly initialize a handle (somewhat costly) if it is discarded by the user.
     * Instead, let's have a version of init that does it explicitly, and a version that doesn't do it at all.
     * TODO: Can we do the same with one function and attributes for the compiler?
     */
    static void init(const std::shared_ptr<scmd::session> &session,
                     id_t id, index_t row_count, index_t column_count,
                     bool force_new = true, index_t block_size = DEFAULT_BLOCK_SIZE) {
        LogInfo("initializing matrix {}...", id);

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

        if (force_new) {
            clear(session, id);
        }

        resize(session, id, row_count, column_count);
        set_block_size(session, id, block_size);

        LogInfo("Initialized matrix {}", id);
    }

    static matrix init_and_return(const std::shared_ptr<scmd::session> &session,
                                  id_t id, index_t row_count, index_t column_count,
                                  bool force_new = true, index_t block_size = DEFAULT_BLOCK_SIZE) {
        init(session, id, row_count, column_count, force_new, block_size);
        return matrix<T>(session, id);
    }

    matrix(const std::shared_ptr<scmd::session> &session, id_t id) : basic_matrix(session, id)
        { LogTrace("A handle created to matrix {}", id); }
    matrix(const matrix& other) = delete;
    matrix& operator=(const matrix &other) = delete;
    matrix(matrix&& other) noexcept = default;
    matrix& operator=(matrix&& other) noexcept = default;

    T get_value(index_t x, index_t y, TRANSPOSE trans = NoTrans) const {
        if (trans != NoTrans) std::swap(x, y);

        auto ans_vec = get_vals_for_query(_get_value_prepared, get_block_row(x), get_block_col(y), x, y);

        if (!ans_vec.empty()) {
            return ans_vec[0].value;
        } else {
            return 0;
        }
    }

    vector_segment<T> get_row(index_t x) const {
        auto row_full = get_vals_for_query(_get_row_prepared, get_block_row(x), x);
        vector_segment<T> answer;

        for (matrix_value<T> &v : row_full) {
            answer.emplace_back(v.col_index, v.value);
        }

        return answer;
    }

    matrix_block<T> get_block(index_t x, index_t y, TRANSPOSE trans = NoTrans) const {
        if (trans != NoTrans) std::swap(x, y);

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
        index_t offset_x = (x - 1) * block_size;
        index_t offset_y = (y - 1) * block_size;

        for (auto &val : block_values) {
            val.row_index -= offset_x;
            val.col_index -= offset_y;
        }

        return scylla_blas::matrix_block(block_values, x, y, trans);
    }

    void insert_value(index_t x, index_t y, T value) {
        if (std::abs(value) < EPSILON) return;

        _session->execute(_insert_value_prepared, get_block_row(x), get_block_col(y), x, y, value);
    }

    void insert_value(index_t block_x, index_t block_y, index_t x, index_t y, T value) {
        if (std::abs(value) < EPSILON) return;

        _session->execute(_insert_value_prepared, block_x, block_y, x, y, value);
    }

    void insert_values(const std::vector<matrix_value<T>> &values) {
        std::string inserts = "";

        for (auto &val: values) {
            /* We do not want to store values equal or close to 0 */
            if (std::abs(val.value) < EPSILON) continue;

            inserts += fmt::format(
                    "INSERT INTO blas.matrix_{} (block_x, block_y, id_x, id_y, value) VALUES ({}, {}, {}, {}, {}); ",
                    id, get_block_row(val.row_index), get_block_col(val.col_index),
                    val.row_index, val.col_index, val.value);
        }

        /* Don't send a query if there's no need to do so. */
        if (inserts == "") return;

        _session->execute("BEGIN BATCH " + inserts + "APPLY BATCH;");
    }

    /* Inserts a given block into the matrix. Old values will not be modified or deleted */
    void insert_row(index_t x, const vector_segment<T> &row_data) {
        std::vector<matrix_value<T>> values;

        for (auto &val : row_data)
            values.emplace_back(x, val.index, val.value);

        insert_values(values);
    }

    void update_row(index_t x, const vector_segment<T> &row_data) {
        clear_row(x);
        insert_row(x, row_data);
    }

    /* Inserts a given block into the matrix. Old values will not be modified or deleted */
    /* TODO: investigate */
    void insert_block(index_t row, index_t column, const matrix_block<T> &block) {
        std::vector<matrix_value<T>> values = block.get_values_raw();
        index_t offset_row = (row - 1) * block_size;
        index_t offset_column = (column - 1) * block_size;

        for (auto &val : values) {
            val.row_index += offset_row;
            val.col_index += offset_column;
        }

        insert_values(values);
    }

private:
    template<class... Args>
    std::vector<matrix_value<T>> get_vals_for_query(const scmd::prepared_query &query, Args... args) const {
        scmd::query_result result = _session->execute(query, args...);

        std::vector<matrix_value<T>> result_vector;
        while (result.next_row()) {
            result_vector.emplace_back(
                    result.get_column<index_t>("id_x"),
                    result.get_column<index_t>("id_y"),
                    result.get_column<T>("value")
            );
        }

        return result_vector;
    }
};

}