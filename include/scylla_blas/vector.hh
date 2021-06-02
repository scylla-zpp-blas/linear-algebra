#pragma once

#include <memory>
#include <iostream>
#include <algorithm>

#include <fmt/format.h>
#include <scmd.hh>

#include "scylla_blas/logging/logging.hh"
#include "scylla_blas/structure/vector_segment.hh"
#include "scylla_blas/structure/vector_value.hh"
#include "scylla_blas/utils/scylla_types.hh"
#include "config.hh"

namespace scylla_blas {

/* Vector classes giving access to a vector loaded into Scylla.
 * Based on the implementation of matrix classes.
 */
class basic_vector {
protected:
    inline static constexpr index_type ceil_div (index_type a, index_type b) { return 1 + (a - 1) / b; }
    inline static constexpr index_type get_segment_index(index_type i) { return ceil_div(i, BLOCK_SIZE); }

    std::shared_ptr<scmd::session> _session;

    scmd::prepared_query _get_meta_prepared;
    scmd::prepared_query _get_value_prepared;
    scmd::prepared_query _get_segment_prepared;
    scmd::prepared_query _get_vector_prepared;
    scmd::prepared_query _insert_value_prepared;
    scmd::prepared_query _clear_value_prepared;
    scmd::prepared_query _clear_segment_prepared;

    index_type get_length() const;

public:
    // Should we make these private, with accessors?
    const index_type id;
    const index_type length;

    /*
     * Length measured in segments is equal to the index of the last segment.
     */
    index_type get_segment_count() const {
        return get_segment_index(length);
    }

    index_type get_segment_offset(index_type segment_number) const {
        return (segment_number - 1) * BLOCK_SIZE;
    }

    static void init_meta(const std::shared_ptr<scmd::session> &session);

    basic_vector(const std::shared_ptr<scmd::session> &session, int64_t id);

    basic_vector(basic_vector&& other) :
        _session(std::move(other._session)),
        _get_meta_prepared(std::move(other._get_meta_prepared)),
        _get_value_prepared(std::move(other._get_value_prepared)),
        _get_segment_prepared(std::move(other._get_segment_prepared)),
        _get_vector_prepared(std::move(other._get_vector_prepared)),
        _insert_value_prepared(std::move(other._insert_value_prepared)),
        _clear_value_prepared(std::move(other._clear_value_prepared)),
        _clear_segment_prepared(std::move(other._clear_segment_prepared)),
        id (other.id),
        length (other.length)
    { }

    bool operator==(const basic_vector &other) const {
        return this->id == other.id;
    }

    static void clear(const std::shared_ptr<scmd::session> &session, int64_t id);
    static void resize(const std::shared_ptr<scmd::session> &session,
                       int64_t id, int64_t new_legnth);

    void clear_all() { clear(_session, id); }
};

template<class T>
class vector : public basic_vector {
    template<class... Args>
    std::vector<vector_value<T>> get_vals_for_query(const scmd::prepared_query &query, Args... args) const {
        scmd::query_result result = _session->execute(query.get_statement().bind(args...));

        std::vector<vector_value<T>> result_vector;
        while (result.next_row()) {
            result_vector.emplace_back(
                    result.get_column<index_type>("idx"),
                    result.get_column<T>("value")
            );
        }

        return result_vector;
    }

public:
    vector(const std::shared_ptr<scmd::session> &session, int64_t id) : basic_vector(session, id)
        { LogInfo("A handle created to vector {}", id); }

    vector(vector &&other) : basic_vector(other) {}

    /* We don't want to implicitly initialize a handle (somewhat costly) if it is discarded by the user.
     * Instead, let's have a version of init that does it explicitly, and a version that doesn't do it at all.
     * TODO: Can we do the same with one function and attributes for the compiler?
     */
    static void init(const std::shared_ptr<scmd::session> &session,
                     int64_t id, index_type length, bool force_new = true) {
        LogInfo("initializing vector {}...", id);

        scmd::statement create_table(fmt::format(R"(
            CREATE TABLE IF NOT EXISTS blas.vector_{0} (
                segment BIGINT,
                idx    BIGINT,
                value   {1},
                PRIMARY KEY (segment, idx));
        )", id, get_type_name<T>()));

        session->execute(create_table.set_timeout(0));

        if (force_new) {
            clear(session, id);
        }

        resize(session, id, length);

        LogInfo("Initialized vector {}", id);
    }

    static vector init_and_return(const std::shared_ptr<scmd::session> &session,
                                  int64_t id, index_type length, bool force_new = true) {
        init(session, id, length, force_new);
        return vector<T>(session, id);
    }

    T get_value(index_type x) const {
        auto ans_vec = get_vals_for_query(_get_value_prepared, get_segment_index(x), x);

        if (!ans_vec.empty()) {
            return ans_vec[0].value;
        } else {
            return 0;
        }
    }

    vector_segment<T> get_segment(index_type x) const {
        std::vector<vector_value<T>> segment_values = get_vals_for_query(_get_segment_prepared, x);

        /*
         * Segments are moved by offset similarly to matrix blocks.
         */
        index_type offset = get_segment_offset(x);

        vector_segment<T> answer;
        for (auto &val : segment_values) {
            answer.emplace_back(val.index - offset, val.value);
        }

        return answer;
    }

    /* The vectors can be very large so we probably only want to use get_whole for visualization/testing purposes */
    vector_segment<T> get_whole() const {
        std::vector<vector_value<T>> values = get_vals_for_query(_get_vector_prepared);

        vector_segment<T> answer;
        for (auto &val : values) {
            answer.emplace_back(val.index, val.value);
        }
        sort(answer.begin(), answer.end(), [](vector_value<T> a, vector_value<T> b) {return a.index < b.index; });
        return answer;
    }

    void clear_value(index_type x) {
        _session->execute(_clear_value_prepared.get_statement()
                                  .bind (get_segment_index(x), x));
    }

    void clear_segment(index_type x) {
        _session->execute(_clear_segment_prepared.get_statement()
                                  .bind(x));
    }

    void update_value(index_type x, T value) {
        if (std::abs(value) < EPSILON) {
            clear_value(x);
            return;
        }

        _session->execute(_insert_value_prepared.get_statement()
                                  .bind(get_segment_index(x), x, value));
    }

    void update_values(const std::vector<vector_value<T>> &values) {
        scmd::batch_query batch(CASS_BATCH_TYPE_UNLOGGED);

        for (auto &val: values) {
            if (std::abs(val.value) < EPSILON) {
                auto stmt = _clear_value_prepared.get_statement();
                batch.add_statement(stmt.bind(get_segment_index(val.index), val.index));
            } else {
                auto stmt = _insert_value_prepared.get_statement();
                batch.add_statement(stmt.bind(get_segment_index(val.index), val.index, val.value));
            }
        }

        _session->execute(batch);
    }

    void update_segment(index_type x, vector_segment<T> segment_data) {
        index_type offset = (x - 1) * BLOCK_SIZE;

        for (auto &val : segment_data) {
            val.index += offset;
        }

        clear_segment(x);
        update_values(segment_data);
    }

};

}
