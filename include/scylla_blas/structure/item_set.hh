#pragma once

#include <iostream>
#include <memory>

#include <fmt/format.h>
#include <session.hh>

#include <scylla_blas/utils/scylla_types.hh>
#include <scylla_blas/utils/utils.hh>

namespace scylla_blas {

template<class T>
class item_set;

class empty_container_error : public std::runtime_error {
public:
    template<class... Args>
    empty_container_error(Args... args) : std::runtime_error(args...) {}
};

}


template<class T>
class scylla_blas::item_set {
    int64_t _id;
    std::shared_ptr<scmd::session> _session;

    /* Can these be safely copy-constructed to a different process? */
    scmd::prepared_query increase_counter_prepared;
    scmd::prepared_query insert_item_prepared;
    scmd::prepared_query read_id_prepared;
    scmd::prepared_query read_value_prepared;
    scmd::prepared_query remove_row_prepared;
    scmd::prepared_query decrease_counter_prepared;

public:
    template<class beginIt, class endIt>
    item_set(std::shared_ptr<scmd::session> session, beginIt begin, endIt end) :
            _id(get_timestamp()),
            _session((session->execute(fmt::format(R"(
                CREATE TABLE blas.item_set_{0} (
                    id bigint PRIMARY KEY,
                    value {1});
                )", _id, get_type_name<T>())), session)),
#define PREPARE(x, args...) x(_session->prepare(fmt::format(args)))
            PREPARE(increase_counter_prepared,
                    "UPDATE blas.item_set_meta SET cnt = cnt + 1 WHERE id = {0};", _id),
            PREPARE(insert_item_prepared,
                    "INSERT INTO blas.item_set_{0} (id, value) VALUES (?, ?);", _id),
            PREPARE(read_id_prepared,
                    "SELECT * FROM blas.item_set_meta WHERE id = {0};", _id),
            PREPARE(read_value_prepared,
                    "SELECT * FROM blas.item_set_{0} WHERE id = ?;", _id),
            PREPARE(remove_row_prepared,
                    "DELETE FROM blas.item_set_{0} WHERE id = ?;", _id),
            PREPARE(decrease_counter_prepared,
                    "UPDATE blas.item_set_meta SET cnt = cnt - 1 WHERE id = {0};", _id)
#undef PREPARE
    {
        for (int64_t item_id = 1; begin != end; item_id++, begin++) {
            _session->execute(increase_counter_prepared.get_statement());

            scmd::statement insert_statement = insert_item_prepared.get_statement();
            insert_statement.bind((int64_t) item_id, (T) *begin);
            _session->execute(insert_statement);
        }
    }

    T get_next() { /* FIXME: maybe there is an std::option that we could use instead of throwing an exception? */
        do {
            /* Spinlock */
            int64_t item_id;
            T result;
            scmd::statement read_statement = read_id_prepared.get_statement();
            scmd::query_result read_answer = _session->execute(read_statement);

            if (!read_answer.next_row() || (item_id = read_answer.get_column<int64_t>("cnt")) <= 0) {
                throw scylla_blas::empty_container_error(fmt::format("Set #{0} empty", _id));
            }

            /* We caught an id that might have not be taken yet */
            read_statement = read_value_prepared.get_statement();
            read_answer = _session->execute(read_statement.bind(item_id));

            if (read_answer.next_row()) {
                result = read_answer.get_column<T>("value");

                try {
                    _session->execute(remove_row_prepared.get_statement().bind(item_id));
                    _session->execute(decrease_counter_prepared.get_statement());
                    return result;
                } catch (const scmd::exception &e) {
                    /* Unable to delete: someone beat us to it. Try again. */
                    std::cerr << "Exception caught: " << e.what() << std::endl;
                }
            }
        } while (true);
    }
};

