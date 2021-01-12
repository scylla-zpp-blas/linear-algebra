#pragma once

#include <boost/core/demangle.hpp>
#include <cstddef>
#include <iostream>
#include <list>
#include <memory>

#include "scylla_types.hh"
#include "structure/vector.hh"

#include "query_result.hh"
#include "session.hh"

namespace scylla_blas {
/**
 * Matrix class giving access to matrix loaded into Scylla.
 */
    template<class T>
    class scylla_matrix {
    private:
        std::string _id;
        std::shared_ptr<scmd::session> _session;

    public:
        std::string get_id() {
            return _id;
        }

        scylla_matrix(std::shared_ptr<scmd::session> session, std::string id, bool force = false) : _id(id), _session(session) {
            std::string value_type_name = boost::core::demangle(typeid(T).name()); // FIXME: May this be not portable?
            std::cerr << "Creating a matrix handle for matrix \'" << _id
                      << "\' of type: " << value_type_name << std::endl;

            _session->execute(fmt::format(R"(
                CREATE TABLE IF NOT EXISTS blas.{0} (
                    idx bigint,
                    idy bigint,
                    value {1},
                    PRIMARY KEY (idx, idy)
                ) WITH CLUSTERING ORDER BY (idy ASC);)", _id, value_type_name));

            if (force) {
                _session->execute(fmt::format("TRUNCATE blas.{0};", _id));
            }
        }

        T get_value(index_type x, index_type y) {
            std::string query = fmt::format(R"(
                SELECT * FROM blas.{0} WHERE idx={1} AND idy={2};
            )", _id, x, y);

            std::cerr << query << std::endl;
            scmd::query_result result = _session->execute(query);

            std::cerr << "Obtained result for: (" << x << ", " << y << ")" << std::endl;
            if (result.next_row()) {
                return result.get_column<T>("value");
            } else {
                return 0;
            }
        }

        scylla_blas::vector<T> get_row(index_type x) {
            std::string query = fmt::format(R"(
                SELECT * FROM blas.{0} WHERE idx={1};
            )", _id, x);

            std::cerr << query << std::endl;
            scmd::query_result result = _session->execute(query);

            std::cerr << "Obtained row " << x << std::endl;
            scylla_blas::vector<T> matrix_row;
            while (result.next_row()) {
                matrix_row.emplace_back(
                    result.get_column<int64_t>("idy"),
                    result.get_column<T>("value")
                );
            }

            return matrix_row;
        }

        void update_value(index_type x, index_type y, T value) {
            std::string query = fmt::format(R"(
                INSERT INTO blas.{0} (idx, idy, value) VALUES ({1}, {2}, {3});
            )", _id, x, y, value);
            std::cerr << query << std::endl;

            _session->execute(query);
        }

        void update_row(index_type x, scylla_blas::vector<T> row_data) {
            if (row_data.empty())
                return;

            std::cout << "Size: " << row_data.size() << std::endl;

            std::string query = fmt::format(R"(
                BEGIN BATCH {} APPLY BATCH;
            )", fmt::join(std::vector<std::string>(row_data.size(),fmt::format(R"(
                INSERT INTO blas.{0} (idx, idy, value) VALUES ({1}, ?, ?);
            )", _id, x)), ""));

            std::cerr << query << std::endl;

            scmd::prepared_query insert_row_prepared = _session->prepare(query);
            scmd::statement insert_statement = insert_row_prepared.get_statement();

            for (auto entry : row_data) {
                insert_statement.bind((int64_t)entry.index, entry.value);
            }
            _session->execute(insert_statement);
        }
    };
}