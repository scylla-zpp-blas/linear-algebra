#include <queue>

#include <boost/test/unit_test.hpp>

#include "scylla_blas/queue/scylla_queue.hh"
#include "scylla_blas/matrix.hh"
#include "fixture.hh"

BOOST_FIXTURE_TEST_SUITE(structure_tests, scylla_fixture)

BOOST_AUTO_TEST_CASE(matrices)
{
    scylla_blas::matrix<float>::init(session, 0, true);

    auto matrix = scylla_blas::matrix<float>(session, 0);
    auto matrix_2 = scylla_blas::matrix<float>(session, 0);

    matrix.update_value(1, 0, M_PI);
    matrix.update_value(1, 1, 42);
    BOOST_REQUIRE_EQUAL(std::ceil(matrix.get_value(1, 0) * 10000), std::ceil(M_PI * 10000));
    BOOST_REQUIRE_EQUAL(matrix.get_value(1, 1), 42);

    matrix.update_value(1, 0, M_PI);
    BOOST_REQUIRE_EQUAL(std::ceil(matrix.get_value(1, 0) * 10000), std::ceil(M_PI * 10000));

    matrix.update_value(1, 1, 100);
    BOOST_REQUIRE_EQUAL(std::ceil(matrix.get_value(1, 0) * 10000), std::ceil(M_PI * 10000));
    BOOST_REQUIRE_EQUAL(matrix.get_value(1, 1), 100);
}


BOOST_AUTO_TEST_CASE(vectors)
{
    auto vector_1 = scylla_blas::vector_segment<float>();

    for (int i = 0; i < 10; i++) {
        vector_1.emplace_back(i, 10);
    }

    std::cout << "Vec_1: ";
    for (auto entry : vector_1) {
        std::cout << "(" << entry.index << ": " << entry.value << "), ";
    }
    std::cout << std::endl;

    auto vector_2 = scylla_blas::vector_segment<float>();
    for (int i = 0; i < 5; i++) {
        vector_2.emplace_back(i, (float)M_PI * i * i);
    }
    for (int i = 15; i < 20; i++) {
        vector_2.emplace_back(i, (float) M_PI * i * i);
    }

    std::cout << "Vec_2: ";
    for (auto entry : vector_2) {
        std::cout << "(" << entry.index << ": " << entry.value << "), ";
    }
    std::cout << std::endl;

    vector_1 *= M_E;
    std::cout << "Vec_1 * e: ";
    for (auto entry : vector_1) {
        std::cout << "(" << entry.index << ": " << entry.value << "), ";
    }
    std::cout << std::endl;

    vector_1 += vector_2;
    std::cout << "Vec_1 * e + Vec_2: ";
    for (auto entry : vector_1) {
        std::cout << "(" << entry.index << ": " << entry.value << "), ";
    }
    std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(scylla_queue_sp_mc)
{
    scylla_blas::scylla_queue::delete_queue(session, 1337);
    scylla_blas::scylla_queue::create_queue(session, 1337);
    BOOST_REQUIRE(scylla_blas::scylla_queue::queue_exists(session, 1337));
    auto queue = scylla_blas::scylla_queue(session, 1337);

    std::vector<int64_t> values = {0, 42, 1410, 1, 1999, 2021, 1000 * 1000 * 1000 + 7, 406};
    std::queue<int64_t> task_ids = {};
    for(auto val : values) {
        scylla_blas::proto::task task {
            .type = scylla_blas::proto::NONE,
            .basic {
                .data = val
            }
        };
        task_ids.push(queue.produce(task));
    }

    for(auto val : values) {
        auto [id, task] = queue.consume();
        BOOST_REQUIRE_EQUAL(val, task.basic.data);
        queue.mark_as_finished(id);
        BOOST_REQUIRE(queue.is_finished(task_ids.front()));
        task_ids.pop();
    }
}

BOOST_AUTO_TEST_CASE(scylla_queue_mp_sc)
{
    scylla_blas::scylla_queue::delete_queue(session, 1337);
    scylla_blas::scylla_queue::create_queue(session, 1337, true, false);
    BOOST_REQUIRE(scylla_blas::scylla_queue::queue_exists(session, 1337));
    auto queue = scylla_blas::scylla_queue(session, 1337);

    std::vector<int64_t> values = {0, 42, 1410, 1, 1999, 2021, 1000 * 1000 * 1000 + 7, 406};
    std::queue<int64_t> task_ids = {};
    for(auto val : values) {
        scylla_blas::proto::task task {
            .type = scylla_blas::proto::NONE,
            .basic {
                .data = val
            }
        };
        task_ids.push(queue.produce(task));
    }

    for(auto val : values) {
        auto [id, task] = queue.consume();
        BOOST_REQUIRE_EQUAL(val, task.basic.data);
        queue.mark_as_finished(id);
        BOOST_REQUIRE(queue.is_finished(task_ids.front()));
        task_ids.pop();
    }
}

BOOST_AUTO_TEST_CASE(scylla_queue_sp_sc)
{
    scylla_blas::scylla_queue::delete_queue(session, 1337);
    scylla_blas::scylla_queue::create_queue(session, 1337, false, false);
    BOOST_REQUIRE(scylla_blas::scylla_queue::queue_exists(session, 1337));
    auto queue = scylla_blas::scylla_queue(session, 1337);

    std::vector<int64_t> values = {0, 42, 1410, 1, 1999, 2021, 1000 * 1000 * 1000 + 7, 406};
    std::queue<int64_t> task_ids = {};
    for(auto val : values) {
        scylla_blas::proto::task task {
            .type = scylla_blas::proto::NONE,
            .basic {
                .data = val
            }
        };
        task_ids.push(queue.produce(task));
    }

    for(auto val : values) {
        auto [id, task] = queue.consume();
        BOOST_REQUIRE_EQUAL(val, task.basic.data);
        queue.mark_as_finished(id);
        BOOST_REQUIRE(queue.is_finished(task_ids.front()));
        task_ids.pop();
    }
}

BOOST_AUTO_TEST_CASE(scylla_queue_mp_mc)
{
    scylla_blas::scylla_queue::delete_queue(session, 1337);
    scylla_blas::scylla_queue::create_queue(session, 1337, true, true);
    BOOST_REQUIRE(scylla_blas::scylla_queue::queue_exists(session, 1337));
    auto queue = scylla_blas::scylla_queue(session, 1337);

    std::vector<int64_t> values = {0, 42, 1410, 1, 1999, 2021, 1000 * 1000 * 1000 + 7, 406};
    std::queue<int64_t> task_ids = {};
    for(auto val : values) {
        scylla_blas::proto::task task {
            .type = scylla_blas::proto::NONE,
            .basic {
                .data = val
            }
        };
        task_ids.push(queue.produce(task));
    }

    for(auto val : values) {
        auto [id, task] = queue.consume();
        BOOST_REQUIRE_EQUAL(val, task.basic.data);
        queue.mark_as_finished(id);
        BOOST_REQUIRE(queue.is_finished(task_ids.front()));
        task_ids.pop();
    }
}

BOOST_AUTO_TEST_SUITE_END();

