#include <queue>

#include <boost/test/unit_test.hpp>

#include "scylla_blas/queue/scylla_queue.hh"
#include "fixture.hh"

BOOST_FIXTURE_TEST_SUITE(queue_tests, scylla_fixture)

static std::vector<int64_t> values = {0, 42, 1410, 1, 1999, 2021, 1000 * 1000 * 1000 + 7, 406};

static void test_queue_simple(scylla_blas::scylla_queue& queue) {
    // Test tasks without results
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
        auto opt = queue.consume();
        BOOST_REQUIRE(opt);
        auto [id, task] = opt.value();
        BOOST_REQUIRE_EQUAL(val, task.basic.data);
        BOOST_REQUIRE(!queue.is_finished(task_ids.front()));
        BOOST_REQUIRE(!queue.get_response(task_ids.front()));
        queue.mark_as_finished(id);
        auto r2_opt = queue.get_response(task_ids.front());
        BOOST_REQUIRE(r2_opt.has_value());
        auto r2 = r2_opt.value();
        BOOST_REQUIRE(r2.type == scylla_blas::proto::R_NONE);
        BOOST_REQUIRE(queue.is_finished(task_ids.front()));

        task_ids.pop();
    }
}

static void test_queue_response(scylla_blas::scylla_queue& queue) {
    std::queue<int64_t> task_ids = {};
    // Test tasks with basic results
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
        auto opt = queue.consume();
        BOOST_REQUIRE(opt);
        auto [id, task] = opt.value();
        scylla_blas::proto::response r{};
        r.type = scylla_blas::proto::R_INT64;
        r.simple.response = val;
        BOOST_REQUIRE_EQUAL(val, task.basic.data);
        BOOST_REQUIRE(!queue.is_finished(task_ids.front()));
        BOOST_REQUIRE(!queue.get_response(task_ids.front()));
        queue.mark_as_finished(id, r);
        auto r2_opt = queue.get_response(task_ids.front());
        BOOST_REQUIRE(r2_opt.has_value());
        auto r2 = r2_opt.value();
        BOOST_REQUIRE(r2.type == scylla_blas::proto::R_INT64);
        BOOST_REQUIRE(r.simple.response == val);
        BOOST_REQUIRE(queue.is_finished(task_ids.front()));

        task_ids.pop();
    }
}

static void test_queue_batch(scylla_blas::scylla_queue& queue) {
        // Test tasks without results
        int64_t tasks_id;
        std::vector<scylla_blas::proto::task> tasks;
        for(auto val : values) {
            scylla_blas::proto::task task {
                    .type = scylla_blas::proto::NONE,
                    .basic {
                            .data = val
                    }
            };
            tasks.push_back(task);
        }

        tasks_id = queue.produce(tasks);

        for(auto val : values) {
            auto opt = queue.consume();
            BOOST_REQUIRE(opt);
            auto [id, task] = opt.value();
            BOOST_REQUIRE_EQUAL(val, task.basic.data);
            BOOST_REQUIRE(!queue.is_finished(tasks_id));
            BOOST_REQUIRE(!queue.get_response(tasks_id));
            queue.mark_as_finished(id);
            auto r2_opt = queue.get_response(tasks_id);
            BOOST_REQUIRE(r2_opt.has_value());
            auto r2 = r2_opt.value();
            BOOST_REQUIRE(r2.type == scylla_blas::proto::R_NONE);
            BOOST_REQUIRE(queue.is_finished(tasks_id));

            tasks_id++;
        }
    }

BOOST_AUTO_TEST_CASE(scylla_queue_sp_mc)
{
    scylla_blas::scylla_queue::delete_queue(session, 1337);
    scylla_blas::scylla_queue::create_queue(session, 1337);
    BOOST_REQUIRE(scylla_blas::scylla_queue::queue_exists(session, 1337));
    auto queue = scylla_blas::scylla_queue(session, 1337);
    test_queue_simple(queue);
    test_queue_response(queue);
    test_queue_batch(queue);
}

BOOST_AUTO_TEST_CASE(scylla_queue_mp_sc)
{
    scylla_blas::scylla_queue::delete_queue(session, 1337);
    scylla_blas::scylla_queue::create_queue(session, 1337, true, false);
    BOOST_REQUIRE(scylla_blas::scylla_queue::queue_exists(session, 1337));
    auto queue = scylla_blas::scylla_queue(session, 1337);
    test_queue_simple(queue);
    test_queue_response(queue);
    test_queue_batch(queue);
}

BOOST_AUTO_TEST_CASE(scylla_queue_sp_sc)
{
    scylla_blas::scylla_queue::delete_queue(session, 1337);
    scylla_blas::scylla_queue::create_queue(session, 1337, false, false);
    BOOST_REQUIRE(scylla_blas::scylla_queue::queue_exists(session, 1337));
    auto queue = scylla_blas::scylla_queue(session, 1337);
    test_queue_simple(queue);
    test_queue_response(queue);
    test_queue_batch(queue);
}

BOOST_AUTO_TEST_CASE(scylla_queue_mp_mc)
{
    scylla_blas::scylla_queue::delete_queue(session, 1337);
    scylla_blas::scylla_queue::create_queue(session, 1337, true, true);
    BOOST_REQUIRE(scylla_blas::scylla_queue::queue_exists(session, 1337));
    auto queue = scylla_blas::scylla_queue(session, 1337);
    test_queue_simple(queue);
    test_queue_response(queue);
    test_queue_batch(queue);
}

BOOST_AUTO_TEST_SUITE_END();