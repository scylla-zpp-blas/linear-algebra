#include <cstdlib>
#include <chrono>
#include <list>
#include <mutex>
#include <condition_variable>

#include <fmt/format.h>

#include <scmd.hh>

constexpr int64_t BATCH_SIZE = 512;
constexpr int64_t MAX_QUEUE_SIZE = 10000;
constexpr int64_t MAX_BATCH_QUEUE_SIZE = 10000;

template<typename F, typename... Args>
double measure_time(F callable, Args... args) {
    auto t1 = std::chrono::high_resolution_clock::now();
    callable(args...);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = t2-t1;
    return duration.count();
}

void benchmark_sync(const std::shared_ptr<scmd::session> &session, int n) {
    auto prepared = session->prepare("INSERT INTO insert_benchmark.test_sync(a, b, c, d, e) VALUES (?, ?, ?, ?, ?);");
    srand(0x1337);
    double result = measure_time([&](){
        int64_t pkey = 0;
        for (int i = 0; i < n; i++) {
            session->execute(prepared, pkey, (int64_t)rand(), (int64_t)rand(), (int64_t)rand(), (int64_t)rand());
            if ((i+1) % BATCH_SIZE == 0) {
                if (pkey % 32 == 0) {
                    //fmt::print("Finished batch {} of {}\n", pkey, 1 + ((n - 1) / BATCH_SIZE));
                }
                pkey++;
            }
        }
    });
    fmt::print("Sync test: {}ms\n", result);
    double seconds = result / 1000;
    double inserts_per_sec = (double)n / seconds;
    fmt::print("Average speed {} inserts/s\n", inserts_per_sec);
}

void benchmark_batch(const std::shared_ptr<scmd::session> &session, int n) {
    auto prepared = session->prepare("INSERT INTO insert_benchmark.test_batch(a, b, c, d, e) VALUES (?, ?, ?, ?, ?);");
    srand(0x1337);
    double result = measure_time([&](){
        int rest = n;
        int64_t pkey = 0;
        while (rest) {
            int batch_size = std::min(rest, (int)BATCH_SIZE);
            scmd::batch_query batch(CASS_BATCH_TYPE_UNLOGGED);
            for(int i = 0; i < batch_size; i++) {
                auto stmt = prepared.get_statement();
                stmt.bind(pkey, (int64_t)rand(), (int64_t)rand(), (int64_t)rand(), (int64_t)rand());
            }
            session->execute(batch);
            if (pkey % 32 == 0) {
                //fmt::print("Finished batch {} of {}\n", pkey, 1 + ((n - 1) / BATCH_SIZE));
            }
            pkey++;
            rest -= batch_size;
        }
    });
    fmt::print("Batch test: {}ms\n", result);
    double seconds = result / 1000;
    double inserts_per_sec = (double)n / seconds;
    fmt::print("Average speed {} inserts/s\n", inserts_per_sec);
}

void benchmark_batch_unprepared(const std::shared_ptr<scmd::session> &session, int n) {
    std::string insert_query("INSERT INTO insert_benchmark.test_batch(a, b, c, d, e) VALUES ({}, {}, {}, {}, {});");
    srand(0x1337);
    double result = measure_time([&](){
        int rest = n;
        int64_t pkey = 0;
        while (rest) {
            int batch_size = std::min(rest, (int)BATCH_SIZE);
            std::string batch = "BEGIN BATCH ";
            for(int i = 0; i < batch_size; i++) {
                batch += fmt::format(insert_query, pkey, (int64_t)rand(), (int64_t)rand(), (int64_t)rand(), (int64_t)rand());
            }
            batch += "APPLY BATCH;";
            session->execute(batch);
            if (pkey % 32 == 0) {
                //fmt::print("Finished batch {} of {}\n", pkey, 1 + ((n - 1) / BATCH_SIZE));
            }
            pkey++;
            rest -= batch_size;
        }
    });
    fmt::print("Batch unprepared test: {}ms\n", result);
    double seconds = result / 1000;
    double inserts_per_sec = (double)n / seconds;
    fmt::print("Average speed {} inserts/s\n", inserts_per_sec);
}

void benchmark_async(const std::shared_ptr<scmd::session> &session, int n) {
    auto prepared = session->prepare("INSERT INTO insert_benchmark.test_sync(a, b, c, d, e) VALUES (?, ?, ?, ?, ?);");
    srand(0x1337);
    double result = measure_time([&](){
        std::mutex m;
        std::condition_variable cv;
        std::condition_variable cv_free;
        std::list<scmd::future> list;

        int64_t pkey = 0;
        for (int i = 0; i < n; i++) {
            if (list.size() >= MAX_QUEUE_SIZE) {
                std::unique_lock<std::mutex> lk(m);
                cv_free.wait(lk, [&]{return list.size() < MAX_QUEUE_SIZE;});
            }
            auto future = session->execute_async(prepared, pkey, (int64_t)rand(), (int64_t)rand(), (int64_t)rand(), (int64_t)rand());
            {
                std::unique_lock<std::mutex> lk(m);
                auto it = list.insert(list.end(), std::move(future));
                lk.unlock();
                it->set_callback([&, it](scmd::future *f){
                    {
                        std::unique_lock<std::mutex> lk2(m);
                        list.erase(it);
                    }
                    if (list.size() < MAX_QUEUE_SIZE) {
                        cv_free.notify_all();
                        if (list.empty()) {
                            cv.notify_one();
                        }
                    }
                });
            }
            if ((i+1) % BATCH_SIZE == 0) {
                if (pkey % 32 == 0) {
                    //fmt::print("Finished batch {} of {}\n", pkey, 1 + ((n - 1) / BATCH_SIZE));
                }
                pkey++;
            }
        }

        {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&]{return list.empty();});
        }
    });
    fmt::print("Async test: {}ms\n", result);
    double seconds = result / 1000;
    double inserts_per_sec = (double)n / seconds;
    fmt::print("Average speed {} inserts/s\n", inserts_per_sec);
}

void benchmark_batch_async(const std::shared_ptr<scmd::session> &session, int n) {
    auto prepared = session->prepare("INSERT INTO insert_benchmark.test_sync(a, b, c, d, e) VALUES (?, ?, ?, ?, ?);");
    srand(0x1337);
    double result = measure_time([&](){
        std::mutex m;
        std::condition_variable cv;
        std::condition_variable cv_free;
        std::list<scmd::future> list;

        int rest = n;
        int64_t pkey = 0;
        while (rest) {
            if (list.size() >= MAX_BATCH_QUEUE_SIZE) {
                std::unique_lock<std::mutex> lk(m);
                cv_free.wait(lk, [&]{return list.size() < MAX_BATCH_QUEUE_SIZE;});
            }
            int batch_size = std::min(rest, (int)BATCH_SIZE);
            scmd::batch_query batch(CASS_BATCH_TYPE_UNLOGGED);
            for(int i = 0; i < batch_size; i++) {
                auto stmt = prepared.get_statement();
                stmt.bind(pkey, (int64_t)rand(), (int64_t)rand(), (int64_t)rand(), (int64_t)rand());
            }
            auto future = session->execute_async(batch);
            std::unique_lock<std::mutex> lk(m);
            auto it = list.insert(list.end(), std::move(future));
            lk.unlock();
            it->set_callback([&, it](scmd::future *f){
                {
                    std::unique_lock<std::mutex> lk2(m);
                    list.erase(it);
                }
                if (list.size() < MAX_BATCH_QUEUE_SIZE) {
                    cv_free.notify_all();
                    if (list.empty()) {
                        cv.notify_one();
                    }
                }
            });
            if (pkey % 32 == 0) {
                //fmt::print("Finished batch {} of {}\n", pkey, 1 + ((n - 1) / BATCH_SIZE));
            }
            pkey++;
            rest -= batch_size;
        }

        {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&]{return list.empty();});
        }
    });
    fmt::print("Async test: {}ms\n", result);
    double seconds = result / 1000;
    double inserts_per_sec = (double)n / seconds;
    fmt::print("Average speed {} inserts/s\n", inserts_per_sec);
}


int main(int argc, char *argv[]) {
    if (argc <= 3) {
        fmt::print("Usage: " + std::string(argv[0]) + " scylla_ip scylla_port n");
        return 0;
    }

    std::string scylla_ip = argv[1];
    std::string scylla_port = argv[2];
    int n = atoi(argv[3]);

    auto session = std::make_shared<scmd::session>(scylla_ip, scylla_port);

    fmt::print("Preparing keyspace and tables\n");

    std::string init_namespace = "CREATE KEYSPACE IF NOT EXISTS insert_benchmark WITH REPLICATION = {"
                                 "  'class' : 'SimpleStrategy',"
                                 "  'replication_factor' : 1"
                                 "};";
    session->execute(init_namespace);

    scmd::statement create_sync_table(R"(CREATE TABLE IF NOT EXISTS insert_benchmark.test_sync (
                                            a BIGINT,
                                            b BIGINT,
                                            c BIGINT,
                                            d BIGINT,
                                            e BIGINT,
                                            PRIMARY KEY (a, c, d)
                                        ))");
    session->execute(create_sync_table);

    scmd::statement create_async_table(R"(CREATE TABLE IF NOT EXISTS insert_benchmark.test_async (
                                            a BIGINT,
                                            b BIGINT,
                                            c BIGINT,
                                            d BIGINT,
                                            e BIGINT,
                                            PRIMARY KEY (a, c, d)
                                        ))");
    session->execute(create_async_table);

    scmd::statement create_batch_table(R"(CREATE TABLE IF NOT EXISTS insert_benchmark.test_batch (
                                            a BIGINT,
                                            b BIGINT,
                                            c BIGINT,
                                            d BIGINT,
                                            e BIGINT,
                                            PRIMARY KEY (a, c, d)
                                        ))");
    session->execute(create_batch_table);

    scmd::statement create_batch_table_unp(R"(CREATE TABLE IF NOT EXISTS insert_benchmark.test_batch_unp (
                                            a BIGINT,
                                            b BIGINT,
                                            c BIGINT,
                                            d BIGINT,
                                            e BIGINT,
                                            PRIMARY KEY (a, c, d)
                                        ))");
    session->execute(create_batch_table_unp);


//    fmt::print("Benchmark sync\n");
//    benchmark_sync(session, n);
//    fmt::print("Benchmark batch\n");
//    benchmark_batch(session, n);
//    fmt::print("Benchmark batch unprepared\n");
//    benchmark_batch_unprepared(session, n);
//    fmt::print("Benchmark async\n");
//    benchmark_async(session, n);
    fmt::print("Benchmark batch async\n");
    benchmark_batch_async(session, n);

    std::string deinit_namespace = "DROP KEYSPACE insert_benchmark;";
    session->execute(deinit_namespace);

}
