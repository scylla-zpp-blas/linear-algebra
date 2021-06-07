#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include <fmt/format.h>
#include <scmd.hh>

#include "scylla_blas/queue/worker_proc.hh"
#include "scylla_blas/queue/scylla_queue.hh"
#include "scylla_blas/logging/logging.hh"

#include "scylla_blas/config.hh"
#include "scylla_blas/matrix.hh"
#include "scylla_blas/vector.hh"
#include "scylla_blas/structure/matrix_block.hh"

namespace po = boost::program_options;

struct options {
    std::string host{};
    uint16_t port{};
    bool is_worker = false;
    bool is_deinit = false;
    bool is_init = false;
    int64_t worker_sleep_time;
    int64_t worker_retries;
};

template<typename ...T>
void exactly_one_of(const boost::program_options::variables_map & vm,
                         const T &...op)
{
    const std::vector<std::string> args = { op... };
    if (std::count_if(args.begin(), args.end(), [&](const std::string& s){ return vm.count(s); }) != 1)
    {
        throw std::logic_error(std::string("Need exactly one of mutually exclusive options"));
    }
}

void parse_arguments(int ac, char *av[], options &options) {
    po::options_description desc(fmt::format("Usage: {} [--init/--worker] [options]", av[0]));
    po::options_description opt("Options");
    opt.add_options()
            ("help", "Show program help")
            ("init", "Initialize Scylla keyspace and tables")
            ("deinit", "Deinitialize Scylla keyspace and tables")
            ("worker", "Connect to Scylla and process incoming requests")
            ("host,H", po::value<std::string>(&options.host)->required(), "Address on which Scylla can be reached")
            ("port,P", po::value<uint16_t>(&options.port)->default_value(SCYLLA_DEFAULT_PORT), "port number on which Scylla can be reached")
            ("worker_sleep,s", po::value<int64_t>(&options.worker_sleep_time)->default_value(DEFAULT_WORKER_SLEEP_TIME_MICROSECONDS),
                    "Worker sleep time after queue is empty, in microseconds")
            ("worker_retries,r", po::value<int64_t>(&options.worker_retries)->default_value(DEFAULT_MAX_WORKER_RETRIES),
                    "How many time worker should attempt to do a task");
    desc.add(opt);
    try {
        auto parsed = po::command_line_parser(ac, av)
                .options(desc)
                .run();
        po::variables_map vm;
        po::store(parsed, vm);
        if (vm.count("help") || ac == 1) {
            std::cout << desc << "\n";
            std::exit(0);
        }

        exactly_one_of(vm, "init", "deinit","worker");
        if (vm.count("init")) options.is_init = true;
        if (vm.count("deinit")) options.is_deinit = true;
        if (vm.count("worker")) options.is_worker = true;

        po::notify(vm);
    } catch (std::exception &e) {
        LogCritical("error: {}", e.what());
        std::exit(1);
    } catch (...) {
        LogCritical("Exception of unknown type!");
        std::exit(1);
    }
}

void init(const struct options& op) {
    LogInfo("Connecting to {}:{}...", op.host, op.port);
    auto session = std::make_shared<scmd::session>(op.host, std::to_string(op.port));

    LogInfo("Initializing blas namespace...");
    std::string init_namespace = "CREATE KEYSPACE IF NOT EXISTS blas WITH REPLICATION = {"
                                 "  'class' : 'SimpleStrategy',"
                                 "  'replication_factor' : 1"
                                 "};";
    session->execute(init_namespace);

    LogInfo("Initializing meta-queue...");
    scylla_blas::scylla_queue::init_meta(session);

    LogInfo("Initializing matrix database...");
    scylla_blas::basic_matrix::init_meta(session);
    
    LogInfo("Initializing vector database...");
    scylla_blas::basic_vector::init_meta(session);
    scylla_blas::vector<float>::init(session, HELPER_FLOAT_VECTOR_ID, 0);
    scylla_blas::vector<double>::init(session, HELPER_DOUBLE_VECTOR_ID, 0);

    LogInfo("Creating main task queue...");
    scylla_blas::scylla_queue::create_queue(session, DEFAULT_WORKER_QUEUE_ID, false, true);

    LogInfo("Database initialized succesfully!");
}

void deinit(const struct options& op) {
    LogInfo("Connecting to {}:{}...", op.host, op.port);

    auto session = std::make_shared<scmd::session>(op.host, std::to_string(op.port));

    scmd::statement drop_all("DROP KEYSPACE blas;");
    drop_all.set_timeout(0);
    session->execute(drop_all);

    LogInfo("Database deinitialized succesfully!");
}

void worker(const struct options& op) {
    LogInfo("Worker connecting to {}:{}...", op.host, op.port);
    auto session = std::make_shared<scmd::session>(op.host, std::to_string(op.port));

    LogInfo("Accessing default task queue...");
    auto base_queue = scylla_blas::scylla_queue(session, DEFAULT_WORKER_QUEUE_ID);

    LogInfo("Starting worker loop...");
    for (;;) {
        auto opt = base_queue.consume();
        if (!opt.has_value()) {
            LogTrace("No task in queue, sleeping");
            scylla_blas::wait_microseconds(op.worker_sleep_time);
            LogTrace("Sleeping done");
            continue;
        }
        auto [task_id, task_data] = opt.value();
        LogInfo("A new task received! task_id: {}", task_id);

        int64_t attempts;
        for (attempts = 0; attempts <= op.worker_retries; attempts++) {
            scylla_blas::worker::procedure_t& proc = scylla_blas::worker::get_procedure_for_task(task_data);

            /* Keep trying until the task is finished – otherwise it will be lost and never marked as finished */
            /* TODO: scylla_queue.mark_as_failed()? */
            try {
                auto result = proc(session, task_data);

                if (result.has_value()) {
                    /* The procedure has generated a partial result to be returned */
                    result.value().type = scylla_blas::proto::R_SOME; /* We don't really need result types beyond NONE and SOME */
                    base_queue.mark_as_finished(task_id, result.value());
                } else {
                    base_queue.mark_as_finished(task_id);
                }

                break;
            } catch (const std::exception &e) {
                LogWarn("Task {} failed. Reason: {}. Retrying...", task_id, e.what());
            }
        }

        if (attempts <= op.worker_retries) {
            LogInfo("Task {} completed succesfully.", task_id);
        } else {
            LogError("Abandoned task {} due to too many failures.", task_id);
        }
    }
}

/* Use this program once to initialize the database */
int main(int argc, char **argv) {
    struct options op;
    parse_arguments(argc, argv, op);
    if (op.is_init) {
        init(op);
    } else if (op.is_deinit) {
        deinit(op);
    } else if (op.is_worker) {
        worker(op);
    } else {
        // This code should be unreachable
        throw std::logic_error("How did we get here?");
    }

    return 0;
}

