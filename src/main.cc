#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>

namespace po = boost::program_options;
#include <fmt/format.h>
#include <scmd.hh>

#include "scylla_blas/queue/scylla_queue.hh"
#include "scylla_blas/config.hh"
#include "scylla_blas/matrix.hh"
#include "scylla_blas/structure/matrix_block.hh"

struct options {
    std::string host{};
    uint16_t port{};
    bool is_worker = false;
    bool is_deinit = false;
    bool is_init = false;
};

template<typename ...T>
void exactly_one_of(const boost::program_options::variables_map & vm,
                         const T &...op)
{
    const std::vector<std::string> args = { op... };
    if(std::count_if(args.begin(), args.end(), [&](const std::string& s){ return vm.count(s); }) != 1)
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
            ("port,P", po::value<uint16_t>(&options.port)->default_value(SCYLLA_DEFAULT_PORT), "port number on which Scylla can be reached");
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
        if(vm.count("init")) options.is_init = true;
        if(vm.count("deinit")) options.is_deinit = true;
        if(vm.count("worker")) options.is_worker = true;

        po::notify(vm);
    } catch (std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        std::exit(1);
    } catch (...) {
        std::cerr << "Exception of unknown type!\n";
        std::exit(1);
    }
}

void init(const struct options& op) {
    std::cerr << "Connecting to " << op.host << ":" << op.port << "..." << std::endl;

    std::cerr << "Initializing blas namespace..." << std::endl;
    auto session = std::make_shared<scmd::session>(op.host, std::to_string(op.port));
    std::string init_namespace = "CREATE KEYSPACE IF NOT EXISTS blas WITH REPLICATION = {"
                                 "  'class' : 'SimpleStrategy',"
                                 "  'replication_factor' : 1"
                                 "};";
    session->execute(init_namespace);

    std::cerr << "Initializing meta-queue..." << std::endl;
    scylla_blas::scylla_queue::init_meta(session);

    std::cerr << "Creating main task queue..." << std::endl;
    scylla_blas::scylla_queue::create_queue(session, DEFAULT_WORKER_QUEUE_ID, true, true);

    std::cerr << "Database initialized succesfully!" << std::endl;
}

void deinit(const struct options& op) {
    std::cerr << "Connecting to " << op.host << ":" << op.port << "..." << std::endl;

    auto session = std::make_shared<scmd::session>(op.host, std::to_string(op.port));

    scmd::statement drop_all("DROP KEYSPACE blas;");
    drop_all.set_timeout(0);
    session->execute(drop_all);

    std::cerr << "Database deinitialized succesfully!" << std::endl;
}

template<typename T>
void multiply_matrices(const std::shared_ptr<scmd::session> &session, scylla_blas::task task) {
    auto [task_queue_id, A_id, B_id, C_id] = task.multiplication_order;

    scylla_blas::matrix<T> A(session, A_id, false);
    scylla_blas::matrix<T> B(session, B_id, false);
    scylla_blas::matrix<T> C(session, C_id, false);
    auto task_queue = scylla_blas::scylla_queue(session, task_queue_id);
    std::cerr << "Linked to queue " << task_queue_id << std::endl;

    while (true) {
        try {
            auto [task_id, block_computation_task] = task_queue.consume();
            std::cerr << "New secondary task obtained; id = " << task_id << std::endl;

            auto [result_row, result_column] = block_computation_task.compute_block;

            scylla_blas::matrix_block<T> result_block({}, C_id, result_row, result_column);

            for (scylla_blas::index_type i = 1; i <= MATRIX_BLOCK_WIDTH; i++) {
                auto block_A = A.get_block(result_row, i);
                auto block_B = B.get_block(i, result_column);

                result_block += block_A * block_B;
            }

            C.update_block(result_row, result_column, result_block);
        } catch (const scylla_blas::empty_container_error& e) {
            // The task queue is empty – nothing left to do.
           break;
        }
    }
}

void worker(const struct options& op) {
    std::cerr << "Worker connecting to " << op.host << ":" << op.port << "..." << std::endl;
    auto session = std::make_shared<scmd::session>(op.host, std::to_string(op.port));

    std::cerr << "Accessing default task queue..." << std::endl;
    auto base_queue = scylla_blas::scylla_queue(session, DEFAULT_WORKER_QUEUE_ID);

    std::cerr << "Starting worker loop...\n";
    for (;;) {
        try {
            auto [task_id, multiplication_task] = base_queue.consume();
            std::cerr << "A new task received! task_id: " << task_id  << std::endl;

            int64_t attempts;
            for (attempts = 0; attempts <= MAX_WORKER_RETRIES; attempts++) {
                /* Keep trying until the task is finished – otherwise it will be lost and never marked as finished */
                /* TODO: scylla_queue.mark_as_failed()? */
                try {
                    /* TODO: deduce the type from the matrix data in the database
                     * instead of specifying it in advance. Worst-case scenario:
                     * indicate the type with an enum in the task description.
                     */
                    multiply_matrices<float>(session, multiplication_task);
                    base_queue.mark_as_finished(task_id);
                    break;
                } catch (const std::exception &e) {
                    std::cerr << "Task " << task_id << " failed. Reason: " << e.what() << std::endl;
                    std::cerr << "Retrying..." << std::endl;
                }
            }

            if (attempts <= MAX_WORKER_RETRIES) {
                std::cerr << "Task " << task_id << " completed succesfully." << std::endl;
            } else {
                std::cerr << "Abandoned task " << task_id << " due to too many failures." << std::endl;
            }
        } catch (const scylla_blas::empty_container_error& e) {
            std::cerr << "Waiting for a task..." << std::endl;
            boost::this_thread::sleep(boost::posix_time::seconds(WORKER_SLEEP_TIME_SECONDS));
        }
    }
}

/* Use this program once to initialize the database */
int main(int argc, char **argv) {
    struct options op;
    parse_arguments(argc, argv, op);
    if(op.is_init) {
        init(op);
    } else if(op.is_deinit) {
        deinit(op);
    } else if (op.is_worker) {
        worker(op);
    } else {
        // This code should be unreachable
        throw std::logic_error("How did we get here?");
    }

    return 0;
}

