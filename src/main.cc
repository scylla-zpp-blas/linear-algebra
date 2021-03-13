#include <iostream>
#include <string>

#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <fmt/format.h>
#include <scmd.hh>

#include "scylla_blas/queue/scylla_queue.hh"
#include "scylla_blas/config.hh"

struct options {
    std::string host{};
    uint16_t port{};
    bool is_worker = false;
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

        exactly_one_of(vm, "init", "worker");
        if(vm.count("init")) options.is_init = true;
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

    auto session = std::make_shared<scmd::session>(op.host, std::to_string(op.port));
    std::string init_namespace = "CREATE KEYSPACE IF NOT EXISTS blas WITH REPLICATION = {"
                                 "  'class' : 'SimpleStrategy',"
                                 "  'replication_factor' : 1"
                                 "};";
    session->execute(init_namespace);

    scylla_blas::scylla_queue::init_meta(session);

    std::cerr << "Database initialized succesfully!" << std::endl;
}

void worker(const struct options& op) {
    std::cout << "Go to worker loop\n";
}

/* Use this program once to initialize the database */
int main(int argc, char **argv) {
    struct options op;
    parse_arguments(argc, argv, op);
    if(op.is_init) {
        init(op);
    } else if (op.is_worker) {
        worker(op);
    } else {
        // This code should be unreachable
        throw std::logic_error("How did we get here?");
    }

    return 0;
}

