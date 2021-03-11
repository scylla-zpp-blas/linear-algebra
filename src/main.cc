#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <iostream>
#include <string>

#include <session.hh>

struct options {
    std::string host{};
    uint16_t port{};
};


void parse_arguments(int ac, char *av[], options *options) {
    try {
        po::options_description desc("Usage");
        desc.add_options()
            ("help", "Show program help")
            ("host,H", po::value<std::string>(&options->host)->required(), "Address on which Scylla can be reached")
            ("port,P", po::value<uint16_t>(&options->port)->required(), "port number on which Scylla can be reached");

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << "\n";
            std::exit(0);
        }
        po::notify(vm);
    } catch (std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        std::exit(1);
    } catch (...) {
        std::cerr << "Exception of unknown type!\n";
        std::exit(1);
    }
}

/* Use this program once to initialize the database */
int main(int argc, char **argv) {
    struct options op;
    parse_arguments(argc, argv, &op);

    if (argc > 3) {
        std::cout << "Usage: " << argv[0] << " [IP address] [port]" << std::endl;
        exit(0);
    }

    std::string ip_address = argc > 1 ? argv[1] : "172.17.0.2"; // docker address = default
    std::string port = argc > 2 ? argv[2] : "9042";

    std::cerr << "Connecting to " << ip_address << ":" << port << "..." << std::endl;

    scmd::session session{ip_address, port};
    std::string init_namespace = "CREATE KEYSPACE IF NOT EXISTS blas WITH REPLICATION = {"
                                 "  'class' : 'SimpleStrategy',"
                                 "  'replication_factor' : 1"
                                 "};";
    session.execute(init_namespace);


    std::string init_sets = "CREATE TABLE IF NOT EXISTS blas.item_set_meta ( "
                            "   id bigint PRIMARY KEY, "
                            "   cnt COUNTER "
                            ");";
    session.execute(init_sets);

    std::cerr << "Database initialized succesfully!" << std::endl;
    return 0;
}

