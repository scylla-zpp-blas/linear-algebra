#include <iostream>
#include <string>

#include "session.hh"

/* Use this program once to initialize the database */
int main(int argc, char **argv) {
    if (argc > 2) {
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

    std::cerr << "Database initialized succesfully!" << std::endl;
    return 0;
}

