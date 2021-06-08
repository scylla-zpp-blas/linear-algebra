#include <iostream>
#include <string>
#include <chrono>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
#include <fmt/format.h>
#include <scmd.hh>

#include "scylla_blas/config.hh"
#include "benchmark.hh"

struct options {
    std::string host{};
    uint16_t port{};
    bool mmmul = false;
    bool mvmul = false;
    bool vvmul = false;
    std::vector<int64_t> block_sizes;
    std::vector<int64_t> problem_sizes;
    int workers;
    double matrix_load;
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
    po::options_description desc(fmt::format("Usage: {} [--mmmul/--mvmul/--vvmul] [options]", av[0]));
    po::options_description opt("Options");
    opt.add_options()
            ("help", "Show program help")
            ("mmmul", "Benchmark matrix * matrix multiplication")
            ("mvmul", "Benchmark matrix * vector multiplication")
            ("vvmul", "Benchmark dot product of 2 vectors")
            ("block_sizes", po::value<std::vector<int64_t>>(&options.block_sizes)->required()->multitoken(), "Block sizes to benchmark")
            ("problem_sizes", po::value<std::vector<int64_t>>(&options.problem_sizes)->required()->multitoken(), "Problem sizes to benchmark (vector length / matrix side length)")
            ("workers", po::value<int>(&options.workers)->required())
            ("matrix_load", po::value<double>(&options.matrix_load)->default_value(0.2), "% of non-zerio matrix element")
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

        exactly_one_of(vm, "mmmul", "mvmul","vvmul");
        if (vm.count("mmmul")) options.mmmul = true;
        if (vm.count("mvmul")) options.mvmul = true;
        if (vm.count("vvmul")) options.vvmul = true;

        po::notify(vm);
    } catch (std::exception &e) {
        LogCritical("error: {}", e.what());
        std::exit(1);
    } catch (...) {
        LogCritical("Exception of unknown type!");
        std::exit(1);
    }
}


/* Use this program once to initialize the database */
int main(int argc, char **argv) {
    auto start_time = std::chrono::system_clock::now();
    struct options op;
    parse_arguments(argc, argv, op);
    auto session = std::make_shared<scmd::session>(op.host, std::to_string(op.port));
    std::unique_ptr<base_benchmark> tester;
    std::string test_name;
    if (op.mmmul) {
        tester = std::make_unique<benchmark_mm>(session);
        test_name = "mmmul";
    } else if (op.mvmul) {
        tester = std::make_unique<benchmark_mv>(session);
        test_name ="mvmul";
    } else if (op.vvmul) {
        tester = std::make_unique<benchmark_vv>(session);
        test_name = "vvmul";
    }
    tester->set_max_workers(op.workers);
    tester->set_matrix_load(op.matrix_load);
    benchmark_result result = perform_benchmark(std::move(tester), op.block_sizes, op.problem_sizes);
    LogInfo("Benchmark results (type={}, workers={})", test_name, op.workers);
    for (auto &[bs, ps, r] : result.tests) {
        std::cout << bs << " " << ps << " " << r.setup_time << " " << r.proc_time << " " << r.teardown_time << "\n";
    }
    auto end_time = std::chrono::system_clock::now();
    std::time_t start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::time_t end_time_t = std::chrono::system_clock::to_time_t(end_time);
    std::cout << "Start time: " << std::ctime(&start_time_t);
    std::cout << "End time: " << std::ctime(&end_time_t);
    return 0;
}


