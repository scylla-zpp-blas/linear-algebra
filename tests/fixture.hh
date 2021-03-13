#pragma once

#include <scmd.hh>

#include "config.hh"

class scylla_fixture {
public:
    std::shared_ptr<scmd::session> session;

    scylla_fixture() : session(nullptr) {
        global_config::init();
        connect();
    }

    ~scylla_fixture() {
    }

    void connect(std::string ip = global_config::scylla_ip, std::string port = global_config::scylla_port) {
        this->session = std::make_shared<scmd::session>(ip, port);
    }
};
