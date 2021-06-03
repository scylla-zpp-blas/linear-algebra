#pragma once

#include <cstdint>

#include <boost/core/demangle.hpp>

namespace scylla_blas {

using index_t = int64_t;
using id_t = int64_t;

template<class T>
static std::string get_type_name() {
    return boost::core::demangle(typeid(T).name());
}

/* ENUMs used in BLAS routines */
enum TRANSPOSE {
    NoTrans = 111, Trans = 112 //, ConjTrans = 113
};

static TRANSPOSE anti_trans(TRANSPOSE trans) {
    return trans == NoTrans ? Trans : NoTrans;
}

enum UPLO {
    Upper = 121, Lower = 122
};
enum DIAG {
    NonUnit = 131, Unit = 132
};
enum SIDE {
    Left = 141, Right = 142
};
}
