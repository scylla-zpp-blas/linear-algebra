#pragma once

#include <cstdint>

namespace scylla_blas {

using index_t = int64_t;
using id_t = int64_t;

template<class T>
static std::string get_type_name();

template<>
std::string get_type_name<double>() { return "double"; }

template<>
std::string get_type_name<float>() { return "float"; }


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
