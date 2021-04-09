#pragma once

#include "scylla_blas/structure/vector_segment.hh"
#include "scylla_blas/structure/vector_value.hh"
#include "scylla_blas/utils/scylla_types.hh"

namespace scylla_blas {

template<class T>
class vector {

    int64_t _id;
    std::shared_ptr<scmd::session> _session;

public:

    T get_value(index_type x);

    vector_segment<T> get_segment(index_type x);
    vector_segment<T> get_whole(index_type x);

    void update_value(index_type x, T value);
    void update_value(std::vector<vector_value<T>> values);

    void update_segment(index_type x, vector_segment<T> segment_data);
};

}