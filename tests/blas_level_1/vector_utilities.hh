#pragma once

template<class T>
void print_vector(const scylla_blas::vector<T>& vec) {
    auto whole = vec.get_whole();

    std::cerr << "Vector " << vec.id << ": ";
    for (auto entry : whole) {
        std::cerr << "(" << entry.index << "-> " << entry.value << "), ";
    }
    std::cerr << std::endl;
}

template <class T>
inline std::optional<scylla_blas::vector_value<T>> assert_zeros(std::vector<T> &values, int start, int end) {
    for (int i = start; i < end; i++) {
        if (abs(values[i]) > scylla_blas::epsilon) {
            return std::make_optional<scylla_blas::vector_value<T>>(i + 1, 0);
        }
    }
    return std::nullopt;
}

// Returns value of scylla_blas::vector that was different.
template<class T>
inline std::optional<scylla_blas::vector_value<T>> cmp_vector(const scylla_blas::vector<T> &vec, std::vector<T> values) {
    scylla_blas::vector_segment<T> whole = vec.get_whole();

    if (values.size() < vec.length)

    int last = -1;
    for (scylla_blas::vector_value entry : whole) {
        if (last + 1 != entry.index - 1) {
            auto opt = assert_zeros(values, last + 1, entry.index - 1);
            if (opt.has_value()) {
                return opt;
            }
        }
        if (abs(entry.value - values[entry.index - 1]) > scylla_blas::epsilon) {
            return std::optional(entry);
        }
        last = entry.index - 1;
    }
    return assert_zeros(values, last + 1, values.size() - 1);
}
