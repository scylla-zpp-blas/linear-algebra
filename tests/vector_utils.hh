#pragma once

template<class T>
void print_vector(const scylla_blas::vector<T> &vec) {
    auto default_precision = std::cout.precision();
    auto whole = vec.get_whole();

    std::cout << std::setprecision(4);
    std::cout << "Vector " << vec.get_id() << ": " << std::endl;

    scylla_blas::index_t expected = 1;
    for (auto entry : whole) {
        while (expected < entry.index) {
            /* Show empty rows */
            std::cout << expected << " ->\t" << 0 << std::endl;
            expected++;
        }

        std::cout << entry.index << " ->\t" << entry.value << std::endl;

        expected = entry.index + 1;
    }

    std::cout << std::endl;
    std::cout << std::setprecision(default_precision);
}

template <class T>
inline std::optional<scylla_blas::vector_value<T>> assert_zeros(std::vector<T> &values, int start, int end) {
    for (int i = start; i < end; i++) {
        if (std::abs(values[i]) > scylla_blas::epsilon) {
            return std::make_optional<scylla_blas::vector_value<T>>(i + 1, 0);
        }
    }
    return std::nullopt;
}

// Returns value of scylla_blas::vector that was different.
template<class T>
inline std::optional<scylla_blas::vector_value<T>> cmp_vector(const scylla_blas::vector<T> &vec, std::vector<T> values) {
    scylla_blas::vector_segment<T> whole = vec.get_whole();

    if (values.size() < whole.back().index) {
        throw std::runtime_error("Not enough values in vector. Fill with zeroes.");
    }

    int last = -1;
    for (scylla_blas::vector_value entry : whole) {
        if (last + 1 != entry.index - 1) {
            auto opt = assert_zeros(values, last + 1, entry.index - 1);
            if (opt.has_value()) {
                return opt;
            }
        }
        if (std::abs(entry.value - values[entry.index - 1]) > scylla_blas::epsilon) {
            return std::optional(entry);
        }
        last = entry.index - 1;
    }
    return assert_zeros(values, last + 1, values.size() - 1);
}
