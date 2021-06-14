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
/**
 *
 * @tparam T - float or double.
 * @param vec - Vector in Scylla we compare.
 * @param dense_values - Dense vector of values to compare with.
 * @return Returns scylla_blas::vector_value<T> with position where values were different and value from vec.
 */
template<class T>
inline std::optional<scylla_blas::vector_value<T>> cmp_vector(const scylla_blas::vector<T> &vec, std::vector<T> dense_values) {
    scylla_blas::vector_segment<T> whole = vec.get_whole();

    if (dense_values.size() < whole.back().index) {
        throw std::runtime_error(fmt::format("Second argument must NOT be a sparse vector.\n"
                                             "Its length ({}) should match the index of "
                                             "the last value in the vector pointed by first argument ({}).\n"
                                             "Fill second argument with zeroes.", dense_values.size(), whole.back().index));
    }

    int last = -1;
    for (scylla_blas::vector_value entry : whole) {
        if (last + 1 != entry.index - 1) {
            auto opt = assert_zeros(dense_values, last + 1, entry.index - 1);
            if (opt.has_value()) {
                return opt;
            }
        }
        if (std::abs(entry.value - dense_values[entry.index - 1]) > scylla_blas::epsilon) {
            return std::optional(entry);
        }
        last = entry.index - 1;
    }
    return assert_zeros(dense_values, last + 1, dense_values.size() - 1);
}
