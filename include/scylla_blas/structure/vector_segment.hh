#pragma once

#include <vector>
#include <utility>
#include <algorithm>

#include "scylla_blas/structure/vector_value.hh"
#include "scylla_blas/utils/scylla_types.hh"

namespace scylla_blas {

template<class T>
class vector_segment : public std::vector<scylla_blas::vector_value<T>> { // maybe inherit from std::unordered_map instead?
public:
    explicit vector_segment (const std::vector<T>& other) {
        for (int i = 1; i < other.size(); i++) {
            if (other[i] != 0) {
                this->emplace_back(i, other[i]);
            }
        }
    }

    template<typename... U>
    explicit vector_segment (U... args) : std::vector<scylla_blas::vector_value<T>>(args...) { }

    vector_segment operator+=(const vector_segment &other) {
        size_t initial_size = this->size();

        std::vector<scylla_blas::vector_value<T>> skipped_vals;
        auto it_1 = this->begin();
        auto it_2 = other.begin();

        /* TODO: I guess there's a CPP-native function
         * that we could use to do this in a cleaner way
         */
        while(it_1 != this->end() && it_2 != other.end()) {
            if (it_1->index < it_2->index) {
                it_1++;
            } else if (it_1->index > it_2->index) {
                skipped_vals.push_back(*it_2);
                it_2++;
            } else {
                it_1->value += it_2->value;
                it_1++;
                it_2++;
            }
        }

        std::copy(skipped_vals.begin(), skipped_vals.end(), std::back_inserter(*this));

        /* Semantics is too specific here to overload the comparison operator. */
        std::inplace_merge(this->begin(), this->begin() + initial_size, this->end(),
                           [](const auto &a, const auto &b) {
                               return a.index < b.index;
                           });

        /* Doing this part only after merge saves a little time */
        std::copy(it_2, other.end(), std::back_inserter(*this));

        return *this;
    }

    const vector_segment operator+(const vector_segment &other) const {
        vector_segment<T> result = *this;

        result += other;
        return result;
    }

    vector_segment operator*=(T factor) {
        for (auto &entry : *this) {
            entry.value *= factor;
        }

        return *this;
    }

    const vector_segment operator*(T factor) const {
        vector_segment result = *this; // Our vectors are shallow, so no need for deep copies; or should we take special care of that?

        result *= factor;
        return result;
    }

    const T prod(const vector_segment &other) {
        T ret = 0;

        auto it_1 = this->begin();
        auto it_2 = other.begin();

        while(it_1 != this->end() && it_2 != other.end()) {
            if (it_1->index < it_2->index) {
                it_1++;
            } else if (it_1->index > it_2->index) {
                it_2++;
            } else {
                ret += it_1->value * it_2->value;
                it_1++;
                it_2++;
            }
        }

        return ret;
    }
};

}