#pragma once

#include <cstdarg>
#include <vector>
#include <utility>

#include "../scylla_types.hh"

namespace scylla_blas {

template<class T>
class vector_value {
public:
    index_type index;
    T value;

    constexpr vector_value(const index_type& index, const T& value) : index(index), value(value) { }
    constexpr vector_value(const std::pair<index_type, T>& p) : index(p.first), value(p.second) { }
};

template<class T>
class vector : public std::vector<scylla_blas::vector_value<T>> { // maybe inherit from std::unordered_map instead?
public:
    vector (const std::vector<T>& other) {
        for (int i = 1; i < other.size(); i++) {
            if (other[i] != 0) {
                this->emplace_back(i, other[i]);
            }
        }
    }

    template<typename... U>
    vector (U... args) : std::vector<scylla_blas::vector_value<T>>(args...) { }

    vector operator+=(const vector &other) {
        *this = (*this + other); // Not really optimal, but shouldn't be a huge problem
        return *this;
    }

    const vector operator+(const vector &other) const {
        vector<T> result;

        auto it_1 = this->begin();
        auto it_2 = other.begin();

        while(it_1 != this->end() && it_2 != other.end()) {
            if (it_1->index < it_2->index) {
                result.emplace_back(*it_1);
                it_1++;
            } else if (it_1->index > it_2->index) {
                result.emplace_back(*it_2);
                it_2++;
            } else {
                result.emplace_back(
                    it_1->index, // == it_2->index
                    it_1->value + it_2->value
                );
                it_1++;
                it_2++;
            }
        }

        // Depending on the case, ne of these will take care of the remainder of the copy, the other will do nothing
        std::copy(it_1, this->end(), std::back_inserter(result));
        std::copy(it_2, other.end(), std::back_inserter(result));

        return result;
    }

    vector operator*=(T factor) {
        for (auto &entry : *this) {
            entry.value *= factor;
        }

        return *this;
    }

    const vector operator*(T factor) const {
        vector result = *this; // Our vectors are shallow, so no need for deep copies; or should we take special care of that?

        result *= factor;
        return result;
    }
};

}