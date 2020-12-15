#include "int_math.hh"

int IntMath::floor_div(int a, int b) {
    return (a + b - 1) / b;
}

size_t IntMath::floor_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}