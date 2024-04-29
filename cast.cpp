#include <cstdint>
#include <iostream>
template<typename T>
class C {
public:
    explicit C(int32_t v) : value(v) {}
    T value;
};

int main() {
    std::int8_t i8 = 1;

    C<int32_t> c(i8);
}