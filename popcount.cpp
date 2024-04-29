#include <bit>
#include <bitset>
#include <cstdint>
#include <iostream>

int main () {
    // std::cout <<  std::popcount(static_cast<uint32_t>(-10)) << std::endl;

    int64_t x = 0xEFFFFFEFFFFF;

    std::cout << std::bitset<64>(x) << std::endl;

    int64_t i64 = 4278124283;
    // 4294967295U

    std::cout << std::bitset<64>(i64) << std::endl;
    UINT32_MAX
}