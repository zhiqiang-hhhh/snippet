#include <bitset>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <sys/types.h>


static inline void reverse_byes(uint8_t * __restrict s, size_t length)
{
    int c, i, j;

    for (i = 0, j = length - 1; i < j; i++, j--)
    {
        c = s[i];
        s[i] = s[j];
        s[j] = c;
    }
}

int64_t encodeStringToI64(const std::string& str) {
    if (str.size() > 7) {
        throw std::runtime_error("String is too long");
    }

    int64_t res = 0;
    uint8_t* __restrict ui8_ptr = reinterpret_cast<uint8_t*>(&res);
    uint8_t size = str.size();

    memcpy(ui8_ptr, str.c_str(), size);
    // "reverse" the order of string on little endian machine.
    reverse_byes(ui8_ptr, sizeof(int64_t));
    // Lowest byte of Integer stores the size of the string, bit left shiflted by 1 so that we can get
    // correct size after right shifting by 1
    memset(ui8_ptr, size << 1, 1);
    

    res = res >> 1;
    res &= 0x7FFFFFFFFFFFFFFF;
    return res;
}

// int64_t encodeStringToI64(const std::string& str) {
//     int64_t res;
//     uint8_t* __restrict ui8_ptr = reinterpret_cast<uint8_t*>(&res);
//     uint8_t str_size = static_cast<uint8_t>(str.size());
//     auto str_ptr = str.c_str();

//     // Lowest byte of Integer stores the size of the string, bit left shiflted by 1 so that we can get
//     // correct size after right shifting by 1
//     memset(ui8_ptr, str_size << 1, 1);
    
//     // "reverse" the order of string on little endian machine.
//     for (int j = str_size - 1, k = 1; j >= 0; --j, ++k) {
//         memcpy(ui8_ptr + k, str_ptr + j, 1);
//     }

//     res = (res >> 1);
//     res &= 0x7FFFFFFFFFFFFFFF;
//     return res;
// }


std::string decodeStringFromI64(int64_t val) {
    auto ui8_ptr = reinterpret_cast<uint8_t*>(&val);
    int strSize = *ui8_ptr;
    std::string res;
    res.reserve(strSize);
    val = val << 1;
    for (int i = strSize - 1, j = 0; i >= 0; --i, ++j) {
        res.push_back(*(ui8_ptr + sizeof(val) - 1 - j));
    }
    return res;
}

__int128_t encodeStringToI128(const std::string& str) {
    if (str.size() > 15) {
        throw std::runtime_error("String is too long");
    }

    __int128_t res = 0;
    auto ui8_ptr = reinterpret_cast<uint8_t*>(&res);

    for (size_t i = 0; i < str.size(); ++i) {
        memcpy(ui8_ptr + sizeof(res) - 1 - i, str.c_str() + i, 1);
    }

    uint8_t size = str.size();
    memset(ui8_ptr, size << 1, 1);
    
    res &= std::numeric_limits<__int128_t>::max();;
    return res >> 1;
}

template<typename T>
void printBytes(T val) {
    std::cout << std::endl;
    uint8_t* ui8_ptr = reinterpret_cast<uint8_t*>(&val);
    for (size_t i = 0; i < sizeof(T); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(*(ui8_ptr + i)) << '\t';
    }
    std::cout << std::dec  << std::endl;
    return;
}

int main() {
    const std::string str1 = "12345";
    const std::string str2 = "1234";

    int64_t xEncodeed1 = encodeStringToI64(str1);
    std::cout << str1 << " encodeStringToI64: " << xEncodeed1 << std::endl;
    std::cout << "decodeStringFromI64: " << decodeStringFromI64(xEncodeed1) << std::endl;
    int64_t xEncodeed2 = encodeStringToI64(str2);
    std::cout << str2 << " encodeStringToI64: " << xEncodeed2 << std::endl;
    std::cout << "decodeStringFromI64: " << decodeStringFromI64(xEncodeed2) << std::endl;
    

    std::cout << (str1 > str2) << std::endl;
    std::cout << (xEncodeed1 > xEncodeed2) << std::endl;
}