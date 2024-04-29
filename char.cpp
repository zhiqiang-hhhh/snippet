#include <bitset>
#include <cstdint>
#include <cstring>
#include <ios>
#include <iostream>
#include <string>
#include <variant>

int main() {
    std::cout << "This is a backslash: \\" << std::endl;
    std::cout << "This is a double quote: \"" << std::endl;
    std::cout << "This is a single quote: \'" << std::endl;
    std::cout << "This is a newline: \n" << std::endl;
    std::cout << "This is a tab: \t" << std::endl;
    std::cout << "This is a carriage return: \r" << std::endl;
    std::cout << "This is a backspace: \b" << std::endl;


    // std::cout << parse_file_path("\"t\es\t \it") << std::endl;
    std::cout << "\\\\:" << "\t"  << " ]"<< std::endl;
    std::cout << "\\e:" << "\e" <<  " ]" << std::endl;
    std::cout << "\\i:" << "\i" << " ]" << std::endl;

    signed char sc = 'A';
    
    uint8_t c16[7] = u8"你好";
    for (auto& i : c16) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
    std::cout << c16 << std::endl;
    std::wcout << c16 << std::endl;

    std::string str = "格式";
    std::cout << str << ':' << str.size() << std::endl;
    std::cout << "std::string: ";
    for (auto& c : str) {
        std::cout << static_cast<int>(c) << ' ';
    }
    std::cout << std::endl;

    int8_t i8a[7] = "格式";
    std::cout << "int8_t: ";
    for (auto& c : i8a) {
        std::cout << static_cast<int>(c) << ' ';
    }
    std::cout << std::endl;

    uint8_t u8a[7] = "格式";
    std::cout << "uint8_t: ";
    for (auto& c : u8a) {
        std::cout << static_cast<int>(c) << ' ';
    }
    std::cout << std::endl;

    char8_t c8a[7] = u8"格式";
    std::cout << "char8: ";
    for (auto& c : c8a) {
        std::cout << static_cast<int>(c) << ' ';
    }
    std::cout << std::endl;

    char16_t c16a[3] = u"格式";
    std::cout << "char16: ";
    for (auto& c : c16a) {
        std::cout << static_cast<int>(c) << ' ';
    }
    std::cout << std::endl;


    int8_t i8 = -1;
    std::cout << std::bitset<8>(i8) << std::endl;
    std::cout << std::bitset<16>(static_cast<int16_t>(i8)) << std::endl;
    std::cout << std::bitset<sizeof(double)*8>(static_cast<double>(i8)) << ' ' << static_cast<double>(i8)<< std::endl;

    char c1 = 'A';
    unsigned c2 = 'A';
    int8_t c3 = 'A';
    uint8_t c4 = 'A';

    std::variant<char, unsigned char, int8_t> sto;
    sto = 'A';
    std::cout << std::get<char>(sto) << std::endl;
    // std::cout << std::get<unsigned char>(sto) << std::endl;
    // std::cout << std::get<int8_t>(sto) << std::endl;// << ' ' << std::get<uint8_t>(sto);

    // char8_t c81 = 'A';
    // std::cout << c81 << std::endl;
    // c8a = u8"€";
    // char tt = static_cast<char>(19999);
    // std::cout << c8a << std::endl;
    

    return 0;

}