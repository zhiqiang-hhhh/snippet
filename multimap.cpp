#include <string>
#include <iostream>
#include <unordered_map>

using namespace std;

int main() {
    std::unordered_multimap<std::string, std::string> multi_map {
        {"A", "1"},
        {"B", "2"},
        {"B", "1"},
    };

    auto itr = multi_map.find("A");
    std::cout << itr->second << std::endl;

    itr = multi_map.find("B");
    std::cout << itr->second << std::endl;

    for (auto itr : multi_map) {
        std::cout << itr.first << ' ' << itr.second << std::endl;
    }
    std::cout << "====\n";
    auto eq_range = multi_map.equal_range("B");
    for (auto it = eq_range.first; it != eq_range.second; ++it) {
        // B : 2
        // B : 1
        std::cout << it->first << ' ' << it->second << std::endl;
    }
    std::cout << eq_range.first->first << ' ' << eq_range.first->second << std::endl;
    std::cout << eq_range.second->first << ' ' << eq_range.second->second << std::endl;

    while (eq_range.first != eq_range.second) {
        std::string v = eq_range.first->second;
        if (v == "2") {
            multi_map.erase(eq_range.first);
            break;
        }
        ++eq_range.first;
    }
    std::cout << "After erase\n";

    eq_range = multi_map.equal_range("B");
    for (auto it = eq_range.first; it != eq_range.second; ++it) {
        // B : 1
        std::cout << it->first << ' ' << it->second << std::endl;
    }
}