#include <variant>
#include <iostream>

struct PrintVisitor {
    void operator()(int value) const {
        std::cout << "Integer: " << value << std::endl;
    }

    void operator()(double value) const {
        std::cout << "Double: " << value << std::endl;
    }

    void operator()(const std::string& value) const {
        std::cout << "String: " << value << std::endl;
    }
};

int main() {
    std::variant<int, double, std::string> v;

    std::visit(PrintVisitor{}, v); // 执行针对存储的 int 的操作

    v = 3.14;
    std::visit(PrintVisitor{}, v); // 执行针对存储的 double 的操作

    v = "Hello";
    std::visit(PrintVisitor{}, v); // 执行针对存储的 std::string 的操作

    return 0;
}
