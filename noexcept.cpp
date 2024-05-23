#include <exception>
#include <iostream>

void throwException() {
    throw std::exception();
}

void methodNoExcept() {
    throwException();
}

void methodMayThrowExcept() {
    std::cout << "methodThrowExcept" << std::endl;
}

int main () {
    try {
        methodMayThrowExcept();
        methodNoExcept();
    } catch (std::exception& e) {
        std::cout << "catch exception from noexcept";
    } catch (...) {
        std::cout << "catch exception from noexcept";
    }
}