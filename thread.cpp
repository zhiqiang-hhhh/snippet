#include <cstdint>
#include <iostream>
#include <thread>
#include <chrono>

class Base {
public:
    virtual void open() {
        std::cout << "Base open\n";
    }
};

class Drive : public Base {
public:
    void open() override {
        std::cout << "Drive open\n";
    }
};

int main()
{
    Drive d;
    d.open();

    int64_t a = -7629445119491449;

    std::cout << static_cast<int>(a) << std::endl;

    std::cout << INT64_MAX << std::endl;

    return 0;
}

