#include <iostream>
#include <memory>

class MyClass : public std::enable_shared_from_this<MyClass> {
public:
    std::shared_ptr<MyClass> getShared() {
        return shared_from_this();
    }
};

int main() {
    auto ptr1 = new MyClass();
    std::shared_ptr<MyClass> ptr2 = ptr1->getShared();

    // std::cout << "ptr1 use count: " << ptr1.use_count() << std::endl;
    std::cout << "ptr2 use count: " << ptr2.use_count() << std::endl;

    return 0;
}