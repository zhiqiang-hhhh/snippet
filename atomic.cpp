#include <iostream>
#include <memory>

int main() {
    std::weak_ptr<int> weakPtr;
    std::weak_ptr<int> weakPtr2;    
    {
        auto sharedPtr = std::make_shared<int>(42);
        weakPtr = sharedPtr;
        weakPtr2 = weakPtr;

        std::cout << "Shared count (use_count): " << sharedPtr.use_count() << std::endl; // 输出 1
        std::cout << "Expired: " << std::boolalpha << weakPtr.expired() << std::endl;   // 输出 false
        std::cout << "Expired: " << std::boolalpha << weakPtr2.expired() << std::endl;   // 输出 false
    }
    // 离开作用域，sharedPtr 被销毁
    std::cout << "Expired after shared_ptr destroyed: " << weakPtr.expired() << std::endl; // 输出 true
    std::cout << "Expired after shared_ptr destroyed: " << weakPtr2.expired() << std::endl; // 输出 true
    return 0;
}
