#include <exception>
#include <iostream> 
#include <memory>

class BlockThrow {
public:
    BlockThrow() noexcept {
        throw std::exception();
    }
};

int main(void) {
  try {
    auto s_ptr = std::make_shared<BlockThrow>();
  } catch(...) {
    std::cout << "BlockThrow Found throw, " << std::endl;
  }
}