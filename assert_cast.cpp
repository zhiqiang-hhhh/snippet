#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <memory>
#include <string>
#include <cstdlib>
#include <cxxabi.h>

// abi::__cxa_demangle returns a C string of known size that should be deleted
// with free().
struct FreeingDeleter
{
    template <typename PointerType>
    void operator() (PointerType ptr)
    {
        std::free(ptr);
    }
};

using DemangleResult = std::unique_ptr<char, FreeingDeleter>;

static DemangleResult tryDemangle(const char * name, int & status)
{
    return DemangleResult(abi::__cxa_demangle(name, nullptr, nullptr, &status));
}

DemangleResult tryDemangle(const char * name)
{
    int status = 0;
    return tryDemangle(name, status);
}


std::string demangle(const char * name, int & status)
{
    auto result = tryDemangle(name, status);
    if (result)
    {
        return std::string(result.get());
    }

    return name;
}


/** Demangles C++ symbol name.
  * When demangling fails, returns the original name and sets status to non-zero.
  * TODO: Write msvc version (now returns the same string)
  */
std::string demangle(const char * name, int & status);

inline std::string demangle(const char * name)
{
    int status = 0;
    return demangle(name, status);
}


template <typename T>
void printType(T value)
{
    if constexpr (std::is_same_v<T, int>)
    {
        std::cout << "Type is int" << std::endl;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        std::cout << "Type is float" << std::endl;
    }
    else
    {
        std::cout << "Unknown type" << std::endl;
    }
}

template <typename To, typename From>
inline To assert_cast(From && from)
{
    try
    {
        if constexpr (std::is_pointer_v<To>)
        {
            std::cout << "From type: " << typeid(*from).name() << " to type: " << typeid(std::remove_pointer_t<To>).name() << std::endl;
            if (typeid(*from) == typeid(std::remove_pointer_t<To>))
                return static_cast<To>(from);
        }
        else
        {
            if (typeid(from) == typeid(To))
                return static_cast<To>(from);
        }
    }
    catch (const std::exception & e)
    {
        throw std::runtime_error("assert cast timeout");
    }
    std::stringstream errMsg;
    errMsg << "Bad cast from type" << demangle(typeid(from).name()) << " to " << demangle(typeid(To).name());
    throw std::runtime_error(errMsg.str());
}

class Base
{
public:
    virtual ~Base() {}
};

class Derived : public Base
{
public:
    void foo() { std::cout << "Derived::foo()" << std::endl; }
};

int main()
{
    printType(10);       // Output: Type is int
    printType(3.14f);    // Output: Type is float
    printType("Hello");  // Output: Unknown type

    Base *base = new Derived();
    try
    {
        Derived *derived = assert_cast<Derived *>(base);
        derived->foo();  // Output: Derived::foo()
    }
    catch (const std::exception & e)
    {
        std::cerr << e.what() << std::endl;
    }

    Base* base2 = new Base();
    try
    {
        Derived *derived = assert_cast<Derived *>(base2);
        derived->foo();
    }
    catch (const std::exception & e)
    {
        std::cerr << e.what() << std::endl;
    }
    
    std::cout << "Type of base: " << typeid(base).name() << std::endl;
    std::cout << "Type of derived: " << typeid(Derived).name() << std::endl;

    return 0;
}