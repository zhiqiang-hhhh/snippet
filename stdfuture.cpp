#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
 
std::mutex m;
 
struct X
{
    void foo(int i, const std::string& str)
    {
        std::lock_guard<std::mutex> lk(m);
        std::cout << str << ' ' << i << '\n';
    }
 
    std::string bar(const std::string& str)
    {
        std::lock_guard<std::mutex> lk(m);
        // make thread sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << str << '\n';
        return str;
    }
 
    int operator()(int i)
    {
        std::lock_guard<std::mutex> lk(m);
        std::cout << i << '\n';
        return i + 10;
    }
};
 
template<typename RandomIt>
int parallel_sum(RandomIt beg, RandomIt end)
{
    auto len = end - beg;
    if (len < 1000)
        return std::accumulate(beg, end, 0);
 
    RandomIt mid = beg + len / 2;
    auto handle = std::async(std::launch::async,
                             parallel_sum<RandomIt>, mid, end);
    int sum = parallel_sum(beg, mid);
    return sum + handle.get();
}
 
int main()
{
    std::vector<int> v(10000, 1);
    std::cout << "The sum is " << parallel_sum(v.begin(), v.end()) << '\n';
 
    X x;
    // Calls (&x)->foo(42, "Hello") with default policy:
    // may print "Hello 42" concurrently or defer execution
    // auto a1 = std::async(&X::foo, &x, 42, "Hello");
    // std::cout << "after a1 created" << '\n';
    // Calls x.bar("world!") with deferred policy
    // prints "world!" when a2.get() or a2.wait() is called
    auto a2 = std::async(std::launch::deferred, &X::bar, x, "world!");
    std::cout << "after a2 created" << '\n';
    // Calls X()(43); with async policy
    // prints "43" concurrently
    // auto a3 = std::async(std::launch::async, X(), 43);
    std::cout << "after a3 created" << '\n';
    // a2.wait();                     // prints "world!"
    auto res = a2.wait_for(std::chrono::seconds(10)); // prints "world!"
    if (res == std::future_status::ready)
        std::cout << "ready!" << '\n';
    else
        std::cout << "not ready!" << '\n';
    std::cout << "after a2 wait" << '\n';
    std::cout << a2.get();                      // if deferred, prints "world!" here

    // std::cout << a3.get() << '\n'; // prints "53"
    // std::cout << "after a3 get" << '\n';
} // if a1 is not done at this point, destructor of a1 prints "Hello 42" here