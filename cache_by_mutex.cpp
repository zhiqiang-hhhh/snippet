#include <exception>
#include <iostream>
#include <unordered_map>
#include <future>
#include <mutex>
#include <memory>
#include <thread>
#include <chrono>
#include <functional>  // For std::bind

struct ReadOptions {
    std::string thread_name = "";
};

// Cache 类的定义
template <typename Key, typename Value>
class Cache {
public:
    using LoadFunction = std::function<Value(const Key&)>;

    // 获取或设置缓存中的值
    Value get_or_set(const Key& key, LoadFunction load_function) {
        Value result;

        {
            // 锁住缓存，检查是否已经存在
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(key);

            if (it != cache.end()) {
                // 缓存命中，直接返回
                result = it->second;
            } else {
                // 缓存未命中，创建一个 future 并放入缓存
                Value result;
                try {
                   result = load_function(key);
                } catch (std::exception e){
                    std::cout << "Caught exception: " << e.what() << std::endl;
                }
                cache[key] = load_function(key);
            }
        }

        std::cout << "Waiting for key: " << key << std::endl;
        // 等待加载结果
        return result;
    }

private:
    std::unordered_map<Key, Value> cache;
    std::mutex cache_mutex;
};

// 模拟加载函数，需要额外的参数
std::string load_data(const std::string& key, const ReadOptions& extra_param) {
    std::cout << "Loading data for key: " << key << " with extra_param: " << extra_param.thread_name << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));  // 模拟数据加载延迟
    return "Result from " + extra_param.thread_name;
}

std::string load_data_with_exception(const std::string& key, const ReadOptions& extra_param) {
    std::cout << "Loading data for key: " << key << " with extra_param: " << extra_param.thread_name << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));  // 模拟数据加载延迟
    throw std::runtime_error("Failed to load data for key: " + key + " with extra_param: " + extra_param.thread_name);
    return "Result from " + extra_param.thread_name;
}

int main() {
    Cache<std::string, std::string> cache;

    // 线程1，使用 std::bind 传递额外参数
    std::thread t1([&]() {
        std::cout << "Thread 1 reading\n";
        ReadOptions options {
            .thread_name = "Thread 1"
        };
        auto load_function = std::bind(load_data_with_exception, std::placeholders::_1, options);
        auto load_read = cache.get_or_set("key1", load_function);
        std::cout << "Thread 1 result: " << load_read << std::endl;
    });

    // 线程2，使用 lambda 表达式传递额外参数
    std::thread t2([&]() {
        std::cout << "Thread 2 reading\n";
        ReadOptions options {
            .thread_name = "Thread 2"
        };
        auto load_function = std::bind(load_data_with_exception, std::placeholders::_1, options);
        auto load_read = cache.get_or_set("key1", load_function);
        std::cout << "Thread 2 result: " << load_read << std::endl;
    });

    // 线程3，使用不同的参数加载
    std::thread t3([&]() {
        std::cout << "Thread 3 reading\n";
        ReadOptions options {
            .thread_name = "Thread 3"
        };
        auto load_function = std::bind(load_data, std::placeholders::_1, options);
        auto load_read = cache.get_or_set("key2", load_function);
        std::cout << "Thread 3 result: " << load_read << std::endl;
    });

    // 等待所有线程结束
    t1.join();
    t2.join();
    t3.join();

    return 0;
}
