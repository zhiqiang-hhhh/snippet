#include <iostream>
#include <cstdlib>
#include <new> // for std::bad_alloc
#include <thread>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <mutex>

// 定义 10MB 的阈值
const size_t THRESHOLD = 10 * 1024 * 1024;

// 线程局部变量，用于统计每个线程的内存分配
thread_local std::atomic<size_t> threadMemoryAllocation{0};
thread_local bool tagger_initialized = false;

// 全局互斥锁，用于保护全局内存统计数据的线程安全访问
std::mutex globalMtx;

// 全局内存统计数据
std::unordered_map<std::thread::id, size_t> globalMemoryAllocations;

// RAII 类，用于在线程结束时更新全局内存统计数据
class ThreadMemoryTracker {
public:
    ThreadMemoryTracker() {}
    ~ThreadMemoryTracker() {
        std::lock_guard<std::mutex> lock(globalMtx);
        globalMemoryAllocations[std::this_thread::get_id()] = threadMemoryAllocation.load();
    }
};

class MemAllocator {
public:
    // 自定义 malloc 函数
    void* allocate(size_t size) {
        if (!tagger_initialized) {
            throw std::runtime_error("AllocationTagger not initialized");
        }

        void* ptr = std::malloc(size);
        if (!ptr) {
            throw std::bad_alloc();
        }

        // 更新线程局部内存统计数据
        threadMemoryAllocation += size;

        // 检查分配大小是否超过阈值，并记录信息
        if (size > THRESHOLD) {
            std::lock_guard<std::mutex> lock(globalMtx);
            std::cout << "Allocated " << size << " bytes in thread " << std::this_thread::get_id() << "\n";
        }

        return ptr;
    }

    // 自定义 free 函数
    void deallocate(void* ptr) {
        std::free(ptr);
    }
};

class AllocationTagger {
public:
    explicit AllocationTagger() {
        tagger_initialized = true;
    }
    ~AllocationTagger() {
        tagger_initialized = false;
    }
};

// 全局分配器实例
MemAllocator globalAllocator;

// 重载全局 new 和 delete 操作符
void* operator new(size_t size) {
    AllocationTagger tracker;
    return globalAllocator.allocate(size);
}

void operator delete(void* ptr) noexcept {
    globalAllocator.deallocate(ptr);
}

// 重载数组形式的 new 和 delete 操作符
void* operator new[](size_t size) {
    AllocationTagger tracker;
    return globalAllocator.allocate(size);
}

void operator delete[](void* ptr) noexcept {
    globalAllocator.deallocate(ptr);
}

void allocateMemory(size_t size) {
    try {
        char* buffer = new char[size];
        delete[] buffer;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << '\n';
    }
}

int main() {
    std::vector<std::thread> threads;

    // 创建多个线程进行内存分配
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back(allocateMemory, (i + 1) * 1024 * 1024 * 3); // 分配 3MB, 6MB, 9MB, 12MB, 15MB
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 输出每个线程的内存分配情况
    for (const auto& entry : globalMemoryAllocations) {
        std::cout << "Thread " << entry.first << " allocated " << entry.second << " bytes in total.\n";
    }

    return 0;
}
