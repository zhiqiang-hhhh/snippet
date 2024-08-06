#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <list>
#include <mutex>
#include <unistd.h>
#include <utility>
#include <vector>
#include <thread>
#include <stdlib.h>

int main() {
    for (int i = 0; i < 0; ++i) {
        std::list<int> data {1,2,3,4,5,6,7,8,9,10,11, 12,13,14,15,16,17,18,19};

        std::mutex res_lock;
        std::list<int> res;
        std::vector<std::thread> temp_threads;

        const size_t batch_size = 3;
        size_t ctr = 0;
        using IterType = decltype(data)::iterator;
        IterType batch_begin;
    
        auto frgmt_func = [&ref_lock = res_lock, &ref_res = res](std::pair<IterType, IterType> batch_range) {
            sleep(1);
            std::lock_guard<std::mutex> lg(ref_lock);
            std::cout << std::this_thread::get_id() << ':' << *batch_range.first << ':' << *batch_range.second << std::endl;
            for (auto& itr = batch_range.first; itr != batch_range.second; ++itr) {
                ref_res.push_back(*itr);
            }
        };

        for (auto itr = std::begin(data); itr != std::end(data); ++itr) {
            ++ctr;
            if (ctr % batch_size == 1) {
                if (ctr > batch_size) {
                    std::pair<IterType, IterType> batch = std::make_pair(batch_begin, itr);
                    temp_threads.emplace_back(std::thread(std::bind(frgmt_func, batch)));
                }
                batch_begin = itr;
            }
        }

        std::pair<IterType, IterType> batch = std::make_pair(batch_begin, std::end(data));
        temp_threads.emplace_back(std::thread(std::bind(frgmt_func, batch)));


        for (auto& thr : temp_threads) {
            thr.join();
        }

        for (const auto& d : res) {
            std::cout << d << ',';
        }
        std::cout << '[' << i <<std::endl;
    }

    std::string str ("1638766800000");
    std::cout << (int64_t)atoi(str.c_str()) << '\t' << (int32_t)atoi(str.c_str()) << std::endl;
std::cout << (long)atol(str.c_str()) << '\t' << (int32_t)atol(str.c_str()) << std::endl;
    return 0;
}
