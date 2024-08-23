#include <bits/types/struct_timespec.h>
#include <iostream>
#include <ctime>
#include <unistd.h>

int main() {
    // 定义timespec结构体来存储时间信息
    timespec check_invalid_query_last_timestamp;

    // 获取时间（CLOCK_MONOTONIC: 表示程序运行的相对时间，不受系统时间变化影响）
    clock_gettime(CLOCK_MONOTONIC, &check_invalid_query_last_timestamp);

    // 输出秒和纳秒
    std::cout << "Seconds: " << check_invalid_query_last_timestamp.tv_sec << std::endl;
    std::cout << "Nanoseconds: " << check_invalid_query_last_timestamp.tv_nsec << std::endl;

    // sleep 1 Seconds
    sleep(1);

    timespec check_invalid_query_last_timestamp2;
    // 获取时间（CLOCK_MONOTONIC: 表示程序运行的相对时间，不受系统时间变化影响）
    clock_gettime(CLOCK_MONOTONIC, &check_invalid_query_last_timestamp2);

    // 输出秒和纳秒
    std::cout << "Seconds: " << check_invalid_query_last_timestamp2.tv_sec << std::endl;
    std::cout << "Nanoseconds: " << check_invalid_query_last_timestamp2.tv_nsec << std::endl;
    return 0;
}
