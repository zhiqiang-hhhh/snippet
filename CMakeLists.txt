project(Snippet)
cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(fmt)
add_subdirectory(dragonbox)
add_subdirectory(googletest)

add_executable(test-main fmt.cpp)
target_link_libraries(test-main PRIVATE dragonbox_to_chars gtest fmt)
