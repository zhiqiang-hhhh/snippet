cmake_minimum_required(VERSION 3.22)
project(Snippet VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-Werror)
add_subdirectory(integrate_faiss)
add_subdirectory(csapp)