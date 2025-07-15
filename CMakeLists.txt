project(Snippet)
cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(contrib/openblas)
add_subdirectory(contrib/faiss)
add_subdirectory(integrate_faiss)