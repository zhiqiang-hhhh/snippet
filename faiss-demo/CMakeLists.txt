
set(FAISS_ENABLE_GPU OFF)
set(FAISS_ENABLE_PYTHON OFF)
set(FAISS_ENABLE_EXTRAS OFF)
set(BUILD_TESTING OFF)
# Add faiss/cmake of parrent directory
add_subdirectory(${PROJECT_SOURCE_DIR}/contrib/faiss ${CMAKE_BINARY_DIR}/contrib/faiss)

add_executable(faiss_demo faiss_demo.cpp)
target_link_libraries(faiss_demo PRIVATE faiss)