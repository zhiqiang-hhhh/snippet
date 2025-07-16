project(integrate_faiss)

set(BUILD_WITHOUT_LAPACK OFF CACHE BOOL "Disable LAPACK support in OpenBLAS")
set(NO_SHARED TRUE CACHE BOOL "Disable shared library in OpenBLAS")
set(C_LAPACK TRUE CACHE BOOL "Enable C interface for LAPACK in OpenBLAS")
set(USE_OPENMP TRUE CACHE BOOL "Enable OpenMP in OpenBLAS")
set(NOFORTRAN ON CACHE BOOL "Disable Fortran in OpenBLAS")
set(BUILD_STATIC_LIBS ON CACHE BOOL "Build static libraries in OpenBLAS")
set(BUILD_TESTING OFF CACHE BOOL "Build shared libraries in OpenBLAS")
set(BUILD_RELAPACK ON CACHE BOOL "Build relapack in OpenBLAS")
set(BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmarks in OpenBLAS")
set(NO_LAPACK OFF CACHE BOOL "Disable LAPACK in OpenBLAS")
set(NO_CBLAS ON CACHE BOOL "Disable CBLAS in OpenBLAS")
set(NO_AVX512 ON CACHE BOOL "Disable AVX512 in OpenBLAS")

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

# Save current C and CXX flags to restore later
set(SAVED_CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(SAVED_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Add comprehensive flags to disable all problematic warnings for OpenBLAS
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")

add_subdirectory(${PROJECT_SOURCE_DIR}/../contrib/openblas ${PROJECT_BINARY_DIR}/openblas)

# Restore original flags for other targets
set(CMAKE_C_FLAGS "${SAVED_CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${SAVED_CMAKE_CXX_FLAGS}")

set(OPENBLAS_LIBRARY "${PROJECT_BINARY_DIR}/openblas/lib/libopenblas.a" CACHE PATH "Path to OpenBLAS build directory")
set(FAISS_ENABLE_MKL OFF CACHE BOOL "Disable MKL support in FAISS")
set(FAISS_ENABLE_GPU OFF CACHE BOOL "Disable GPU support in FAISS")
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "Disable Python support in FAISS")
set(FAISS_ENABLE_EXTRAS OFF CACHE BOOL "Disable FAISS extras")
set(BUILD_TESTING OFF CACHE BOOL "Disable FAISS testing")
add_subdirectory(${PROJECT_SOURCE_DIR}/../contrib/faiss ${PROJECT_BINARY_DIR}/faiss)

add_executable(integrate_faiss main.cpp)
target_link_libraries(integrate_faiss PRIVATE faiss)