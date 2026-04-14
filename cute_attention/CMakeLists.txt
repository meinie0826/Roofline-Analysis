cmake_minimum_required(VERSION 3.24)
project(cute_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(cuDNN REQUIRED)
find_package(CUTLASS REQUIRED)

# CuTe is part of CUTLASS >= 3.0
include_directories(${CUTLASS_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kernels)

# CUDA architectures - support Hopper (90) and Blackwell (100)
set(CMAKE_CUDA_ARCHITECTURES 90;100)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")

# Create shared library
add_library(cute_attention SHARED
    kernels/flash_attention_kernel.cu
)

target_link_libraries(cute_attention
    CUDA::cudart
    CUDA::cublas
    ${CUTLASS_LIBRARIES}
)

# Install
install(TARGETS cute_attention LIBRARY DESTINATION lib)
install(DIRECTORY kernels/ DESTINATION include/cute_attention/kernels)
