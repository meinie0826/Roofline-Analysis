cmake_minimum_required(VERSION 3.18)
project(attention_kernels LANGUAGES CXX CUDA)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 80;90;100)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")

# BF16 support
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DBF16_SUPPORTED")

# Create shared library
add_library(attention_kernels SHARED
    kernels/stage0_naive.cu
    kernels/stage1_tiled.cu
    kernels/stage2_shared_mem.cu
    kernels/stage3_tensor_core.cu
    kernels/stage4_final.cu
)

target_include_directories(attention_kernels PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(attention_kernels PRIVATE
    CUDA::cudart
)

# Set output directory
set_target_properties(attention_kernels PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Install target
install(TARGETS attention_kernels LIBRARY DESTINATION lib)

# Print configuration
message(STATUS "")
message(STATUS "FlashAttention Kernels Configuration:")
message(STATUS "  CUDA Version:    ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Output Directory: ${CMAKE_BINARY_DIR}")
message(STATUS "")
