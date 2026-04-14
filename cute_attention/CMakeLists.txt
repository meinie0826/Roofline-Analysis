cmake_minimum_required(VERSION 3.24)
project(cute_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# CUTLASS - header only
set(CUTLASS_INCLUDE_DIRS "${CUTLASS_INCLUDE_DIRS}" CACHE PATH "CUTLASS include directory")
if(NOT CUTLASS_INCLUDE_DIRS)
    if(DEFINED ENV{CUTLASS_PATH})
        set(CUTLASS_INCLUDE_DIRS "$ENV{CUTLASS_PATH}/include")
    elseif(EXISTS "$ENV{HOME}/cutlass/include")
        set(CUTLASS_INCLUDE_DIRS "$ENV{HOME}/cutlass/include")
    else()
        message(FATAL_ERROR "CUTLASS not found. Please set CUTLASS_INCLUDE_DIRS or CUTLASS_PATH env var")
    endif()
endif()

message(STATUS "CUTLASS include: ${CUTLASS_INCLUDE_DIRS}")

# Include paths
include_directories(${CUTLASS_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kernels)

# CUDA architectures - B200 (100) and H100 (90)
set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures")
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 100)
endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")

# Create shared library
add_library(cute_attention SHARED
    kernels/flash_attention_kernel.cu
)

target_link_libraries(cute_attention
    CUDA::cudart
    CUDA::cublas
)

# Install
install(TARGETS cute_attention LIBRARY DESTINATION lib)
install(DIRECTORY kernels/ DESTINATION include/cute_attention/kernels)
