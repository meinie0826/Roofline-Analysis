
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>


// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        printf("Got Cuda Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
    } \
}

#endif

typedef struct {
    cudaLibrary_t module;
} 2cta_manual_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_2cta_manual_cuda_init(void **);
void _mlir_2cta_manual_cuda_load_to_device(void **);
static inline void 2cta_manual_Kernel_Module_Load(2cta_manual_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_2cta_manual_cuda_init((void **)(&initArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
    int32_t device_id = 0;
    struct {
        cudaLibrary_t **library;
        int32_t *device_id;
        cudaError_t *ret;
    } loadArgs = {&libraryPtr, &device_id, &ret};
    int32_t device_count;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
    for (int32_t i = 0; i < device_count; i++) {
        device_id = i;
        _mlir_2cta_manual_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void 2cta_manual_Kernel_Module_Unload(2cta_manual_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

typedef struct {
    void *data;
    int32_t dynamic_shapes[2];
    int64_t dynamic_strides[1];
} 2cta_manual_Tensor_a_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[2];
    int64_t dynamic_strides[1];
} 2cta_manual_Tensor_b_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[2];
    int64_t dynamic_strides[1];
} 2cta_manual_Tensor_c_t;

#ifdef __cplusplus
extern "C"
#endif
void _mlir_2cta_manual__mlir_ciface_cutlass_host_function_Tensorgmemodiv128i64div1281_Tensorgmemodiv128i64div1281_Tensorgmemodiv512i64div5121(void **args, int32_t num_args);

static inline int32_t cute_dsl_2cta_manual_wrapper(2cta_manual_Kernel_Module_t *module, 2cta_manual_Tensor_a_t *a, 2cta_manual_Tensor_b_t *b, 2cta_manual_Tensor_c_t *c) {
    int32_t ret;
    void *args[4] = {
        a, b, c,
        &ret
    };
    _mlir_2cta_manual__mlir_ciface_cutlass_host_function_Tensorgmemodiv128i64div1281_Tensorgmemodiv128i64div1281_Tensorgmemodiv512i64div5121(args, 4);
    return ret;
}
