NVCC ?= nvcc
ARCH ?= sm_100a
NVCC_FLAGS ?= -std=c++17 -O3 -lineinfo -arch=$(ARCH)
LDFLAGS ?= -lcublas
CUTLASS_PATH ?= ../../cutlass
CUTLASS_INCLUDES := -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include

TARGET := bench_gemm_reference
STEP2_TARGET := bench_step2_tcgen05_mma
SOURCES := \
	bench_gemm_reference.cu \
	naive_gemm.cu \
	cublas_reference.cu \
	simt_gemm_cp_async.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) gemm_reference_common.h
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $@ $(LDFLAGS)

$(STEP2_TARGET): step2_tcgen05_mma_gemm.cu gemm_reference_common.h
	$(NVCC) $(NVCC_FLAGS) --expt-relaxed-constexpr $(CUTLASS_INCLUDES) \
		-DCUTLASS_ARCH_MMA_SM100_ENABLED -DCUTLASS_ARCH_MMA_SM100A_ENABLED -DCUTLASS_ARCH_MMA_SM103A_ENABLED \
		$< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) $(STEP2_TARGET)
