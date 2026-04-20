NVCC ?= nvcc
ARCH ?= sm_100a
NVCC_FLAGS ?= -std=c++17 -O3 -lineinfo -arch=$(ARCH)
LDFLAGS ?= -lcublas

ifeq ($(ARCH),sm_100a)
STEP2_GENCODE := -gencode arch=compute_100a,code=sm_100a
else ifeq ($(ARCH),sm_103a)
STEP2_GENCODE := -gencode arch=compute_103a,code=sm_103a
else
$(error step2 tcgen05 target only supports ARCH=sm_100a or ARCH=sm_103a)
endif

TARGET := bench_gemm_reference
STEP2_TARGET := bench_step2_tcgen05_mma
STEP3_TARGET := bench_step3_hopper_tma
STEP4_TARGET := bench_step4_hopper_ws_tma
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
	$(NVCC) -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr $(STEP2_GENCODE) $< -o $@ $(LDFLAGS)

$(STEP3_TARGET): step3_hopper_tma_gemm.cu gemm_reference_common.h
	$(NVCC) -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr $(STEP2_GENCODE) $< -o $@ $(LDFLAGS) -lcuda

$(STEP4_TARGET): step4_hopper_ws_tma_gemm.cu gemm_reference_common.h
	$(NVCC) -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr $(STEP2_GENCODE) $< -o $@ $(LDFLAGS) -lcuda

clean:
	rm -f $(TARGET) $(STEP2_TARGET) $(STEP3_TARGET) $(STEP4_TARGET)
