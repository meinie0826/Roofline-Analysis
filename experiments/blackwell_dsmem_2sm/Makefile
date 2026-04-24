# Makefile for Blackwell DSMEM 2SM experiments

NVCC = nvcc
ARCH ?= sm_100a
NVCC_FLAGS = -std=c++17 -O3 -arch=$(ARCH) --expt-relaxed-constexpr -Xcudafe --diag_suppress=20012 -Xcudafe --diag_suppress=20013

# CUTLASS path (for existing benchmarks)
CUTLASS_PATH ?= ../../3rd/cutlass
CUTLASS_INCLUDES = -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include

# Benchmarks
BENCHMARKS = bench_cutlass_2sm_gemm bench_cutlass_mma2sm_cost bench_d1_only bench_umma_1sm_2sm_cost

.PHONY: all clean cutlass

all: cutlass

cutlass: $(BENCHMARKS)

# CUTLASS benchmarks
bench_cutlass_2sm_gemm: bench_cutlass_2sm_gemm.cu common.h
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDES) $< -o $@

bench_cutlass_mma2sm_cost: bench_cutlass_mma2sm_cost.cu common.h
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDES) $< -o $@

bench_d1_only: bench_d1_only.cu common.h
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDES) $< -o $@

bench_umma_1sm_2sm_cost: bench_umma_1sm_2sm_cost.cu common.h
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(BENCHMARKS)
