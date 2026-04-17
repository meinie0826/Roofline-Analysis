# Makefile for Blackwell DSMEM 2SM experiments

NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 -arch=sm_103a

# CUTLASS path (for existing benchmarks)
CUTLASS_PATH ?= ../../cutlass
CUTLASS_INCLUDES = -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include

# Hand-written benchmarks (no CUTLASS)
HAND_WRITTEN = bench_3way bench_tc_simple

# CUTLASS benchmarks
CUTLASS_BENCHMARKS = bench_cutlass_2sm_gemm bench_cutlass_mma2sm_cost bench_d1_only

.PHONY: all clean hand-written cutlass

# Build hand-written benchmarks by default
all: hand-written

hand-written: $(HAND_WRITTEN)

cutlass: $(CUTLASS_BENCHMARKS)

# Hand-written benchmarks: no CUTLASS includes needed
bench_3way: bench_3way.cu common.h
	$(NVCC) $(NVCC_FLAGS) $< -o $@

bench_tc_simple: bench_tc_simple.cu common.h
	$(NVCC) $(NVCC_FLAGS) $< -o $@

bench_tc_full: bench_tc_full.cu common.h
	$(NVCC) $(NVCC_FLAGS) $< -o $@

.PHONY: tc
tc: bench_tc_simple bench_tc_full

# CUTLASS benchmarks
bench_cutlass_2sm_gemm: bench_cutlass_2sm_gemm.cu common.h
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDES) $< -o $@

bench_cutlass_mma2sm_cost: bench_cutlass_mma2sm_cost.cu common.h
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDES) $< -o $@

bench_d1_only: bench_d1_only.cu common.h
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDES) $< -o $@

clean:
	rm -f $(HAND_WRITTEN) $(CUTLASS_BENCHMARKS)
