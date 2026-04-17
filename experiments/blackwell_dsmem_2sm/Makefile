# Makefile for final comparison benchmark

CUTLASS_PATH ?= ../../cutlass
NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 -arch=sm_103a
INCLUDES = -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include

all: bench_final_comparison

bench_final_comparison: bench_final_comparison.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f bench_final_comparison

.PHONY: all clean
