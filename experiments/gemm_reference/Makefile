NVCC ?= nvcc
ARCH ?= sm_100a
NVCC_FLAGS ?= -std=c++17 -O3 -lineinfo -arch=$(ARCH)
LDFLAGS ?= -lcublas

TARGET := bench_gemm_reference
SOURCES := \
	bench_gemm_reference.cu \
	naive_gemm.cu \
	cublas_reference.cu \
	simt_gemm_cp_async.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) gemm_reference_common.h
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
