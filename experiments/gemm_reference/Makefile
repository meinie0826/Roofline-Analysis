NVCC ?= nvcc
ARCH ?= sm_100a
NVCC_FLAGS ?= -std=c++17 -O3 -lineinfo -arch=$(ARCH)
LDFLAGS ?= -lcublas

TARGET := bench_gemm_reference

.PHONY: all clean

all: $(TARGET)

$(TARGET): bench_gemm_reference.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
