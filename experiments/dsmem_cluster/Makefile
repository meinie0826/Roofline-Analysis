NVCC ?= nvcc
ARCH ?= sm_100a
CXXFLAGS ?= -O3 -std=c++17
NVCCFLAGS ?= $(CXXFLAGS) -arch=$(ARCH) --expt-relaxed-constexpr --use_fast_math

TARGET := dsmem_cluster_bench

.PHONY: all clean

all: $(TARGET)

$(TARGET): dsmem_cluster_bench.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
