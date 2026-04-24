NVCC ?= nvcc
ARCH ?= sm_100a
NVCC_FLAGS ?= -std=c++17 -O3 -lineinfo -arch=$(ARCH)
LDFLAGS ?= -lcublas
CUTLASS_DIR ?= ../../3rd/cutlass
SYNCLOG_PATCH ?= ../blackwell_dsmem_2sm/synclog_host_device.patch

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
STEP5_TARGET := bench_step5_hopper_multistage_ws_tma
STEP6_TARGET := bench_step6_hopper_swizzle_ws_tma
STEP7_TARGET := bench_step7_tcgen05_umma
SOURCES := \
	bench_gemm_reference.cu \
	naive_gemm.cu \
	cublas_reference.cu \
	simt_gemm_cp_async.cu

.PHONY: all clean prepare_cutlass_synclog

all: $(TARGET)

$(TARGET): $(SOURCES) gemm_reference_common.h
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $@ $(LDFLAGS)

$(STEP2_TARGET): step2_tcgen05_mma_gemm.cu gemm_reference_common.h
	$(NVCC) -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr $(STEP2_GENCODE) $< -o $@ $(LDFLAGS)

prepare_cutlass_synclog:
	@if [ ! -f "$(SYNCLOG_PATCH)" ]; then \
		echo "ERROR: synclog patch not found: $(SYNCLOG_PATCH)"; \
		exit 1; \
	fi
	@if patch -p1 -d "$(CUTLASS_DIR)" --dry-run < "$(SYNCLOG_PATCH)" >/dev/null 2>&1; then \
		echo "Applying CUTLASS synclog host/device patch"; \
		patch -p1 -d "$(CUTLASS_DIR)" < "$(SYNCLOG_PATCH)"; \
	else \
		echo "CUTLASS synclog patch already applied or not needed"; \
	fi

$(STEP3_TARGET): step3_hopper_tma_gemm.cu gemm_reference_common.h
	$(NVCC) -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr $(STEP2_GENCODE) $< -o $@ $(LDFLAGS) -lcuda

$(STEP4_TARGET): step4_hopper_ws_tma_gemm.cu gemm_reference_common.h
	$(NVCC) -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr $(STEP2_GENCODE) $< -o $@ $(LDFLAGS) -lcuda

$(STEP5_TARGET): step5_hopper_multistage_ws_tma_gemm.cu gemm_reference_common.h
	$(NVCC) -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr $(STEP2_GENCODE) $< -o $@ $(LDFLAGS) -lcuda

$(STEP6_TARGET): step6_hopper_swizzle_ws_tma_gemm.cu gemm_reference_common.h
	$(NVCC) -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr $(STEP2_GENCODE) $< -o $@ $(LDFLAGS) -lcuda

$(STEP7_TARGET): step7_tcgen05_umma_gemm.cu gemm_reference_common.h prepare_cutlass_synclog
	$(NVCC) -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr $(STEP2_GENCODE) \
		-I$(CUTLASS_DIR)/include \
		-DCUTLASS_ARCH_MMA_SM100_ENABLED -DCUTLASS_ARCH_MMA_SM100A_ENABLED -DCUTLASS_ARCH_MMA_SM103A_ENABLED \
		$< -o $@ $(LDFLAGS) -lcuda

clean:
	rm -f $(TARGET) $(STEP2_TARGET) $(STEP3_TARGET) $(STEP4_TARGET) $(STEP5_TARGET) $(STEP6_TARGET) $(STEP7_TARGET)
