# hwsim WS5: GPU compute throttler + calibration victim bench.
# Standalone build; not part of the Sirius extension build.
#
#   make                # sm_100 SASS + compute_100 PTX (PTX JIT covers newer
#                       # parts: cc 10.3 GB300, sm_120 workstation Blackwell)
#   make ARCH=native    # compile for the GPU actually in this box
#   make ARCH=sm_103a   # explicit arch — keep ONE arch across every binary
#                       # whose numbers you compare (arch changes codegen)

NVCC ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
ARCH ?= sm_100
ifeq ($(ARCH),native)
GENCODE ?= -arch=native
else ifeq ($(ARCH),sm_100)
# Blackwell. sm_100 SASS + compute_100 PTX (JIT fallback covers cc 10.3 on GB300).
GENCODE ?= -gencode arch=compute_100,code=sm_100 -gencode arch=compute_100,code=compute_100
else
GENCODE ?= -arch=$(ARCH)
endif
NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE) -Xcompiler -pthread

BINS = throttle_compute victim_bench probe_concurrency

all: $(BINS)

throttle_compute: throttle_compute.cu throttle_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

victim_bench: victim_bench.cu throttle_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcuda

probe_concurrency: probe_concurrency.cu throttle_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
