# hwsim WS5: GPU compute throttler + calibration victim bench.
# Standalone build; not part of the Sirius extension build.

NVCC    ?= /usr/local/cuda/bin/nvcc
# Blackwell. sm_100 SASS + compute_100 PTX (JIT fallback covers cc 10.3 on GB300).
GENCODE ?= -gencode arch=compute_100,code=sm_100 -gencode arch=compute_100,code=compute_100
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
