# hwsim membw throttle tools: rate-controlled bandwidth eaters + victim benchmarks.
#
#   make            # build membw_eater + membw_victim with nvcc (CUDA required)
#   make nocuda     # build *_nocuda variants with g++ only (dram domain only)
#   make ARCH=sm_103a   # override GPU arch (default sm_100; be consistent -
#                       # sm_103a vs sm_100 changes codegen on GB300)
#
# nvcc lookup order: $(NVCC), then PATH, then /usr/local/cuda/bin/nvcc.

NVCC ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CXX  ?= g++
ARCH ?= sm_100

NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) --compiler-options -Wall
CXXFLAGS  = -O3 -std=c++17 -Wall -DNO_CUDA -x c++
LDLIBS    = -lpthread

BINS = membw_eater membw_victim

all: $(BINS)

membw_eater: membw_eater.cu membw_common.h
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDLIBS)

membw_victim: membw_victim.cu membw_common.h
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDLIBS)

# CUDA-free fallback: host-DRAM domain only.
nocuda: membw_eater_nocuda membw_victim_nocuda

membw_eater_nocuda: membw_eater.cu membw_common.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

membw_victim_nocuda: membw_victim.cu membw_common.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f $(BINS) membw_eater_nocuda membw_victim_nocuda

.PHONY: all nocuda clean
