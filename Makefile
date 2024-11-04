PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=sirius
EXT_CONFIG=${PROJ_DIR}extension_config.cmake
GEN=ninja
BUILD_PYTHON=1

# CUDA configuration
CUDA_DIR := /usr/local/cuda-12.5
CXXFLAGS += -I$(CUDA_DIR)/include
CFLAGS += -I$(CUDA_DIR)/include
LDFLAGS += -L$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib64/stubs

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile