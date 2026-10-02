PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

OVERRIDE_GIT_DESCRIBE ?= v1.5.6
DISABLE_SANITIZER ?= 1

SIRIUS_DUCKDB_LINKAGE ?= shared
# Scan verification resolves the DuckDB host's exported factories.
override EXT_FLAGS += -DEXPORT_DYNAMIC_SYMBOLS=ON -DSIRIUS_DUCKDB_LINKAGE=$(SIRIUS_DUCKDB_LINKAGE)
ifeq ($(SIRIUS_DUCKDB_LINKAGE),static)
export VCPKG_CUDA_ARCHITECTURES ?= $(CUDAARCHS)
override EXT_FLAGS += -DVCPKG_MANIFEST_FEATURES=static -DVCPKG_INSTALL_OPTIONS=--clean-buildtrees-after-build
endif

EXT_NAME=sirius
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

GEN?=ninja
ifeq ($(GEN),ninja)
GENERATOR=-GNinja -DFORCE_COLORED_OUTPUT=1
endif

include extension-ci-tools/makefiles/duckdb_extension.Makefile
