PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

DISABLE_SANITIZER ?= 1

SIRIUS_DUCKDB_LINKAGE ?= $(if $(VCPKG_TOOLCHAIN_PATH),static,shared)
# Scan verification resolves the DuckDB host's exported factories.
override EXT_FLAGS += -DEXPORT_DYNAMIC_SYMBOLS=ON -DSIRIUS_DUCKDB_LINKAGE=$(SIRIUS_DUCKDB_LINKAGE)
ifeq ($(SIRIUS_DUCKDB_LINKAGE),static)
export VCPKG_CUDA_ARCHITECTURES ?= $(CUDAARCHS)
override EXT_FLAGS += -DVCPKG_MANIFEST_FEATURES=static -DVCPKG_INSTALL_OPTIONS=--clean-buildtrees-after-build
endif
ifneq ($(strip $(SIRIUS_INSTALL_DIR)),)
override EXT_FLAGS += -Dsirius_DIR='$(SIRIUS_INSTALL_DIR)/lib/cmake/sirius'
endif

GEN ?= ninja

EXT_NAME=sirius
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

include extension-ci-tools/makefiles/duckdb_extension.Makefile

.PHONY: configure_sirius_submodules
configure_ci: configure_sirius_submodules
configure_sirius_submodules:
	git -C '$(PROJ_DIR)..' submodule update --init --recursive -- duckdb cucascade substrait
