PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

DISABLE_SANITIZER ?= 1

SIRIUS_DUCKDB_LINKAGE ?= $(if $(VCPKG_TOOLCHAIN_PATH),static,shared)
# Scan verification resolves the DuckDB host's exported factories.
override EXT_FLAGS += -DEXPORT_DYNAMIC_SYMBOLS=ON -DSIRIUS_DUCKDB_LINKAGE=$(SIRIUS_DUCKDB_LINKAGE)
# Always passed, empty off the release commit, so a reused cache can't keep a stale version.
SIRIUS_DUCKDB_EXPLICIT_VERSION := $(if $(filter d54cbbe3df17e8e5ce68312f483b4155836112aa,$(shell git -C '$(PROJ_DIR)duckdb' rev-parse HEAD 2>/dev/null)),v2.0.0,)
override EXT_FLAGS += -DDUCKDB_EXPLICIT_VERSION=$(SIRIUS_DUCKDB_EXPLICIT_VERSION)
ifeq ($(SIRIUS_DUCKDB_LINKAGE),static)
export VCPKG_CUDA_ARCHITECTURES ?= $(CUDAARCHS)
# Conda's CUDA activation adds the toolkit include dir, whose CCCL (2.x on CUDA 12)
# would shadow the newer one vcpkg installs.
ifneq ($(strip $(CONDA_PREFIX)),)
CONDA_CUDA_INCLUDE := -I$(CONDA_PREFIX)/targets/%/include
export CFLAGS := $(filter-out $(CONDA_CUDA_INCLUDE),$(CFLAGS))
export CXXFLAGS := $(filter-out $(CONDA_CUDA_INCLUDE),$(CXXFLAGS))
export CPPFLAGS := $(filter-out $(CONDA_CUDA_INCLUDE),$(CPPFLAGS))
endif
override EXT_FLAGS += -DVCPKG_MANIFEST_INSTALL=ON -DVCPKG_INSTALL_OPTIONS=--clean-buildtrees-after-build
else ifneq ($(strip $(VCPKG_TOOLCHAIN_PATH)),)
override EXT_FLAGS += -DVCPKG_MANIFEST_INSTALL=OFF
endif
ifneq ($(strip $(SIRIUS_INSTALL_DIR)),)
override EXT_FLAGS += -Dsirius_ROOT='$(SIRIUS_INSTALL_DIR)'
endif

GEN ?= ninja

EXT_NAME=sirius
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

include extension-ci-tools/makefiles/duckdb_extension.Makefile

# The upstream release target hardcodes build/release. Keep its flags and
# prerequisites, but allow the parent Sirius build to select an isolated root.
CMAKE ?= cmake
BUILD_DIR ?= build
.PHONY: sirius-release
sirius-release: ${EXTENSION_CONFIG_STEP}
	$(CMAKE) $(GENERATOR) $(BUILD_FLAGS) $(EXT_RELEASE_FLAGS) $(VCPKG_MANIFEST_FLAGS) \
		-DCMAKE_BUILD_TYPE=Release -S $(DUCKDB_SRCDIR) -B "$(BUILD_DIR)/release"
	$(CMAKE) --build "$(BUILD_DIR)/release" --config Release

.PHONY: configure_sirius_submodules
configure_ci: configure_sirius_submodules
configure_sirius_submodules:
	git -C '$(PROJ_DIR)..' submodule update --init --recursive -- duckdb cucascade substrait
