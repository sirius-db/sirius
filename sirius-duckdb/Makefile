PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

CMAKE ?= cmake
SIRIUS_SOURCE_DIR ?= $(abspath $(PROJ_DIR)..)
SIRIUS_CONFIG ?= release

OVERRIDE_GIT_DESCRIBE ?= v1.5.6
DISABLE_SANITIZER ?= 1

SIRIUS_DUCKDB_LINKAGE ?= $(if $(VCPKG_TOOLCHAIN_PATH),static,shared)
# Scan verification resolves the DuckDB host's exported factories.
override EXT_FLAGS += -DEXPORT_DYNAMIC_SYMBOLS=ON -DSIRIUS_DUCKDB_LINKAGE=$(SIRIUS_DUCKDB_LINKAGE)
ifeq ($(SIRIUS_DUCKDB_LINKAGE),static)
export VCPKG_CUDA_ARCHITECTURES ?= $(CUDAARCHS)
VCPKG_INSTALLED_DIR ?= $(PROJ_DIR)build/$(SIRIUS_CONFIG)/vcpkg_installed
override EXT_FLAGS += -DVCPKG_INSTALLED_DIR='$(VCPKG_INSTALLED_DIR)'
override EXT_FLAGS += -DVCPKG_MANIFEST_FEATURES=static -DVCPKG_INSTALL_OPTIONS=--clean-buildtrees-after-build
endif

EXT_NAME=sirius
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

GEN?=ninja
ifeq ($(GEN),ninja)
GENERATOR=-GNinja -DFORCE_COLORED_OUTPUT=1
endif

# An explicit install directory skips the local engine build.
ifeq ($(strip $(SIRIUS_INSTALL_DIR)),)
BUILD_LOCAL_SIRIUS := 1
SIRIUS_BUILD_ROOT ?= $(PROJ_DIR)build/sirius-$(SIRIUS_DUCKDB_LINKAGE)
SIRIUS_INSTALL_DIR = $(SIRIUS_BUILD_ROOT)/$(SIRIUS_CONFIG)/install
endif
override EXT_FLAGS += -Dsirius_DIR='$(SIRIUS_INSTALL_DIR)/lib/cmake/sirius'

include extension-ci-tools/makefiles/duckdb_extension.Makefile

release sirius-release: SIRIUS_CONFIG=release
release sirius-release: SIRIUS_BUILD_TYPE=Release
debug sirius-debug clangd extension_configuration_default: SIRIUS_CONFIG=debug
debug sirius-debug clangd extension_configuration_default: SIRIUS_BUILD_TYPE=Debug
reldebug sirius-reldebug: SIRIUS_CONFIG=reldebug
reldebug sirius-reldebug: SIRIUS_BUILD_TYPE=RelWithDebInfo
relassert sirius-relassert: SIRIUS_CONFIG=relassert
relassert sirius-relassert: SIRIUS_BUILD_TYPE=RelWithDebInfo
relassert sirius-relassert: SIRIUS_CONFIG_FLAGS=-DFORCE_ASSERT=ON

ifeq ($(BUILD_LOCAL_SIRIUS),1)
release tidy-check: sirius-release
debug clangd extension_configuration_default: sirius-debug
reldebug: sirius-reldebug
relassert: sirius-relassert

SIRIUS_COMPILER_CACHE ?= $(shell command -v sccache 2>/dev/null || command -v ccache 2>/dev/null)
ifneq ($(SIRIUS_COMPILER_CACHE),)
SIRIUS_CACHE_FLAGS = -DCMAKE_C_COMPILER_LAUNCHER='$(SIRIUS_COMPILER_CACHE)' \
  -DCMAKE_CXX_COMPILER_LAUNCHER='$(SIRIUS_COMPILER_CACHE)' \
  -DCMAKE_CUDA_COMPILER_LAUNCHER='$(SIRIUS_COMPILER_CACHE)'
endif

ifeq ($(SIRIUS_DUCKDB_LINKAGE),static)
SIRIUS_LIBRARY_FLAGS = -DSIRIUS_BUILD_STATIC=ON -DSIRIUS_BUILD_SHARED=OFF -DCPM_LOCAL_PACKAGES_ONLY=ON \
  -DVCPKG_MANIFEST_DIR='$(PROJ_DIR)' -DVCPKG_MANIFEST_FEATURES=static \
  -DVCPKG_INSTALLED_DIR='$(VCPKG_INSTALLED_DIR)' \
  -DVCPKG_INSTALL_OPTIONS=--clean-buildtrees-after-build \
  -DCMAKE_CUDA_ARCHITECTURES='$(VCPKG_CUDA_ARCHITECTURES)'
else
SIRIUS_LIBRARY_FLAGS = -DSIRIUS_BUILD_SHARED=ON -DSIRIUS_BUILD_STATIC=OFF
endif

.PHONY: sirius-release sirius-debug sirius-reldebug sirius-relassert configure_sirius_submodules
sirius-release sirius-debug sirius-reldebug sirius-relassert:
	$(CMAKE) -S '$(SIRIUS_SOURCE_DIR)' -B '$(SIRIUS_BUILD_ROOT)/$(SIRIUS_CONFIG)' $(GENERATOR) \
		-DCMAKE_BUILD_TYPE=$(SIRIUS_BUILD_TYPE) -DCMAKE_INSTALL_LIBDIR=lib \
		-DSIRIUS_BUILD_TESTS=OFF $(TOOLCHAIN_FLAGS) $(SIRIUS_LIBRARY_FLAGS) $(SIRIUS_CACHE_FLAGS) $(SIRIUS_CMAKE_FLAGS) $(SIRIUS_CONFIG_FLAGS)
	$(CMAKE) --build '$(SIRIUS_BUILD_ROOT)/$(SIRIUS_CONFIG)' --target sirius_library --config $(SIRIUS_BUILD_TYPE)
	$(CMAKE) --install '$(SIRIUS_BUILD_ROOT)/$(SIRIUS_CONFIG)' --prefix '$(SIRIUS_INSTALL_DIR)' --component sirius_library --config $(SIRIUS_BUILD_TYPE)

configure_ci: configure_sirius_submodules
configure_sirius_submodules:
	git -C '$(SIRIUS_SOURCE_DIR)' submodule update --init --recursive -- duckdb cucascade substrait
endif
