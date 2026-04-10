# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

CMAKE ?= cmake
DUCKDB_DIR ?= duckdb
TEST_PATH ?= build/release/test/unittest
TEST_PATH_DEBUG ?= build/debug/test/unittest
TEST_PATH_RELWITHDEBINFO ?= build/relwithdebinfo/test/unittest
TEST_BUILD_TARGET ?= unittest

.PHONY: all release debug reldebug relwithdebinfo debug-release \
	clang-release clang-debug clang-relwithdebinfo \
	ci-release configure_ci set_duckdb_version \
	test test_release test_debug test_reldebug clean list-presets

PRESETS_LINK := $(DUCKDB_DIR)/CMakePresets.json

# Inputs that should trigger a CMake re-configure
CMAKE_INPUTS := cmake/CMakePresets.json CMakeLists.txt extension_config.cmake $(wildcard cmake/*.cmake)

all: release

$(PRESETS_LINK): cmake/CMakePresets.json
	rm -f $(DUCKDB_DIR)/CMakeUserPresets.json
	ln -sf ../cmake/CMakePresets.json $@

# Configure step — only re-runs when cmake inputs change
build/%/build.ninja: $(CMAKE_INPUTS) | $(PRESETS_LINK)
	cd $(DUCKDB_DIR) && $(CMAKE) --preset $*

release: build/release/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset release
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset release --target $(TEST_BUILD_TARGET)
endif

debug: build/debug/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset debug
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset debug --target $(TEST_BUILD_TARGET)
endif

reldebug: relwithdebinfo

debug-release: relwithdebinfo

relwithdebinfo: build/relwithdebinfo/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset relwithdebinfo
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset relwithdebinfo --target $(TEST_BUILD_TARGET)
endif

clang-release: build/clang-release/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-release

clang-debug: build/clang-debug/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-debug

clang-relwithdebinfo: build/clang-relwithdebinfo/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-relwithdebinfo

ci-release: build/ci-release/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset ci-release

configure_ci:
	@echo "configure_ci step is skipped for this extension build..."

set_duckdb_version:
	@echo "DuckDB version is pinned by the submodule; skipping checkout of $(DUCKDB_GIT_VERSION)."

test: test_release

test_release: release
	@echo "SQL logic tests use the legacy gpu_processing path and are skipped by default."
	@echo "Run C++ unit tests with: ./build/release/extension/sirius/test/cpp/sirius_unittest"

test_debug: debug
	@echo "SQL logic tests use the legacy gpu_processing path and are skipped by default."
	@echo "Run C++ unit tests with: ./build/debug/extension/sirius/test/cpp/sirius_unittest"

test_reldebug: relwithdebinfo
	@echo "SQL logic tests use the legacy gpu_processing path and are skipped by default."
	@echo "Run C++ unit tests with: ./build/relwithdebinfo/extension/sirius/test/cpp/sirius_unittest"

clean:
	rm -rf build

list-presets: $(PRESETS_LINK)
	cd $(DUCKDB_DIR) && $(CMAKE) --list-presets
