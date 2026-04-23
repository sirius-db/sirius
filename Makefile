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
	test test_release test_debug test_reldebug clean list-presets \
	s3-up s3-down s3-test s3-cpp-test

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

# -----------------------------------------------------------------------------
# S3 integration test scaffolding (PR15)
# -----------------------------------------------------------------------------
# `make s3-up`        starts the pinned MinIO container and populates fixtures
#                     (binary blobs plus the standard integration parquet
#                     fixtures under test/cpp/integration/data/parquet).
# `make s3-down`      tears it down (including the data volume).
# `make s3-test`      alias for `s3-cpp-test`; kept for forward compatibility
#                     if SQL-level integration returns via a new target later.
# `make s3-cpp-test`  runs the Catch2 [s3][integration] tag, which also
#                     selects tests tagged [s3][parquet][integration].
#
# See test/cpp/integration/s3/README.md for details.

S3_DIR := test/cpp/integration/s3
S3_COMPOSE := $(S3_DIR)/docker-compose.yml
S3_TEST_BIN ?= build/release/extension/sirius/test/cpp/sirius_unittest

s3-up:
	docker compose -f $(S3_COMPOSE) up -d
	$(S3_DIR)/fixtures.sh

s3-down:
	docker compose -f $(S3_COMPOSE) down -v

s3-test: SHELL := /bin/bash
s3-test: s3-cpp-test

s3-cpp-test: SHELL := /bin/bash
s3-cpp-test:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-cpp-test: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@source $(S3_DIR)/env.sh && export SIRIUS_TEST_S3_STRICT=1 && $(S3_TEST_BIN) "[s3][integration]"
