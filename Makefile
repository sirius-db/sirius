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
TEST_BUILD_TARGET ?= sirius_unittest
MAIN_BUILD_TARGETS ?= duckdb duckdb_local_extension_repo

.PHONY: all release debug reldebug relwithdebinfo debug-release \
	legacy-release \
	clang-release clang-debug clang-relwithdebinfo \
	ci-release configure_ci set_duckdb_version \
	test test_release test_debug test_reldebug test_ci-release clean list-presets \
	s3-up s3-up-large s3-down s3-test s3-sql-test s3-test-large s3-bench s3-bench-fixtures

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
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset release --target $(MAIN_BUILD_TARGETS)
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset release --target $(TEST_BUILD_TARGET)
endif

debug: build/debug/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset debug --target $(MAIN_BUILD_TARGETS)
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset debug --target $(TEST_BUILD_TARGET)
endif

reldebug: relwithdebinfo

debug-release: relwithdebinfo

relwithdebinfo: build/relwithdebinfo/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset relwithdebinfo --target $(MAIN_BUILD_TARGETS)
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset relwithdebinfo --target $(TEST_BUILD_TARGET)
endif

legacy-release: build/legacy-release/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset legacy-release --target $(MAIN_BUILD_TARGETS)

clang-release: build/clang-release/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-release --target $(MAIN_BUILD_TARGETS)
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-release --target $(TEST_BUILD_TARGET)
endif

clang-debug: build/clang-debug/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-debug --target $(MAIN_BUILD_TARGETS)
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-debug --target $(TEST_BUILD_TARGET)
endif

clang-relwithdebinfo: build/clang-relwithdebinfo/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-relwithdebinfo --target $(MAIN_BUILD_TARGETS)
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-relwithdebinfo --target $(TEST_BUILD_TARGET)
endif

ci-release: build/ci-release/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset ci-release --target $(MAIN_BUILD_TARGETS)
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset ci-release --target $(TEST_BUILD_TARGET)
endif

configure_ci:
	@echo "configure_ci step is skipped for this extension build..."

set_duckdb_version:
	@echo "DuckDB version is pinned by the submodule; skipping checkout of $(DUCKDB_GIT_VERSION)."

test: test_release

test_release: release
	./build/release/extension/sirius/test/cpp/sirius_unittest

test_debug: debug
	./build/debug/extension/sirius/test/cpp/sirius_unittest

test_reldebug: relwithdebinfo
	./build/relwithdebinfo/extension/sirius/test/cpp/sirius_unittest

test_ci-release: ci-release
	./build/ci-release/extension/sirius/test/cpp/sirius_unittest

clean:
	rm -rf build

list-presets: $(PRESETS_LINK)
	cd $(DUCKDB_DIR) && $(CMAKE) --list-presets

# -----------------------------------------------------------------------------
# S3 integration test scaffolding
# -----------------------------------------------------------------------------
# `make s3-up`        starts the pinned MinIO container and populates fixtures
#                     (binary blobs plus the standard integration parquet
#                     fixtures under test/cpp/integration/data/parquet).
# `make s3-down`      tears it down (including the data volume).
# `make test`         runs the default Catch2 suite without starting MinIO.
# `make s3-test`      one-shot standard S3 correctness gate: starts MinIO,
#                     sources env.sh, runs every Catch2 test tagged [s3]
#                     except [large] in strict mode, then tears MinIO down
#                     even on failure.
# `make s3-sql-test`  one-shot SQL-over-S3 end-to-end subset: same fixture
#                     lifecycle as s3-test, but runs only [s3][sql] except
#                     [large].
# `make s3-test-large`
#                     one-shot large-SF10 SQL-over-S3 gate: starts MinIO,
#                     uploads standard fixtures plus lineitem_sf10.parquet via
#                     fixtures.sh --perf, then runs [s3][sql][large].
#
# See test/cpp/integration/s3/README.md for details.

S3_DIR := test/cpp/integration/s3
S3_COMPOSE := $(S3_DIR)/docker-compose.yml
S3_TEST_BIN ?= build/release/extension/sirius/test/cpp/sirius_unittest

s3-up:
	docker compose -f $(S3_COMPOSE) up -d
	$(S3_DIR)/fixtures.sh

s3-up-large:
	docker compose -f $(S3_COMPOSE) up -d
	$(S3_DIR)/fixtures.sh --perf

s3-down:
	docker compose -f $(S3_COMPOSE) down -v

s3-test: SHELL := /bin/bash
s3-test:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-test: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@set -e; \
	trap '$(MAKE) s3-down' EXIT; \
	$(MAKE) s3-up; \
	source $(S3_DIR)/env.sh; \
	export SIRIUS_TEST_S3_STRICT=1; \
	$(S3_TEST_BIN) "[s3]~[large]"

s3-sql-test: SHELL := /bin/bash
s3-sql-test:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-sql-test: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@set -e; \
	trap '$(MAKE) s3-down' EXIT; \
	$(MAKE) s3-up; \
	source $(S3_DIR)/env.sh; \
	export SIRIUS_TEST_S3_STRICT=1; \
	$(S3_TEST_BIN) "[s3][sql]~[large]"

s3-test-large: SHELL := /bin/bash
s3-test-large:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-test-large: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@set -e; \
	trap '$(MAKE) s3-down' EXIT; \
	$(MAKE) s3-up-large; \
	source $(S3_DIR)/env.sh; \
	export SIRIUS_TEST_S3_STRICT=1; \
	$(S3_TEST_BIN) "[s3][sql][large][large-count]"; \
	$(S3_TEST_BIN) "[s3][sql][large][large-q1]"; \
	$(S3_TEST_BIN) "[s3][sql][large][large-join]"

# -----------------------------------------------------------------------------
# S3 perf benchmark (Catch2 [!benchmark][perf][bench] hidden tag - not in the
# default CI suite, and deliberately NOT tagged [s3] so the [s3] integration
# gate does not pull the benchmark in). `make s3-bench-fixtures` runs
# fixtures.sh --perf, which first uploads the standard fixtures and then adds
# the SF10 lineitem parquet. Generates a JSON record under
# build/release/extension/sirius/test/cpp/log/perf_<ts>.json for tracking.
# Override SIRIUS_BENCH_BACKEND=aws-s3 to portably hit AWS instead of MinIO;
# see test/cpp/integration/s3/fixtures/README.md for the env-var contract.

s3-bench-fixtures: SHELL := /bin/bash
s3-bench-fixtures:
	@if [ ! -x $(S3_DIR)/fixtures.sh ]; then \
	  echo "s3-bench-fixtures: $(S3_DIR)/fixtures.sh not executable" >&2; \
	  exit 1; \
	fi
	@$(S3_DIR)/fixtures.sh --perf

s3-bench: SHELL := /bin/bash
s3-bench:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-bench: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@if [ "$${SIRIUS_BENCH_BACKEND:-minio}" = "minio" ]; then \
	  source $(S3_DIR)/env.sh; \
	  export SIRIUS_BENCH_S3_ENDPOINT="$${SIRIUS_BENCH_S3_ENDPOINT:-$$SIRIUS_TEST_S3_ENDPOINT}"; \
	  export SIRIUS_BENCH_S3_REGION="$${SIRIUS_BENCH_S3_REGION:-$$SIRIUS_TEST_S3_REGION}"; \
	  export SIRIUS_BENCH_S3_ACCESS_KEY="$${SIRIUS_BENCH_S3_ACCESS_KEY:-$$SIRIUS_TEST_S3_ACCESS_KEY}"; \
	  export SIRIUS_BENCH_S3_SECRET_KEY="$${SIRIUS_BENCH_S3_SECRET_KEY:-$$SIRIUS_TEST_S3_SECRET_KEY}"; \
	  export SIRIUS_BENCH_S3_BUCKET="$${SIRIUS_BENCH_S3_BUCKET:-$$SIRIUS_TEST_S3_BUCKET}"; \
	fi; \
	export SIRIUS_BENCH_GIT_SHA="$$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"; \
	$(S3_TEST_BIN) "[!benchmark][perf][bench]"
