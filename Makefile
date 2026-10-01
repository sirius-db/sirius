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
MAIN_BUILD_TARGETS ?= duckdb duckdb_local_extension_repo sirius_shared

BUILD_TARGETS := $(MAIN_BUILD_TARGETS) $(TEST_BUILD_TARGET)

.PHONY: all release debug reldebug relwithdebinfo debug-release \
	clang-release clang-debug clang-relwithdebinfo clang-asan clang-tsan \
	ci-release configure_ci set_duckdb_version \
	test test_release test_debug test_reldebug test_ci-release clean list-presets \
	s3-test s3-test-large s3-tpch \
	s3-test-aws s3-test-aws-sigv4 s3-test-aws-broker \
	slot-gate-test

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
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset release --target $(BUILD_TARGETS)

debug: build/debug/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset debug --target $(BUILD_TARGETS)

reldebug: relwithdebinfo

debug-release: relwithdebinfo

relwithdebinfo: build/relwithdebinfo/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset relwithdebinfo --target $(BUILD_TARGETS)

clang-release: build/clang-release/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-release --target $(BUILD_TARGETS)

clang-debug: build/clang-debug/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-debug --target $(BUILD_TARGETS)

clang-relwithdebinfo: build/clang-relwithdebinfo/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-relwithdebinfo --target $(BUILD_TARGETS)

# AddressSanitizer build (RelWithDebInfo + clang). Run inside `pixi shell` (so
# llvm-symbolizer is auto-detected on PATH) with:
#   ASAN_OPTIONS="protect_shadow_gap=0:detect_leaks=0:halt_on_error=0:abort_on_error=1" \
#     ./build/clang-asan/extension/sirius/test/cpp/sirius_unittest
clang-asan: build/clang-asan/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-asan --target $(BUILD_TARGETS)

# ThreadSanitizer build (RelWithDebInfo + clang). Run inside `pixi shell` (so
# llvm-symbolizer is auto-detected on PATH) with:
#   TSAN_OPTIONS="suppressions=$$PWD/tsan.supp:ignore_noninstrumented_modules=1:halt_on_error=0:history_size=7:detect_deadlocks=0" \
#     ./build/clang-tsan/extension/sirius/test/cpp/sirius_unittest
clang-tsan: build/clang-tsan/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-tsan --target $(BUILD_TARGETS)

ci-release: build/ci-release/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset ci-release --target $(MAIN_BUILD_TARGETS)

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
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset ci-release --target $(TEST_BUILD_TARGET)
	./build/ci-release/extension/sirius/test/cpp/sirius_unittest

clean:
	rm -rf build

list-presets: $(PRESETS_LINK)
	cd $(DUCKDB_DIR) && $(CMAKE) --list-presets

# -----------------------------------------------------------------------------
# S3 integration test gates
# -----------------------------------------------------------------------------
# MinIO is started by the test binary itself (test/cpp/utils/s3_container.*) when
# SIRIUS_TEST_S3_AUTO=1 is set. The testcontainers-native bridge it uses is
# fetched and patched at configure time (cmake/testcontainers_native.cmake,
# third_party/testcontainers-native.patch) when SIRIUS_BUILD_S3_TESTS=ON. There
# is no separate `s3-up`/`s3-down` step, no docker-compose, and no env.sh to
# source: the binary spins up HTTP + TLS MinIO on dynamic ports, uploads
# fixtures, runs the tests, and tears the containers down on exit.
#
# `make test`         runs the default Catch2 suite. Without
#                     SIRIUS_TEST_S3_AUTO it does not start MinIO, and the
#                     MinIO-backed cases skip.
# `make s3-test`      standard S3 gate: runs [s3][integration] except
#                     [large]/[aws] (incl. the SQL-over-S3 surface and the tiny
#                     TPC-H Q1-Q22 suite) with MinIO auto-managed, in strict mode.
# `make s3-test-large`
#                     large-fixture gate, run as two processes. Both run the
#                     SF10 lineitem cases, with cache.mode sirius in the first
#                     and cache.mode none in the second
#                     (SIRIUS_TEST_S3_LARGE=1 makes the harness generate and
#                     upload lineitem_sf10.parquet; needs the DuckDB CLI from
#                     `make release`). The first process also runs the SF1 TPC-H
#                     suite (SIRIUS_TEST_S3_TPCH=1) and the 1001-object glob case
#                     (SIRIUS_TEST_S3_GLOB_SCALE=1).
# `make s3-test-aws`  MANUAL real-AWS gate: runs the live [s3][aws] tests against
#                     a real S3 endpoint. It does not set SIRIUS_TEST_S3_AUTO and
#                     expects the caller to provide the endpoint; it is
#                     deliberately excluded from CI. Export the AWS
#                     environment yourself first — including
#                     SIRIUS_TEST_S3_ENDPOINT — (regional S3 endpoint, real
#                     bucket, and assume-role TEMPORARY credentials including the
#                     session token); keep usage bounded.
#
# The S3 targets pass `--order decl`, so cases run in declaration order;
# Catch2 3.9.0 and later would otherwise pick a random order.
#
# Deprecated names, kept for one round: `s3-tpch` (its suites also run in
# s3-test and s3-test-large), `s3-test-aws-sigv4` (runs s3-test-aws, which
# selects the same cases) and `s3-test-aws-broker` (no case carries [broker]).
#
# See test/cpp/integration/s3/README.md for details.

S3_TEST_BIN ?= build/release/extension/sirius/test/cpp/sirius_unittest

# Query-lifecycle concurrency gates. Runs the hidden [slot_leak_gate] cases
# (the worker-pressure gate needs a TPC-H lineitem parquet fixture) plus the
# concurrent keyed-log segmentation check driven through tools/log_analyzer.
# Manual gate for lifecycle/logging changes; not wired into any CI workflow.
# The fixture check is loud on purpose: the hidden cases fail hard when the
# fixture is missing, so a mis-provisioned run must not look green.
SLOT_GATE_TPCH_DIR ?= test_datasets/tpch_parquet

slot-gate-test:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "slot-gate-test: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@if [ ! -f $(SLOT_GATE_TPCH_DIR)/lineitem.parquet ]; then \
	  echo "slot-gate-test: $(SLOT_GATE_TPCH_DIR)/lineitem.parquet missing - export TPC-H SF1 lineitem there or set SLOT_GATE_TPCH_DIR" >&2; \
	  exit 1; \
	fi
	@set -e; \
	SIRIUS_TEST_TPCH_DIR=$(SLOT_GATE_TPCH_DIR) $(S3_TEST_BIN) "[slot_leak_gate]"; \
	python3 tools/log_analyzer/verify_query_lifecycle_segments.py

s3-test:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-test: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@set -e; \
	export SIRIUS_TEST_S3_AUTO=1 SIRIUS_TEST_S3_STRICT=1; \
	$(S3_TEST_BIN) --order decl "[s3][integration]~[large]~[aws]"

s3-test-large:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-test-large: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@# Two processes, so MinIO is brought up once per group: first the SF10
	@# lineitem cases with cache.mode sirius ([large-cache]) plus the SF1 TPC-H
	@# suite and the 1001-object glob case, which use cache.mode none; then the
	@# SF10 lineitem cases with cache.mode none ([large-nocache]). Catch2
	@# OR-combines specs within one argument via commas (multiple positional args
	@# are AND-concatenated instead).
	@# SIRIUS_TEST_S3_TPCH gates the SF1 TPC-H fixture + [tpch][large]; SIRIUS_TEST_S3_GLOB_SCALE
	@# gates the 1001-object fixture + [glob-scale]. Both are scoped to the first group only, so
	@# the second group's bring-up must not see them (it would re-generate / re-upload).
	@set -e; \
	export SIRIUS_TEST_S3_AUTO=1 SIRIUS_TEST_S3_LARGE=1 SIRIUS_TEST_S3_STRICT=1; \
	SIRIUS_TEST_S3_TPCH=1 SIRIUS_TEST_S3_GLOB_SCALE=1 $(S3_TEST_BIN) --order decl "[s3][sql][large][large-cache],[s3][integration][sql][tpch][large],[s3][large][glob-scale]"; \
	$(S3_TEST_BIN) --order decl "[s3][sql][large][large-nocache]"

# Deprecated: the tiny TPC-H suite runs in s3-test and the SF1 suite in
# s3-test-large. Kept for one round with its old selection (uploads the SF1
# TPC-H fixture with SIRIUS_TEST_S3_TPCH=1 and runs both suites).
s3-tpch:
	@echo "s3-tpch is deprecated: the TPC-H suites run in s3-test (tiny) and s3-test-large (SF1)." >&2
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-tpch: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@set -e; \
	export SIRIUS_TEST_S3_AUTO=1 SIRIUS_TEST_S3_STRICT=1 SIRIUS_TEST_S3_TPCH=1; \
	$(S3_TEST_BIN) --order decl "[s3][integration][sql][tpch]"

# Manual real-AWS gates. These never start MinIO/Docker and are excluded from
# CI. Export the AWS environment yourself before invoking (regional S3 endpoint,
# real bucket, and assume-role TEMPORARY credentials including the session
# token); keep usage bounded. SIRIUS_TEST_S3_STRICT=1 turns a missing-env skip
# into a hard failure so a misconfigured run is loud rather than silently green.
s3-test-aws:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-test-aws: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@set -e; \
	export SIRIUS_TEST_S3_STRICT=1; \
	$(S3_TEST_BIN) --order decl "[s3][aws]"

# Deprecated: selects the same cases as s3-test-aws.
s3-test-aws-sigv4:
	@echo "s3-test-aws-sigv4 is deprecated: use s3-test-aws, which selects the same cases." >&2
	@$(MAKE) --no-print-directory s3-test-aws

# Deprecated: no test case carries [broker], so this target runs nothing.
s3-test-aws-broker:
	@echo "s3-test-aws-broker is deprecated and runs nothing: no test case carries [broker]." >&2
