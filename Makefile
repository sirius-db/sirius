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
	clang-release clang-debug clang-relwithdebinfo clang-asan clang-tsan \
	ci-release configure_ci set_duckdb_version \
	test test_release test_debug test_reldebug test_ci-release clean list-presets \
	s3-test s3-test-large \
	s3-test-aws s3-test-aws-sigv4 s3-test-aws-broker s3-bench

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

# AddressSanitizer build (RelWithDebInfo + clang). Run inside `pixi shell` (so
# llvm-symbolizer is auto-detected on PATH) with:
#   ASAN_OPTIONS="protect_shadow_gap=0:detect_leaks=0:halt_on_error=0:abort_on_error=1" \
#     ./build/clang-asan/extension/sirius/test/cpp/sirius_unittest
clang-asan: build/clang-asan/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-asan --target $(MAIN_BUILD_TARGETS)
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-asan --target $(TEST_BUILD_TARGET)
endif

# ThreadSanitizer build (RelWithDebInfo + clang). Run inside `pixi shell` (so
# llvm-symbolizer is auto-detected on PATH) with:
#   TSAN_OPTIONS="suppressions=$$PWD/tsan.supp:ignore_noninstrumented_modules=1:halt_on_error=0:history_size=7:detect_deadlocks=0" \
#     ./build/clang-tsan/extension/sirius/test/cpp/sirius_unittest
clang-tsan: build/clang-tsan/build.ninja
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-tsan --target $(MAIN_BUILD_TARGETS)
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-tsan --target $(TEST_BUILD_TARGET)
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
# S3 integration test gates
# -----------------------------------------------------------------------------
# MinIO is now started by the test binary itself (test/cpp/utils/s3_container.*,
# via the vendored testcontainers-native bridge) when SIRIUS_TEST_S3_AUTO=1 is
# set. There is no separate `s3-up`/`s3-down` step, no docker-compose, and no
# env.sh to source: the binary spins up HTTP + TLS MinIO on dynamic ports,
# uploads fixtures, runs the tests, and tears the containers down on exit.
#
# `make test`         runs the default Catch2 suite; AUTO is unset, so no Docker.
# `make s3-test`      standard S3 correctness gate: runs [s3][integration] except
#                     [large]/[aws] (incl. the SQL-over-S3 surface) with MinIO
#                     auto-managed, in strict mode.
# `make s3-test-large`
#                     large-SF10 SQL-over-S3 gate. SIRIUS_TEST_S3_LARGE=1 makes
#                     the harness generate + upload lineitem_sf10.parquet (needs
#                     the DuckDB CLI from `make release`), then runs
#                     [s3][sql][large].
#
# The s3-test-aws* targets are MANUAL real-AWS gates: they never start MinIO
# (AUTO is unset) and are deliberately excluded from CI. Export the AWS
# environment yourself first — including SIRIUS_TEST_S3_ENDPOINT — (regional S3
# endpoint, real bucket, and assume-role TEMPORARY credentials including the
# session token); keep usage bounded.
# `make s3-test-aws`  runs the live [s3][aws] tests against a real S3 endpoint.
# `make s3-test-aws-sigv4`
#                     subset using Sirius's built-in SigV4 presigner only
#                     ([s3][aws] minus [broker]).
# `make s3-test-aws-broker`
#                     subset driven by an external presign broker
#                     ([s3][aws][broker]).
#
# See test/cpp/integration/s3/README.md for details.

S3_TEST_BIN ?= build/release/extension/sirius/test/cpp/sirius_unittest

s3-test:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-test: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@set -e; \
	export SIRIUS_TEST_S3_AUTO=1 SIRIUS_TEST_S3_STRICT=1; \
	$(S3_TEST_BIN) "[s3][integration]~[large]~[aws]"

s3-test-large:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-test-large: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@# Grouped by config (chunk-prewarm on vs off) so MinIO is brought up once per
	@# group. Catch2 OR-combines specs within one argument via commas (multiple
	@# positional args are AND-concatenated instead), so each group runs in a
	@# single process where same-config cases share one SiriusContext lifecycle.
	@set -e; \
	export SIRIUS_TEST_S3_AUTO=1 SIRIUS_TEST_S3_LARGE=1 SIRIUS_TEST_S3_STRICT=1; \
	$(S3_TEST_BIN) "[s3][sql][large][large-count],[s3][sql][large][large-q1],[s3][sql][large][large-join]"; \
	$(S3_TEST_BIN) "[s3][sql][large][large-count-no-prewarm],[s3][sql][large][large-q1-no-prewarm],[s3][sql][large][large-join-no-prewarm]"

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
	$(S3_TEST_BIN) "[s3][aws]"

s3-test-aws-sigv4:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-test-aws-sigv4: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@set -e; \
	export SIRIUS_TEST_S3_STRICT=1; \
	$(S3_TEST_BIN) "[s3][aws]~[broker]"

s3-test-aws-broker:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-test-aws-broker: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@set -e; \
	export SIRIUS_TEST_S3_STRICT=1; \
	$(S3_TEST_BIN) "[s3][aws][broker]"

# -----------------------------------------------------------------------------
# S3 perf benchmarks. All S3 benchmarks share the hidden [.][s3][bench] tag
# (real-AWS ones also [aws][live]); the selector follows SIRIUS_BENCH_BACKEND:
# minio (default) runs [s3][bench]~[aws], aws-s3 runs [s3][bench][aws]. The tag
# carries [.] (out of the default `make test`) and lacks [integration] (so the
# [s3] integration gate `make s3-test` never pulls it in). Default backend = minio:
# the harness auto-manages MinIO (SIRIUS_TEST_S3_AUTO=1), generates/uploads the
# SF10 lineitem fixture (SIRIUS_TEST_S3_LARGE=1), and SIRIUS_TEST_S3_STRICT=1
# makes a MinIO bring-up failure fail the benchmark loud instead of skipping to a
# false green. The JSON baseline emits
# build/release/extension/sirius/test/cpp/log/s3_rest_perf_<ts>.json and appends
# per-run history under doc/s3support/perf-history-<backend>.jsonl.
# Set SIRIUS_BENCH_BACKEND=aws-s3 to hit real AWS instead of MinIO (manual only —
# export the SIRIUS_TEST_S3_* env yourself: regional endpoint, bucket, assume-role
# TEMPORARY credentials incl. the session token; SIRIUS_BENCH_S3_KEY overrides the
# object key); AUTO/STRICT stay off in that case and MinIO is never started.

s3-bench:
	@if [ ! -x $(S3_TEST_BIN) ]; then \
	  echo "s3-bench: $(S3_TEST_BIN) not found - run \`make release\` first" >&2; \
	  exit 1; \
	fi
	@set -e; \
	if [ "$${SIRIUS_BENCH_BACKEND:-minio}" = "minio" ]; then \
	  export SIRIUS_TEST_S3_AUTO=1 SIRIUS_TEST_S3_LARGE=1 SIRIUS_TEST_S3_STRICT=1; \
	  selector="[s3][bench]~[aws]"; \
	else \
	  selector="[s3][bench][aws]"; \
	fi; \
	export SIRIUS_BENCH_GIT_SHA="$$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"; \
	export HOSTNAME="$${HOSTNAME:-$$(hostname)}"; \
	$(S3_TEST_BIN) "$$selector"
