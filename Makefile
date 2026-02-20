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
	test test_release test_debug test_reldebug clean list-presets

all: release

release:
	cd $(DUCKDB_DIR) && $(CMAKE) --preset release
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset release
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset release --target $(TEST_BUILD_TARGET)
endif

debug:
	cd $(DUCKDB_DIR) && $(CMAKE) --preset debug
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset debug
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset debug --target $(TEST_BUILD_TARGET)
endif

reldebug: relwithdebinfo

debug-release: relwithdebinfo

relwithdebinfo:
	cd $(DUCKDB_DIR) && $(CMAKE) --preset relwithdebinfo
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset relwithdebinfo
ifneq ($(TEST_BUILD_TARGET),)
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset relwithdebinfo --target $(TEST_BUILD_TARGET)
endif

clang-release:
	cd $(DUCKDB_DIR) && $(CMAKE) --preset clang-release
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-release

clang-debug:
	cd $(DUCKDB_DIR) && $(CMAKE) --preset clang-debug
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-debug

clang-relwithdebinfo:
	cd $(DUCKDB_DIR) && $(CMAKE) --preset clang-relwithdebinfo
	cd $(DUCKDB_DIR) && $(CMAKE) --build --preset clang-relwithdebinfo

test: test_release

test_release: release
	./$(TEST_PATH) "$(CURDIR)/test/*"

test_debug: debug
	./$(TEST_PATH_DEBUG) "$(CURDIR)/test/*"

test_reldebug: relwithdebinfo
	./$(TEST_PATH_RELWITHDEBINFO) "$(CURDIR)/test/*"

clean:
	rm -rf build

list-presets:
	cd $(DUCKDB_DIR) && $(CMAKE) --list-presets
