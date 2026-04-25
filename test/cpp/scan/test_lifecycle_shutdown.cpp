/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// test
#include <catch.hpp>

// standard library
#include <sys/wait.h>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

// Locate the freshly-built duckdb CLI. Falls back to the standard build/release layout
// derived from __FILE__; in non-standard layouts set SIRIUS_TEST_DUCKDB_BIN.
std::string find_duckdb_binary()
{
  if (const char* env = std::getenv("SIRIUS_TEST_DUCKDB_BIN")) {
    if (std::filesystem::exists(env)) { return env; }
  }
  auto const repo_root = std::filesystem::path(__FILE__)
                           .parent_path()   // scan/
                           .parent_path()   // cpp/
                           .parent_path()   // test/
                           .parent_path();  // <repo>
  for (auto const& candidate :
       {repo_root / "build/release/duckdb", repo_root / "build/debug/duckdb"}) {
    if (std::filesystem::exists(candidate)) { return candidate.string(); }
  }
  return {};
}

}  // namespace

// In-process tests can't reproduce this bug because both make_test_db_and_connection paths
// destroy SiriusContext before DatabaseInstance (the safe order). The duckdb CLI's exit
// chain inverts that order: ~DatabaseInstance runs while task_creator_ still holds a
// unique_ptr<GlobalTableFunctionState> referencing storage objects (BlockMemory,
// RowGroups). Pre-fix, that races with storage teardown and SIGSEGVs (exit 139).
TEST_CASE("scan lifecycle - clean exit after duckdb_scan_task query (regression: QueryEnd UAF)",
          "[scan][lifecycle][shutdown]")
{
  auto const duckdb_bin = find_duckdb_binary();
  if (duckdb_bin.empty()) {
    WARN("duckdb binary not found; set SIRIUS_TEST_DUCKDB_BIN to enable this test");
    SUCCEED();
    return;
  }

  // Strip env vars the unit-test harness sets but the spawned CLI must not inherit:
  //   SIRIUS_DISABLE          — set by shared_test_env::create_db; would block CLI init
  //   SIRIUS_CONFIG_FILE      — points at the test-only memory.yaml
  //   SIRIUS_INTEGRATION_*    — TPC-H test data path
  std::string const cmd =
    "env -u SIRIUS_DISABLE -u SIRIUS_CONFIG_FILE -u SIRIUS_INTEGRATION_TEST_DB_PATH " + duckdb_bin +
    " -unsigned -c \""
    "CREATE TABLE t(s VARCHAR); "
    "INSERT INTO t SELECT 'x' FROM range(1000); "
    "CHECKPOINT; "
    "CALL gpu_execution('SELECT count(*) FROM t');"
    "\" >/tmp/sirius_lifecycle_test.out 2>&1";

  int const raw = std::system(cmd.c_str());
  INFO("command: " << cmd);
  if (WIFSIGNALED(raw)) {
    FAIL("duckdb CLI terminated by signal "
         << WTERMSIG(raw)
         << " (SIGSEGV = 11; CLI exit code = 139 indicates the QueryEnd UAF regressed)");
  }
  REQUIRE(WIFEXITED(raw));
  REQUIRE(WEXITSTATUS(raw) == 0);
}
