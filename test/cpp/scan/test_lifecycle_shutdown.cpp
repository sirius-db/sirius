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

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>

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

// Run a shell command, capturing merged stdout+stderr inline. Returns {raw_status, output}
// where raw_status is the value popen/pclose hands back (use W* macros to interpret).
std::pair<int, std::string> run_capture(std::string const& cmd)
{
  std::string const full = cmd + " 2>&1";
  FILE* pipe             = ::popen(full.c_str(), "r");
  if (!pipe) { return {-1, "popen() failed"}; }

  std::string output;
  char buf[4096];
  while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
    output.append(buf);
  }
  int const raw = ::pclose(pipe);
  return {raw, std::move(output)};
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

  // Sirius's default GPU memory pool grabs ~14 GB upfront, which OOMs on CI runners.
  // Point the spawned CLI at the small (2 GB GPU / 4 GB host) test config sitting next
  // to this test file. Strip SIRIUS_DISABLE (shared_test_env sets it; would block CLI
  // init) and SIRIUS_INTEGRATION_TEST_DB_PATH (TPC-H test data path, irrelevant here).
  auto const test_config = std::filesystem::path(__FILE__).parent_path() / "memory.yaml";
  REQUIRE(std::filesystem::exists(test_config));

  std::string const cmd =
    "env -u SIRIUS_DISABLE -u SIRIUS_INTEGRATION_TEST_DB_PATH "
    "SIRIUS_CONFIG_FILE=" +
    test_config.string() + " " + duckdb_bin +
    " -unsigned -c \""
    "CREATE TABLE t(s VARCHAR); "
    "INSERT INTO t SELECT 'x' FROM range(1000); "
    "CHECKPOINT; "
    "CALL gpu_execution('SELECT count(*) FROM t');"
    "\"";

  auto const [raw, output] = run_capture(cmd);
  INFO("command: " << cmd);
  INFO("subprocess merged stdout+stderr:\n" << output);

  if (WIFSIGNALED(raw)) {
    FAIL("duckdb CLI terminated by signal "
         << WTERMSIG(raw)
         << " (SIGSEGV = 11; CLI exit code = 139 indicates the QueryEnd UAF regressed)");
  }
  REQUIRE(WIFEXITED(raw));
  if (WEXITSTATUS(raw) != 0) {
    FAIL("duckdb CLI exited with status "
         << WEXITSTATUS(raw)
         << " (non-zero, non-signal). This usually means a setup precondition failed before "
            "the scan ran (init error, missing GPU, missing config). The subprocess output "
            "above contains the actual error from the CLI.");
  }
}
