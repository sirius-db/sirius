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

#pragma once

/**
 * @file gpu_execution_fixture.hpp
 * @brief Shared Catch2 fixture for end-to-end "run plain SQL on GPU, compare to
 *        DuckDB CPU" integration tests over the GPU DuckDB-native scan path.
 *
 * The native scan reads raw on-disk blocks through a SingleFileBlockManager, so
 * tables under test must live in a single-file (on-disk) database. The shared
 * integration connection is `:memory:`, so the fixture ATTACHes a fresh,
 * uniquely-named file-backed database and USEs it; unqualified CREATE/SELECT
 * then resolve against it. Persist with `CHECKPOINT` after loading data so the
 * native scan can see it on disk.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace sirius::test {

/// Collects stringified result rows, sorted by default for order-insensitive comparison;
/// pass `sort = false` to preserve emitted order (e.g. to verify ORDER BY).
inline std::vector<std::vector<std::string>> collect_rows(duckdb::MaterializedQueryResult& result,
                                                          bool sort = true)
{
  std::vector<std::vector<std::string>> rows;
  for (duckdb::idx_t r = 0; r < result.RowCount(); r++) {
    std::vector<std::string> row;
    row.reserve(result.ColumnCount());
    for (duckdb::idx_t c = 0; c < result.ColumnCount(); c++) {
      row.push_back(result.GetValue(c, r).ToString());
    }
    rows.push_back(std::move(row));
  }
  if (sort) { std::sort(rows.begin(), rows.end()); }
  return rows;
}

/// RAII guard that points Sirius at a config file for the lifetime of a fixture
/// that spins up its own (non-shared) host database.
struct sirius_config_env_guard {
  explicit sirius_config_env_guard(const std::string& config_path)
  {
    setenv("SIRIUS_CONFIG_FILE", config_path.c_str(), 1);
  }
  ~sirius_config_env_guard() { unsetenv("SIRIUS_CONFIG_FILE"); }
};

/**
 * @brief File-backed DuckDB connection plus a GPU-vs-CPU result comparator.
 *
 * When a shared integration SiriusContext is active we borrow its connection;
 * otherwise we spin up a private in-memory host DB pointed at integration.yaml.
 * Either way a fresh on-disk database is ATTACHed and USEd so the GPU native
 * scan has real blocks to read.
 */
class GpuExecutionFixture {
 public:
  GpuExecutionFixture()
  {
    namespace fs = std::filesystem;

    // Unique file + attach alias per fixture so sequential tests on the shared
    // connection never collide, even if a prior test aborted before teardown.
    static std::atomic<unsigned> counter{0};
    auto const id = counter.fetch_add(1);
    temp_db_path  = (fs::temp_directory_path() / ("sirius_gpu_test_" + std::to_string(getpid()) +
                                                 "_" + std::to_string(id) + ".db"))
                     .string();
    attach_alias = "gpu_db_" + std::to_string(getpid()) + "_" + std::to_string(id);

    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      // integration.yaml lives in test/cpp/integration/; this header is in
      // test/cpp/utils/, so step up to test/cpp and back down.
      auto cfg_path =
        fs::path(__FILE__).parent_path().parent_path() / "integration" / "integration.yaml";
      REQUIRE(fs::exists(cfg_path));
      config_guard = std::make_unique<sirius_config_env_guard>(cfg_path.string());
      db           = std::make_unique<duckdb::DuckDB>(nullptr);  // in-memory host DB
      con          = std::make_unique<duckdb::Connection>(*db);
    }

    // Route all subsequent DDL/DML/queries into the on-disk database so the
    // native GPU scan can read it via SingleFileBlockManager.
    run_ok("ATTACH '" + temp_db_path + "' AS " + attach_alias + ";");
    run_ok("USE " + attach_alias + ";");
  }

  ~GpuExecutionFixture()
  {
    namespace fs = std::filesystem;
    if (con) {
      // Can't DETACH the in-use database; switch back to the default in-memory
      // catalog first. A poisoned connection may reject these.
      con->Query("USE memory;");
      con->Query("DETACH " + attach_alias + ";");
    }
    con.reset();
    db.reset();
    if (!temp_db_path.empty() && fs::exists(temp_db_path)) {
      fs::remove(temp_db_path);
      // Remove .wal file if it exists
      auto wal_path = temp_db_path + ".wal";
      if (fs::exists(wal_path)) { fs::remove(wal_path); }
    }
  }

  void run_ok(const std::string& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("setup query error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
  }

  /// @copydoc sirius::test::collect_rows
  static std::vector<std::vector<std::string>> collect_rows(duckdb::MaterializedQueryResult& result,
                                                            bool sort = true)
  {
    return sirius::test::collect_rows(result, sort);
  }

  /// Cell equality used by the comparator. Exact string match by default; when
  /// `rel_tol > 0` and both cells parse fully as numbers, they match within a
  /// relative tolerance (with an absolute floor near zero), so floating-point
  /// aggregation-order differences don't register as mismatches. A partial
  /// numeric parse ("12abc") or a non-number ("NULL") is never approximate.
  /// Exposed as a static so it can be unit-tested directly.
  static bool cells_equal(const std::string& a, const std::string& b, double rel_tol)
  {
    if (a == b) { return true; }
    if (rel_tol <= 0.0) { return false; }
    // std::stod is C-locale sensitive; correct under CI's default C locale (DuckDB
    // prints numerics with '.'). Use std::from_chars if that assumption changes.
    try {
      std::size_t pa  = 0;
      std::size_t pb  = 0;
      double const da = std::stod(a, &pa);
      double const db = std::stod(b, &pb);
      // Reject partial parses ("12abc", "NULL") so only fully-numeric cells are
      // compared approximately.
      if (pa != a.size() || pb != b.size()) { return false; }
      // Non-finite values (inf / -inf / NaN) only match via exact string equality
      // (already handled above). The tolerance test below would otherwise treat
      // inf vs a finite or opposite-sign value as equal, since inf <= rel_tol*inf.
      if (!std::isfinite(da) || !std::isfinite(db)) { return false; }
      double const diff  = std::fabs(da - db);
      double const scale = std::max(std::fabs(da), std::fabs(db));
      return diff <= rel_tol * scale || diff <= rel_tol;
    } catch (...) {
      return false;
    }
  }

  /// GPU-vs-CPU comparison that ignores row order (rows are sorted first). Use
  /// for filters, joins, aggregates, and any query without an ORDER BY.
  void compare_gpu_vs_cpu(const std::string& query) { compare_gpu_vs_cpu_impl(query, false); }

  /// GPU-vs-CPU comparison that preserves row order. Use for queries with a
  /// deterministic output order (ORDER BY / LIMIT) so NULLS FIRST|LAST placement
  /// is actually verified rather than sorted away.
  void compare_gpu_vs_cpu_ordered(const std::string& query)
  {
    compare_gpu_vs_cpu_impl(query, true);
  }

  /// GPU-vs-CPU comparison with a relative tolerance applied ONLY to the columns
  /// named in `approx_cols` (0-based). Those columns match within `rel_tol` (with
  /// an absolute floor near zero); every other column -- keys, counts, strings,
  /// NULLs -- still compares exactly. Use for floating-point aggregation (SUM/AVG
  /// over DECIMAL/DOUBLE), where GPU parallel reduction and CPU serial summation
  /// legitimately differ in the low bits. Rows are matched order-insensitively by
  /// sorting stringified rows, which is only valid when the leading column(s) form
  /// an exact, unique key -- so `approx_cols` must NOT include column 0 of a
  /// multi-row result (enforced at runtime). NOTE: not
  /// for aggregates prone to catastrophic cancellation (e.g. sums of signed values
  /// that nearly cancel) -- there the two summation orders can diverge beyond any
  /// meaningful tolerance.
  void compare_gpu_vs_cpu_approx(const std::string& query,
                                 const std::set<size_t>& approx_cols,
                                 double rel_tol = 1e-6)
  {
    compare_gpu_vs_cpu_impl(query, false, approx_cols, rel_tol);
  }

  /// Asserts the query does NOT run purely on the GPU: it triggers a runtime
  /// fallback to DuckDB CPU (which still produces the correct result). Use for
  /// operations that are unsupported on-device -- this is a coverage fact, not a
  /// result divergence, so it is a normal passing assertion rather than a
  /// [!shouldfail] case.
  void expect_gpu_fallback(const std::string& query)
  {
    con->Query("SET gpu_execution = true;");
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto result = con->Query(query);
    auto after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("execution error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    if (after.runtime_fallbacks <= before.runtime_fallbacks) {
      UNSCOPED_INFO("expected a runtime fallback to CPU, but none occurred");
    }
    REQUIRE(after.runtime_fallbacks > before.runtime_fallbacks);
  }

  /// Asserts the query is rejected during GPU plan generation and completes via DuckDB's CPU
  /// plan, returning exactly the CPU results. Distinct from expect_gpu_fallback, which asserts a
  /// fallback *after* GPU execution started: a plan-time rejection moves `fallbacks` and never
  /// increments `executions`. Use for shapes Sirius screens out at plan time on purpose.
  void expect_plan_fallback_matches_cpu(const std::string& query)
  {
    run_ok("SET gpu_execution = true;");
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto gpu_result   = con->Query(query);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) { UNSCOPED_INFO("query error: " << gpu_result->GetError()); }
    REQUIRE_FALSE(gpu_result->HasError());
    if (after.fallbacks == before.fallbacks) {
      UNSCOPED_INFO("expected a plan-time fallback to CPU, but none occurred");
    }
    REQUIRE(after.fallbacks == before.fallbacks + 1);
    REQUIRE(after.executions == before.executions);

    run_ok("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    run_ok("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    REQUIRE_FALSE(cpu_result->HasError());

    REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());
    auto gpu_rows = collect_rows(gpu_result->Cast<duckdb::MaterializedQueryResult>(), true);
    auto cpu_rows = collect_rows(cpu_result->Cast<duckdb::MaterializedQueryResult>(), true);
    REQUIRE(gpu_rows == cpu_rows);
  }

 private:
  void compare_gpu_vs_cpu_impl(const std::string& query,
                               bool ordered,
                               const std::set<size_t>& approx_cols = {},
                               double rel_tol                      = 0.0)
  {
    // Run on GPU (transparent, plain SQL goes through the Sirius optimizer hook).
    con->Query("SET gpu_execution = true;");
    auto before_gpu_stats = sirius::test::get_transparent_execution_stats(*con);

    auto gpu_result = con->Query(query);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) {
      UNSCOPED_INFO("transparent GPU execution error: " << gpu_result->GetError());
    }
    REQUIRE_FALSE(gpu_result->HasError());
    auto after_gpu_stats = sirius::test::get_transparent_execution_stats(*con);
    // Exactly one GPU execution, no fallback: proves the query ran on the GPU.
    sirius::test::require_transparent_execution_delta(before_gpu_stats, after_gpu_stats, 1, 0, 1);

    // Run on CPU.
    con->Query("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    con->Query("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    REQUIRE_FALSE(cpu_result->HasError());

    REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

    auto& gpu_mat = gpu_result->Cast<duckdb::MaterializedQueryResult>();
    auto& cpu_mat = cpu_result->Cast<duckdb::MaterializedQueryResult>();
    // For ordered queries, keep emitted order so NULLS FIRST|LAST is verified;
    // otherwise sort both sides for an order-insensitive multiset comparison.
    auto gpu_rows = collect_rows(gpu_mat, !ordered);
    auto cpu_rows = collect_rows(cpu_mat, !ordered);

    // Order-insensitive approx matching pairs rows by their sorted stringified
    // values, so an approximate *leading* key (col 0) could misalign rows. Guard
    // that misuse: with multiple rows, col 0 must be compared exactly.
    if (!ordered && !approx_cols.empty() && gpu_rows.size() > 1) {
      REQUIRE(approx_cols.count(0) == 0);
    }

    for (size_t r = 0; r < gpu_rows.size(); r++) {
      for (size_t c = 0; c < gpu_rows[r].size(); c++) {
        // Only explicitly-approximate columns get the tolerance; keys, counts,
        // and everything else compare exactly (tol = 0).
        double const tol = approx_cols.count(c) != 0 ? rel_tol : 0.0;
        bool const match = cells_equal(gpu_rows[r][c], cpu_rows[r][c], tol);
        if (!match) {
          UNSCOPED_INFO("Row " << r << " Col " << c << " mismatch: GPU=[" << gpu_rows[r][c]
                               << "] CPU=[" << cpu_rows[r][c] << "]");
        }
        REQUIRE(match);
      }
    }
  }

 public:
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
  std::string temp_db_path;
  std::string attach_alias;
  std::unique_ptr<sirius_config_env_guard> config_guard;
};

}  // namespace sirius::test
