/*
 * Copyright 2026, Sirius Contributors.
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
 * @file agg_table_test_utils.hpp
 * @brief Shared helpers for the tests of the table path of GROUP BY (`src/cuda/agg_table`,
 *        `src/op/aggregate/table_aggregate*`).
 *
 * Every check runs one query three times on one connection: on DuckDB's CPU execution, on the GPU
 * with the table path off (today's cudf::groupby -> PARTITION -> merge), and on the GPU with it
 * on. Both GPU runs must be real GPU executions with no fallback and must return the CPU's column
 * types and rows; the two GPU runs are also compared with each other.
 *
 * The names of the engine settings live here and nowhere else.
 */

#include "op/aggregate/table_aggregate.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace sirius::test::agg_table {

// ---------------------------------------------------------------------------------------------
// Engine names. These match what src/sirius_extension.cpp registers at the time of writing
// (operator_params::agg_table_*); if a name changes, this is the one place to follow it.
// ---------------------------------------------------------------------------------------------

/// BOOLEAN setting that turns the table path of GROUP BY on. Off by default.
inline constexpr char kEnableSetting[] = "agg_table_enabled";
/// UBIGINT setting: cap in bytes on one table, 0 = no cap. Forces "rows that do not fit go down
/// the old path" and the flush at a batch boundary. Tests that need it skip while it is not
/// registered.
inline constexpr char kBudgetSetting[] = "agg_table_budget_bytes";
/// DOUBLE setting in (0, 1]: share of one GPU's memory a table may hold.
inline constexpr char kMemoryFractionSetting[] = "agg_table_memory_fraction";
/// UBIGINT test-only setting (registered under SIRIUS_ENABLE_TEST_OPTIONS=1, like
/// sirius_test_inject_gpu_task_oom): the first N table folds of each grouped aggregate throw
/// rmm::out_of_memory AFTER the fold was applied and before the task recorded its result, so the
/// task is replayed with the same batch. Tests that need it skip while it is not registered.
inline constexpr char kFoldOomSetting[] = "sirius_test_inject_agg_table_fold_oom";

/// Existing engine settings the tests use to shape the input.
inline constexpr char kScanBatch[]      = "scan_task_batch_size";
inline constexpr char kPartitionBytes[] = "hash_partition_bytes";
inline constexpr char kConcatBytes[]    = "concat_batch_bytes";

/// Whether @p name is a registered setting on @p con's database.
inline bool setting_exists(duckdb::Connection& con, std::string const& name)
{
  auto result = con.Query("SELECT count(*) FROM duckdb_settings() WHERE name = '" + name + "';");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  return result->Collection().GetValue(0, 0).ToString() != "0";
}

/// Fails with a message that names the fix when the table path's switch is not registered.
inline void require_enable_setting(duckdb::Connection& con)
{
  if (!setting_exists(con, kEnableSetting)) {
    FAIL("setting '" << kEnableSetting
                     << "' is not registered: set kEnableSetting in "
                        "test/cpp/utils/agg_table_test_utils.hpp to the name the engine registers");
  }
}

/**
 * @brief How many batches the table path has folded in this process so far
 *        (`sirius::op::table_aggregate_process_stats`). Process-wide and monotonic: the
 *        difference across one query says whether that query folded into a table.
 */
inline std::optional<std::uint64_t> table_fold_count(duckdb::Connection& /*con*/)
{
  return sirius::op::table_aggregate_process_stats().folded_batches;
}

/// All the table path's counters; `after.since(before)` is what one query did, including the
/// number of GPUs whose table folded a batch (`gpus_folded()`).
inline sirius::op::table_aggregate_stats table_stats()
{
  return sirius::op::table_aggregate_process_stats();
}

/// Appends what one query did on the table path to the file named by the environment variable
/// SIRIUS_AGG_TABLE_PATH_LOG, when it is set: one tab-separated line per GPU run of a check (test
/// case, setting, folded batches, folded rows, rows returned to the cuDF path or as partial rows,
/// batches that skipped the table, replayed folds, boundary flushes, end-of-input flushes, GPUs
/// that folded). A run of the suite with it set shows how many checks really went through a table.
inline void log_table_path(bool enabled, sirius::op::table_aggregate_stats const& delta)
{
  char const* path = std::getenv("SIRIUS_AGG_TABLE_PATH_LOG");
  if (path == nullptr || *path == '\0') { return; }
  std::ofstream out(path, std::ios::app);
  out << Catch::getResultCapture().getCurrentTestName() << '\t' << (enabled ? "on" : "off") << '\t'
      << delta.folded_batches << '\t' << delta.folded_rows << '\t'
      << delta.returned_rows + delta.returned_partial_rows << '\t' << delta.legacy_batches << '\t'
      << delta.replayed_folds << '\t' << delta.boundary_flushes << '\t'
      << delta.end_of_input_flushes << '\t' << delta.gpus_folded() << '\n';
}

/// Replaces every "$1" in @p sql with @p value.
inline std::string subst(std::string sql, std::string const& value)
{
  for (std::size_t pos = 0; (pos = sql.find("$1", pos)) != std::string::npos;) {
    sql.replace(pos, 2, value);
    pos += value.size();
  }
  return sql;
}

/// Which GROUP BY path a check expects with the table path on.
enum class expected_path {
  any,     ///< either: the contract leaves the choice to the engine
  table,   ///< the table path must fold at least one batch
  legacy,  ///< the table path must fold nothing (ineligible keys or aggregates)
};

struct compare_options {
  bool ordered   = false;  ///< keep emitted order (queries with a total ORDER BY)
  bool canonical = false;  ///< spell -nan as nan and drop the sign of a zero before comparing
  std::set<std::size_t> approx_cols{};  ///< 0-based columns compared with rel_tol
  bool approx_float_types = false;      ///< also compare every FLOAT / DOUBLE column with rel_tol
  double rel_tol          = 0.0;
  expected_path path      = expected_path::any;
  /// The existing path is known to return wrong rows for this query: with the table path off
  /// only the run itself, its column types and its row count are required, and a difference from
  /// the CPU is reported with WARN. With the table path on the CPU's rows are required as usual.
  bool legacy_known_wrong = false;
};

namespace detail {

using rows_t = std::vector<std::vector<std::string>>;

inline rows_t prepared_rows(duckdb::QueryResult& result, compare_options const& options)
{
  auto rows = sirius::test::collect_rows(result, false);
  if (options.canonical) {
    for (auto& row : rows) {
      for (auto& cell : row) {
        cell = GpuExecutionFixture::canonical_float_key(std::move(cell));
      }
    }
  }
  if (!options.ordered) { std::sort(rows.begin(), rows.end()); }
  return rows;
}

/// The first cell in which @p lhs and @p rhs differ, described, or nullopt when they agree.
inline std::optional<std::string> first_difference(rows_t const& lhs,
                                                   rows_t const& rhs,
                                                   std::set<std::size_t> const& approx_cols,
                                                   double rel_tol,
                                                   char const* lhs_name,
                                                   char const* rhs_name)
{
  if (lhs.size() != rhs.size()) {
    return std::string{lhs_name} + " has " + std::to_string(lhs.size()) + " rows, " + rhs_name +
           " has " + std::to_string(rhs.size());
  }
  for (std::size_t r = 0; r < lhs.size(); r++) {
    if (lhs[r].size() != rhs[r].size()) { return std::string{"column counts differ"}; }
    for (std::size_t c = 0; c < lhs[r].size(); c++) {
      double const tol = approx_cols.count(c) != 0 ? rel_tol : 0.0;
      if (!GpuExecutionFixture::cells_equal(lhs[r][c], rhs[r][c], tol)) {
        return "Row " + std::to_string(r) + " Col " + std::to_string(c) + " mismatch: " + lhs_name +
               "=[" + lhs[r][c] + "] " + rhs_name + "=[" + rhs[r][c] + "]";
      }
    }
  }
  return std::nullopt;
}

inline void require_same_rows(rows_t const& lhs,
                              rows_t const& rhs,
                              std::set<std::size_t> const& approx_cols,
                              double rel_tol,
                              char const* lhs_name,
                              char const* rhs_name)
{
  auto const difference = first_difference(lhs, rhs, approx_cols, rel_tol, lhs_name, rhs_name);
  if (difference) { UNSCOPED_INFO(*difference); }
  REQUIRE_FALSE(difference.has_value());
}

}  // namespace detail

/**
 * @brief Runs @p query on the CPU, on the GPU with the table path off and on the GPU with it on,
 *        and requires the three results to agree.
 *
 * Each GPU run must be exactly one GPU execution with no plan-time or runtime fallback and must
 * return the CPU's column types.
 */
inline void compare_on_off(duckdb::Connection& con,
                           std::string const& query,
                           compare_options const& options = {})
{
  INFO(query);
  require_enable_setting(con);

  con.Query("SET gpu_execution = false;");
  auto cpu_result = con.Query(query);
  con.Query("SET gpu_execution = true;");
  REQUIRE(cpu_result);
  if (cpu_result->HasError()) { UNSCOPED_INFO("CPU error: " << cpu_result->GetError()); }
  REQUIRE_FALSE(cpu_result->HasError());
  auto const cpu_rows = detail::prepared_rows(*cpu_result, options);

  auto approx_cols = options.approx_cols;
  if (options.approx_float_types) {
    auto const& types = cpu_result->GetTypes();
    for (std::size_t c = 0; c < types.size(); c++) {
      if (types[c].id() == duckdb::LogicalTypeId::FLOAT ||
          types[c].id() == duckdb::LogicalTypeId::DOUBLE) {
        approx_cols.insert(c);
      }
    }
  }
  // Unordered rows are paired by sorting their text, so an approximate leading column could
  // pair the wrong rows.
  if (!options.ordered && !approx_cols.empty() && cpu_rows.size() > 1) {
    REQUIRE(approx_cols.count(0) == 0);
  }

  detail::rows_t gpu_rows[2];
  for (bool const enabled : {false, true}) {
    INFO(kEnableSetting << " = " << (enabled ? "true" : "false"));
    scoped_sirius_setting const table_path{con, kEnableSetting, enabled};
    auto const folds_before = table_fold_count(con);
    auto const stats_before = table_stats();
    auto const before       = sirius::test::get_transparent_execution_stats(con);
    auto gpu_result         = con.Query(query);
    auto const after        = sirius::test::get_transparent_execution_stats(con);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) { UNSCOPED_INFO("GPU error: " << gpu_result->GetError()); }
    REQUIRE_FALSE(gpu_result->HasError());
    CHECK(after.read_view_mismatches == before.read_view_mismatches);
    // Exactly one GPU execution and no fallback: the rows below came from the GPU.
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);

    REQUIRE(gpu_result->GetTypes() == cpu_result->GetTypes());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());
    gpu_rows[enabled] = detail::prepared_rows(*gpu_result, options);
    if (enabled || !options.legacy_known_wrong) {
      detail::require_same_rows(gpu_rows[enabled],
                                cpu_rows,
                                approx_cols,
                                options.rel_tol,
                                enabled ? "GPU(table on)" : "GPU(table off)",
                                "CPU");
    } else if (auto const difference = detail::first_difference(gpu_rows[enabled],
                                                                cpu_rows,
                                                                approx_cols,
                                                                options.rel_tol,
                                                                "GPU(table off)",
                                                                "CPU")) {
      WARN("known defect of the existing path, not required to match: " << *difference << " in "
                                                                        << query);
    }

    log_table_path(enabled, table_stats().since(stats_before));
    if (auto const folds_after = table_fold_count(con); folds_before && folds_after) {
      bool const folded = *folds_after > *folds_before;
      if (!enabled || options.path == expected_path::legacy) { CHECK_FALSE(folded); }
      if (enabled && options.path == expected_path::table) { CHECK(folded); }
    }
  }
  if (!options.legacy_known_wrong) {
    detail::require_same_rows(
      gpu_rows[1], gpu_rows[0], approx_cols, options.rel_tol, "GPU(table on)", "GPU(table off)");
  }
}

/**
 * @brief Requires @p query to be rejected at GPU plan time with the table path off and on, and to
 *        return the CPU's types and rows through DuckDB's own plan both times.
 */
inline void compare_on_off_plan_fallback(duckdb::Connection& con, std::string const& query)
{
  INFO(query);
  require_enable_setting(con);

  con.Query("SET gpu_execution = false;");
  auto cpu_result = con.Query(query);
  con.Query("SET gpu_execution = true;");
  REQUIRE(cpu_result);
  REQUIRE_FALSE(cpu_result->HasError());
  auto const cpu_rows = sirius::test::collect_rows(*cpu_result, true);

  for (bool const enabled : {false, true}) {
    INFO(kEnableSetting << " = " << (enabled ? "true" : "false"));
    scoped_sirius_setting const table_path{con, kEnableSetting, enabled};
    auto const before = sirius::test::get_transparent_execution_stats(con);
    auto result       = con.Query(query);
    auto const after  = sirius::test::get_transparent_execution_stats(con);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    REQUIRE(after.fallbacks == before.fallbacks + 1);
    REQUIRE(after.executions == before.executions);
    REQUIRE(result->GetTypes() == cpu_result->GetTypes());
    REQUIRE(sirius::test::collect_rows(*result, true) == cpu_rows);
  }
}

/**
 * @brief GpuExecutionFixture (a fresh file-backed database on the shared one-GPU integration
 *        connection) with the three-way comparison as member functions.
 *
 * Member names are what scripts and reviewers grep for: every SQL string a test checks is the
 * first argument of an `on_off*` call or of `run_ok`.
 */
class AggTableFixture : public GpuExecutionFixture {
 public:
  /// Rows as a multiset, every cell exact.
  void on_off(std::string const& query, expected_path path = expected_path::any)
  {
    compare_on_off(*con, query, {.path = path});
  }

  /// Rows in emitted order, every cell exact. For queries whose ORDER BY is total.
  void on_off_ordered(std::string const& query, expected_path path = expected_path::any)
  {
    compare_on_off(*con, query, {.ordered = true, .path = path});
  }

  /// Rows as a multiset; @p approx_cols within @p rel_tol, every other cell exact.
  void on_off_approx(std::string const& query,
                     std::set<std::size_t> const& approx_cols,
                     double rel_tol     = 1e-6,
                     expected_path path = expected_path::any)
  {
    compare_on_off(*con, query, {.approx_cols = approx_cols, .rel_tol = rel_tol, .path = path});
  }

  /// Rows as a multiset after spelling -nan as nan and dropping the sign of a zero. For MIN / MAX
  /// over floats, where -0.0 and +0.0 are equal in SQL and either may represent the group.
  void on_off_canonical(std::string const& query, expected_path path = expected_path::any)
  {
    compare_on_off(*con, query, {.canonical = true, .path = path});
  }

  /// on_off_canonical for a query the existing path is known to answer wrongly (NaN in a float
  /// MIN or MAX): the CPU's rows are required with the table path on, and only reported with it
  /// off.
  void on_off_nan(std::string const& query, expected_path path = expected_path::any)
  {
    compare_on_off(*con, query, {.canonical = true, .path = path, .legacy_known_wrong = true});
  }

  /// The query must be rejected at GPU plan time either way and still return the CPU's rows.
  void on_off_fallback(std::string const& query) { compare_on_off_plan_fallback(*con, query); }
};

}  // namespace sirius::test::agg_table
