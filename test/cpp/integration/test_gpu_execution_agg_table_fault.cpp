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

/**
 * @file test_gpu_execution_agg_table_fault.cpp
 * @brief Task replay and failure with the table path of GROUP BY off and on, end to end.
 *
 * An out-of-memory inside a pipeline task replays the task with the same input
 * (gpu_pipeline_task::compute_task -> oom_reschedule_exception). A fold that changed the table
 * before the exception must not be applied a second time. Three hooks are used, all test-only
 * settings read under SIRIUS_ENABLE_TEST_OPTIONS=1, which the unit-test binary sets:
 *
 *  - `sirius_test_inject_agg_table_fold_oom = N`: the first N table folds of each grouped
 *    aggregate throw rmm::out_of_memory after the fold was applied. This is the case that tests
 *    exactly-once folding. The cases that use it skip while the setting is not registered.
 *  - `sirius_test_inject_gpu_task_oom = N` and `sirius_test_inject_gpu_task_launch_error = N`:
 *    the first N operator executions of the query throw BEFORE the operator runs. The counter is
 *    query-wide and the first operator of a scan pipeline is the scan, so these replay the tasks
 *    that feed the aggregate, never a fold that already changed the table: they show that the
 *    table path survives rescheduled input, not that a fold is applied once.
 *  - `sirius_test_gpu_task_retry_limit = 1` with a large OOM count: the query fails on the GPU
 *    and DuckDB replays it on the CPU. The table must be released on that path, which the
 *    finalize hook does not cover, and the next query must find the engine intact.
 */

#include <catch.hpp>
#include <utils/agg_table_test_utils.hpp>

#include <cstdint>
#include <string>

using sirius::test::scoped_sirius_setting;
using sirius::test::agg_table::AggTableFixture;
using sirius::test::agg_table::kBudgetSetting;
using sirius::test::agg_table::kEnableSetting;
using sirius::test::agg_table::kFoldOomSetting;
using sirius::test::agg_table::kScanBatch;
using sirius::test::agg_table::setting_exists;

#define AGG_TABLE_FAULT_TAGS "[integration][gpu_execution][aggregate][agg_table][agg_table_fault]"

namespace {

constexpr char kTaskOomSetting[]      = "sirius_test_inject_gpu_task_oom";
constexpr char kLaunchErrorSetting[]  = "sirius_test_inject_gpu_task_launch_error";
constexpr char kRetryLimitSetting[]   = "sirius_test_gpu_task_retry_limit";
constexpr char kRetryBackoffSetting[] = "sirius_test_gpu_task_retry_backoff_ms";

/// Five row groups, so five scan batches with one batch per row group. Every batch holds every
/// ordinary key, the reserved keys -1 and -2, NULL keys, a hot key and values that pass INT64 when
/// summed, so a fold applied twice changes a count, a sum, and the wide half of a sum.
void create_fault_tables(AggTableFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE ft AS SELECT CASE WHEN i % 40 = 0 THEN -1 WHEN i % 40 = 1 THEN -2 "
    "  WHEN i % 40 = 2 THEN NULL WHEN i % 4 = 3 THEN 7 ELSE hash(i) % 5000 END::BIGINT AS k, "
    "(i % 1000)::BIGINT AS v, "
    "CASE WHEN i % 5 = 0 THEN 4000000000000000000 ELSE i % 100 END::BIGINT AS big, "
    "((i % 9000) / 100.0)::DECIMAL(12,2) AS p FROM range(600000) r(i)");
  fx.run_ok("CHECKPOINT");
}

void check_fault_queries(AggTableFixture& fx)
{
  fx.on_off("SELECT k, count(*) AS c, sum(v) AS s, min(v) AS mn, max(v) AS mx FROM ft GROUP BY k");
  fx.on_off("SELECT k, count(*) AS c, sum(big) AS s, sum(p) AS sp FROM ft GROUP BY k");
  fx.on_off(
    "SELECT count(*) AS groups, sum(c) AS n, sum(s) AS s FROM (SELECT k, count(*) AS c, "
    "sum(v) AS s FROM ft GROUP BY k) t");
  fx.on_off_approx("SELECT k, avg(v) AS a, avg(p) AS ap FROM ft GROUP BY k", {1, 2}, 1e-12);
}

}  // namespace

//===----------------------------------------------------------------------===//
// A fold replayed after it changed the table
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(
  AggTableFixture,
  "agg table fault - a fold that ran out of memory after it was applied is applied once",
  AGG_TABLE_FAULT_TAGS)
{
  sirius::test::agg_table::require_enable_setting(*con);
  if (!setting_exists(*con, kFoldOomSetting)) {
    SKIP("setting '" << kFoldOomSetting
                     << "' is not registered: no hook replays a fold after it changed the table");
  }
  create_fault_tables(*this);
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  scoped_sirius_setting const no_backoff{*con, kRetryBackoffSetting, std::uint64_t{0}};
  // One fold, every fold of a five-batch input, and more replays than there are batches (each
  // replayed fold counts again).
  for (std::uint64_t const folds :
       {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{5}, std::uint64_t{12}}) {
    INFO(kFoldOomSetting << " = " << folds);
    scoped_sirius_setting const fold_oom{*con, kFoldOomSetting, folds};
    check_fault_queries(*this);
  }
}

TEST_CASE_METHOD(AggTableFixture,
                 "agg table fault - replayed folds with a table too small for the groups",
                 AGG_TABLE_FAULT_TAGS)
{
  sirius::test::agg_table::require_enable_setting(*con);
  if (!setting_exists(*con, kFoldOomSetting) || !setting_exists(*con, kBudgetSetting)) {
    SKIP("settings '" << kFoldOomSetting << "' and '" << kBudgetSetting << "' are not registered");
  }
  // A replayed batch whose first attempt applied some rows and returned the others for the old
  // path: neither half may be counted twice.
  create_fault_tables(*this);
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  scoped_sirius_setting const no_backoff{*con, kRetryBackoffSetting, std::uint64_t{0}};
  for (std::uint64_t const budget : {std::uint64_t{64} << 10, std::uint64_t{1}}) {
    INFO(kBudgetSetting << " = " << budget);
    scoped_sirius_setting const table_budget{*con, kBudgetSetting, budget};
    for (std::uint64_t const folds : {std::uint64_t{1}, std::uint64_t{5}}) {
      INFO(kFoldOomSetting << " = " << folds);
      scoped_sirius_setting const fold_oom{*con, kFoldOomSetting, folds};
      check_fault_queries(*this);
    }
  }

  // The budgets above are below the smallest table the library builds (about 2^18 slots), so no
  // table exists there and every row takes the old path. For a table that exists and overflows:
  // 16 MiB against 700000 groups of 40 bytes of state and more, in random order, each key in
  // several batches.
  run_ok(
    "CREATE TABLE ft_many AS SELECT CASE WHEN i % 40 = 2 THEN NULL WHEN i % 4 = 3 THEN 7 "
    "  ELSE hash(i // 3) % 700000 END::BIGINT AS k, (i % 1000)::BIGINT AS v, "
    "CASE WHEN i % 5 = 0 THEN 4000000000000000000 ELSE i % 100 END::BIGINT AS big "
    "FROM range(2400000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const table_budget{*con, kBudgetSetting, std::uint64_t{16} << 20};
  for (std::uint64_t const folds : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{7}}) {
    INFO(kBudgetSetting << " = 16 MiB, " << kFoldOomSetting << " = " << folds);
    scoped_sirius_setting const fold_oom{*con, kFoldOomSetting, folds};
    auto const before = sirius::test::agg_table::table_stats();
    on_off(
      "SELECT count(*) AS groups, sum(c) AS n, sum(s) AS s, sum(sb) AS sb, sum(mx - mn) AS d "
      "FROM (SELECT k, count(*) AS c, sum(v) AS s, sum(big) AS sb, min(v) AS mn, max(v) AS mx "
      "FROM ft_many GROUP BY k) t");
    on_off(
      "SELECT k, count(*) AS c, sum(v) AS s, sum(big) AS sb FROM ft_many GROUP BY k "
      "HAVING k IS NULL OR k = 7 OR k % 4999 = 0");
    // The table took rows and gave rows back: this is the case the test is named for.
    auto const delta = sirius::test::agg_table::table_stats().since(before);
    CHECK(delta.folded_batches > 0);
    CHECK(delta.returned_rows + delta.returned_partial_rows > 0);
    if (folds > 0) { CHECK(delta.replayed_folds > 0); }
  }
}

//===----------------------------------------------------------------------===//
// Tasks replayed before their operator ran
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(
  AggTableFixture,
  "agg table fault - tasks replayed after an injected out-of-memory feed the table once",
  AGG_TABLE_FAULT_TAGS)
{
  create_fault_tables(*this);
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  scoped_sirius_setting const no_backoff{*con, kRetryBackoffSetting, std::uint64_t{0}};
  for (std::uint64_t const attempts :
       {std::uint64_t{1}, std::uint64_t{3}, std::uint64_t{7}, std::uint64_t{40}}) {
    INFO(kTaskOomSetting << " = " << attempts);
    scoped_sirius_setting const task_oom{*con, kTaskOomSetting, attempts};
    check_fault_queries(*this);
  }
}

TEST_CASE_METHOD(
  AggTableFixture,
  "agg table fault - tasks replayed after an injected launch failure feed the table once",
  AGG_TABLE_FAULT_TAGS)
{
  create_fault_tables(*this);
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  scoped_sirius_setting const no_backoff{*con, kRetryBackoffSetting, std::uint64_t{0}};
  for (std::uint64_t const attempts : {std::uint64_t{1}, std::uint64_t{4}}) {
    INFO(kLaunchErrorSetting << " = " << attempts);
    scoped_sirius_setting const launch_error{*con, kLaunchErrorSetting, attempts};
    check_fault_queries(*this);
  }
}

//===----------------------------------------------------------------------===//
// A query that fails on the GPU
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(AggTableFixture,
                 "agg table fault - a query that exhausts its retries is replayed on the CPU and "
                 "leaves the engine intact",
                 AGG_TABLE_FAULT_TAGS)
{
  sirius::test::agg_table::require_enable_setting(*con);
  create_fault_tables(*this);
  std::string const query =
    "SELECT k, count(*) AS c, sum(v) AS s, sum(big) AS sb, min(v) AS mn FROM ft GROUP BY k";
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  scoped_sirius_setting const no_backoff{*con, kRetryBackoffSetting, std::uint64_t{0}};

  run_ok("SET gpu_execution = false");
  auto cpu_result = con->Query(query);
  run_ok("SET gpu_execution = true");
  REQUIRE(cpu_result);
  REQUIRE_FALSE(cpu_result->HasError());
  auto const cpu_rows = collect_rows(*cpu_result, true);

  for (bool const enabled : {false, true}) {
    INFO(kEnableSetting << " = " << (enabled ? "true" : "false"));
    {
      scoped_sirius_setting const table_path{*con, kEnableSetting, enabled};
      scoped_sirius_setting const retry_limit{*con, kRetryLimitSetting, std::uint64_t{1}};
      scoped_sirius_setting const task_oom{*con, kTaskOomSetting, std::uint64_t{1000}};
      auto const before = sirius::test::get_transparent_execution_stats(*con);
      auto result       = con->Query(query);
      auto const after  = sirius::test::get_transparent_execution_stats(*con);
      REQUIRE(result);
      if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
      REQUIRE_FALSE(result->HasError());
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
      REQUIRE(result->GetTypes() == cpu_result->GetTypes());
      REQUIRE(collect_rows(*result, true) == cpu_rows);
    }
    // The failed query's tables and their share of the GPU are gone: the same aggregations run on
    // the GPU again, with the table path off and on.
    check_fault_queries(*this);
  }
}
