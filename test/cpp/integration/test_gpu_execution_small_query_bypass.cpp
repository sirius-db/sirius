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
 * @file test_gpu_execution_small_query_bypass.cpp
 * @brief End-to-end correctness tests for the small-query bypass (issue #990).
 *
 * Primary Super Sirius coverage for this feature (Catch2 / GpuExecutionFixture).
 * Pipeline structure assertions live in test_small_query_bypass_converter.cpp.
 *
 * By default, queries whose summed base-scan bytes fall under the 256 MiB
 * `small_query_bytes_threshold` skip the partition stages (PARTITION,
 * SORT_SAMPLE, SORT_PARTITION) at plan-generation time. These tests run a battery of
 * queries over tiny tables — every one of them takes the bypass — through
 * transparent GPU execution and compare against DuckDB CPU results. The battery
 * deliberately covers the shapes whose correctness depends on the retained
 * terminal operators: AVG and COUNT(DISTINCT) finalization (MERGE_GROUP_BY),
 * OFFSET (MERGE_TOP_N), non-inner joins (build CONCAT fold), and an ORDER BY
 * whose projection drops the sort key (MERGE_SORT final projection).
 *
 * A final test re-runs a subset with the threshold at 0 (bypass disabled) to
 * confirm identical behavior on the normal partitioned path.
 *
 * Known limits of the gate (see operator_params::small_query_bytes_threshold):
 * leaf-scan byte sum (no join blow-up model), CTE/delim excluded,
 * STANDARD joins only (no BUILD_PROBE / dynamic filters on the bypass path).
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

class SmallQueryBypassExecFixture : public sirius::test::GpuExecutionFixture {
 public:
  void create_test_tables()
  {
    run_ok("CREATE TABLE dim (id INTEGER, name VARCHAR);");
    run_ok("INSERT INTO dim VALUES (1, 'a'), (2, 'b'), (3, 'c');");
    run_ok("CREATE TABLE fact (dim_id INTEGER, qty INTEGER, price DOUBLE);");
    run_ok("INSERT INTO fact VALUES (1, 10, 1.5), (1, 20, 2.5), (2, 30, 3.5), (4, 40, 4.5);");
    run_ok("CHECKPOINT;");
  }

  void set_threshold_256mb() { run_ok("SET small_query_bytes_threshold = 268435456;"); }

  void set_threshold_disabled() { run_ok("SET small_query_bytes_threshold = 0;"); }
};

}  // namespace

TEST_CASE_METHOD(SmallQueryBypassExecFixture,
                 "small-query bypass exec - aggregates (AVG, COUNT DISTINCT)",
                 "[integration][gpu_execution][small_query_bypass]")
{
  create_test_tables();
  set_threshold_256mb();

  // AVG partial states (SUM + COUNT) are finalized in MERGE_GROUP_BY.
  compare_gpu_vs_cpu(
    "SELECT dim_id, AVG(qty), COUNT(*), SUM(qty) FROM fact GROUP BY dim_id ORDER BY dim_id;");
  // COUNT(DISTINCT) collect-lists are finalized in MERGE_GROUP_BY. (Ungrouped distinct aggregates
  // are unsupported on the GPU path — UNGROUPED_AGGREGATE, not MERGE_GROUP_BY — and orthogonal to
  // the bypass, so only the grouped case is exercised here.)
  compare_gpu_vs_cpu("SELECT dim_id, COUNT(DISTINCT qty) FROM fact GROUP BY dim_id;");
  // Ungrouped aggregate (chain already minimal; must stay correct with the flag on).
  compare_gpu_vs_cpu("SELECT SUM(qty), AVG(price) FROM fact;");

  set_threshold_disabled();
}

TEST_CASE_METHOD(SmallQueryBypassExecFixture,
                 "small-query bypass exec - order by and top-n with offset",
                 "[integration][gpu_execution][small_query_bypass]")
{
  create_test_tables();
  set_threshold_256mb();

  // Projection drops the sort key: MERGE_SORT applies the final projection.
  compare_gpu_vs_cpu("SELECT qty FROM fact ORDER BY dim_id DESC, qty DESC;");
  compare_gpu_vs_cpu("SELECT dim_id, qty FROM fact ORDER BY qty;");
  // OFFSET is applied only in MERGE_TOP_N.
  compare_gpu_vs_cpu("SELECT qty FROM fact ORDER BY qty LIMIT 2 OFFSET 1;");
  compare_gpu_vs_cpu("SELECT qty FROM fact ORDER BY qty DESC LIMIT 10 OFFSET 2;");

  set_threshold_disabled();
}

TEST_CASE_METHOD(SmallQueryBypassExecFixture,
                 "small-query bypass exec - join types",
                 "[integration][gpu_execution][small_query_bypass]")
{
  create_test_tables();
  set_threshold_256mb();

  // INNER
  compare_gpu_vs_cpu(
    "SELECT d.name, f.qty FROM fact f JOIN dim d ON f.dim_id = d.id ORDER BY f.qty;");
  // LEFT (unmatched probe rows must survive — needs the concat_all build fold)
  compare_gpu_vs_cpu(
    "SELECT f.qty, d.name FROM fact f LEFT JOIN dim d ON f.dim_id = d.id ORDER BY f.qty;");
  // RIGHT (DuckDB plans it as a flipped LEFT join)
  compare_gpu_vs_cpu(
    "SELECT d.name, f.qty FROM fact f RIGHT JOIN dim d ON f.dim_id = d.id "
    "ORDER BY d.name, f.qty;");
  // SEMI via IN
  compare_gpu_vs_cpu("SELECT qty FROM fact WHERE dim_id IN (SELECT id FROM dim) ORDER BY qty;");
  // ANTI via NOT IN (dim.id has no NULLs)
  compare_gpu_vs_cpu("SELECT qty FROM fact WHERE dim_id NOT IN (SELECT id FROM dim) ORDER BY qty;");
  // MARK join (IN as a projected boolean)
  compare_gpu_vs_cpu("SELECT id, id IN (SELECT dim_id FROM fact) FROM dim ORDER BY id;");

  set_threshold_disabled();
}

TEST_CASE_METHOD(SmallQueryBypassExecFixture,
                 "small-query bypass exec - multi-join and join + aggregate",
                 "[integration][gpu_execution][small_query_bypass]")
{
  create_test_tables();
  run_ok("CREATE TABLE dim2 (id INTEGER, tag VARCHAR);");
  run_ok("INSERT INTO dim2 VALUES (1, 'x'), (2, 'y');");
  run_ok("CHECKPOINT;");
  set_threshold_256mb();

  // Two joins: one is mid-pipeline (split_intermediate_joins bypass path).
  compare_gpu_vs_cpu(
    "SELECT f.qty, d.name, d2.tag FROM fact f "
    "JOIN dim d ON f.dim_id = d.id JOIN dim2 d2 ON f.dim_id = d2.id ORDER BY f.qty;");
  // Join feeding a grouped aggregate feeding an order-by: chains all bypass shapes.
  compare_gpu_vs_cpu(
    "SELECT d.name, AVG(f.qty) AS a, COUNT(*) FROM fact f "
    "JOIN dim d ON f.dim_id = d.id GROUP BY d.name ORDER BY a DESC;");

  set_threshold_disabled();
}

TEST_CASE_METHOD(SmallQueryBypassExecFixture,
                 "small-query bypass exec - correlated subquery is excluded and falls back to CPU",
                 "[integration][gpu_execution][small_query_bypass]")
{
  create_test_tables();
  set_threshold_256mb();

  // A correlated scalar subquery plans as a delim join whose inner join is a JoinType::SINGLE.
  // That (a) makes the whole query ineligible for the bypass (delim exclusion), and (b) is
  // unsupported by sirius_physical_concat (SINGLE), so GPU plan generation throws and the query
  // falls back to DuckDB CPU at plan time -- still returning correct results. Assert the graceful
  // plan-time fallback: not rebound, never GPU-executed, no runtime failure.
  const std::string query =
    "SELECT d.id, (SELECT COUNT(*) FROM fact f WHERE f.dim_id = d.id) FROM dim d ORDER BY d.id;";
  con->Query("SET gpu_execution = true;");
  const auto before = sirius::test::get_transparent_execution_stats(*con);
  auto result       = con->Query(query);
  const auto after  = sirius::test::get_transparent_execution_stats(*con);
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  CHECK(after.successful_rebinds == before.successful_rebinds);  // not rebound onto the GPU
  CHECK(after.executions == before.executions);                  // never GPU-executed
  CHECK(after.fallbacks > before.fallbacks);                     // plan-time fallback to CPU
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);    // not a runtime failure

  set_threshold_disabled();
}

TEST_CASE_METHOD(SmallQueryBypassExecFixture,
                 "small-query bypass exec - threshold 0 disables the bypass",
                 "[integration][gpu_execution][small_query_bypass]")
{
  create_test_tables();
  set_threshold_disabled();

  // Same shapes on the normal partitioned path: results must be identical.
  compare_gpu_vs_cpu(
    "SELECT dim_id, AVG(qty), COUNT(DISTINCT qty) FROM fact GROUP BY dim_id ORDER BY dim_id;");
  compare_gpu_vs_cpu(
    "SELECT f.qty, d.name FROM fact f LEFT JOIN dim d ON f.dim_id = d.id ORDER BY f.qty;");
  compare_gpu_vs_cpu("SELECT qty FROM fact ORDER BY qty LIMIT 2 OFFSET 1;");
}
