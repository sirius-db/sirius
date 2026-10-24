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

#include "event/query_event_subscriber.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
bool has_window(std::string const& plan)
{
  return plan.find("WINDOW") != std::string::npos || plan.find("Window") != std::string::npos;
}

// Observe production task deployment instead of inferring activation from EXPLAIN alone.
class window_task_observer : public sirius::event::query_event_subscriber {
 public:
  explicit window_task_observer(sirius::event::query_event_publisher& publisher)
    : query_event_subscriber(publisher, {sirius::event::event_type::task_deployed}, true)
  {
    start();
  }
  ~window_task_observer() override { stop(); }
  std::string_view name() const noexcept override { return "window-sql-test"; }
  bool failed() const noexcept { return delivery_failed(); }
  std::size_t window_tasks = 0;
  void on_task_deployed(sirius::event::event_id_t,
                        sirius::event::timestamp_t,
                        sirius::query_id_t,
                        std::size_t,
                        sirius::op::SiriusPhysicalOperatorType type,
                        int) noexcept override
  {
    if (type == sirius::op::SiriusPhysicalOperatorType::WINDOW) { ++window_tasks; }
  }
};

class WindowFixture : public sirius::test::GpuExecutionFixture {
 public:
  WindowFixture()
  {
    run_ok(
      "CREATE TABLE w AS SELECT i::INTEGER id, (i%7)::INTEGER p, "
      "CASE WHEN i%11=0 THEN NULL ELSE (i%3)::INTEGER END q, "
      "CASE WHEN i%5=0 OR i%7=3 THEN NULL ELSE (i%17)::INTEGER END v, "
      "CASE WHEN i%5=0 THEN NULL ELSE (i%13*0.25)::DECIMAL(18,2) END d, "
      "(i%19*0.5)::DOUBLE f, (CASE WHEN i%3=0 THEN NULL ELSE 'key' END)::VARCHAR s "
      "FROM range(3000) r(i)");
    run_ok("CHECKPOINT");
    // Keep a real WINDOW in these regression plans. Production optimizer behavior is probed
    // separately below using the original TPC-DS queries and its default settings.
    auto original = con->Query("SELECT current_setting('disabled_optimizers')");
    REQUIRE_FALSE(original->HasError());
    original_optimizers = original->Collection().GetValue(0, 0).ToString();
    window_optimizers =
      original_optimizers.empty() ? "window_self_join" : original_optimizers + ",window_self_join";
    run_ok("SET disabled_optimizers='" + window_optimizers + "'");
    run_ok("SET hash_partition_bytes=1024");
  }
  ~WindowFixture() { con->Query("SET disabled_optimizers='" + original_optimizers + "'"); }
  std::string original_optimizers;
  std::string window_optimizers;
  sirius::test::scratch_dir files{"window_sql"};

  void compare_window(std::string const& sql,
                      std::size_t minimum_tasks           = 1,
                      std::set<size_t> const& approximate = {})
  {
    INFO(sql);
    run_ok("SET gpu_execution=false");
    auto plan = con->Query("EXPLAIN " + sql);
    REQUIRE_FALSE(plan->HasError());
    REQUIRE(has_window(plan->ToString()));
    run_ok("SET gpu_execution=true");
    window_task_observer observer(
      sirius::test::get_registered_sirius_context(*con)->get_event_publisher());
    if (approximate.empty()) {
      compare_gpu_vs_cpu(sql);
    } else {
      compare_gpu_vs_cpu_approx(sql, approximate);
    }
    observer.stop();
    REQUIRE_FALSE(observer.failed());
    REQUIRE(observer.window_tasks >= minimum_tasks);
  }
};
}  // namespace

TEST_CASE_METHOD(WindowFixture,
                 "whole-partition SQL windows retain row association and nulls",
                 "[integration][gpu_execution][window_sql]")
{
  compare_window(
    "SELECT id,p,q,v,sum(v) OVER (PARTITION BY p,q), "
    "avg(v) OVER (PARTITION BY q,p),sum(d) OVER (PARTITION BY p,q) FROM w",
    1,
    {5});
  compare_window("SELECT id,sum(v) OVER (),avg(d) OVER (),sum(f) OVER () FROM w", 1, {2, 3});
  compare_window(
    "SELECT id,sum(v+1) OVER (PARTITION BY p+1), "
    "avg(v+1) OVER (PARTITION BY p+1) FROM w",
    1,
    {2});
  compare_window("SELECT id,sum(v) OVER (PARTITION BY s,q) FROM w");
  compare_window(
    "SELECT id,sum(v) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) "
    "FROM w");
}

TEST_CASE_METHOD(WindowFixture,
                 "whole-partition count and extrema execute on GPU",
                 "[integration][gpu_execution][window_sql]")
{
  compare_window(
    "SELECT id,p,q,count(*) OVER (PARTITION BY p,q),count(v) OVER (PARTITION BY q,p),"
    "min(v) OVER (PARTITION BY p,q),max(v) OVER (PARTITION BY p,q),"
    "sum(v) OVER (PARTITION BY p,q),avg(v) OVER (PARTITION BY p,q) FROM w",
    2,
    {8});
  compare_window(
    "SELECT id,count(s) OVER (PARTITION BY p),min(s) OVER (PARTITION BY p),"
    "max(s) OVER (PARTITION BY p),min(d) OVER (PARTITION BY p),"
    "max(f) OVER (PARTITION BY p) FROM w");
  compare_window(
    "SELECT id,count(*) OVER (),count(v) OVER (),min(v) OVER (),max(v) OVER () FROM w");
  compare_window(
    "SELECT id,min(x) OVER (PARTITION BY p),max(x) OVER (PARTITION BY p),"
    "min(x) OVER (PARTITION BY p) FROM "
    "(SELECT id,p,CASE WHEN p=0 THEN 'NaN'::DOUBLE WHEN p=1 THEN NULL::DOUBLE "
    "WHEN id%5=0 THEN 'NaN'::DOUBLE ELSE f END x FROM w) t");
  compare_window(
    "SELECT id,min(x) OVER (),max(x) OVER (),count(x) OVER () FROM "
    "(SELECT id,CASE WHEN id%5=0 THEN 'NaN'::FLOAT ELSE f::FLOAT END x FROM w) t");
  compare_window("SELECT id,count(s) OVER (),min(s) OVER (),max(s) OVER () FROM w");
  // No payload columns remain, and COUNT(*) must still count the complete input.
  compare_window("SELECT count(*) OVER () FROM w");
  compare_window("SELECT count(*) OVER (PARTITION BY p) FROM w", 2);
  compare_window(
    "SELECT id,count(NULL::INTEGER) OVER (PARTITION BY p),"
    "min(NULL::INTEGER) OVER (PARTITION BY p),max(NULL::INTEGER) OVER (PARTITION BY p) FROM w");
  compare_window(
    "SELECT id,count(v) OVER (PARTITION BY p,q),min(v) OVER (PARTITION BY p,q),"
    "max(v) OVER (PARTITION BY p,q) FROM (SELECT * FROM w UNION ALL SELECT * FROM w) t",
    2);
  run_ok("CREATE TABLE window_types(id INTEGER,q INTEGER,d DATE,b BOOLEAN)");
  run_ok(
    "INSERT INTO window_types VALUES (0,0,DATE '2000-01-01',true),"
    "(1,0,DATE '1999-01-01',false),(2,NULL,NULL,NULL),(3,NULL,DATE '2001-01-01',NULL)");
  run_ok("CHECKPOINT");
  compare_window(
    "SELECT id,q,min(d) OVER (PARTITION BY q),max(d) OVER (PARTITION BY q),"
    "min(b) OVER (PARTITION BY q),max(b) OVER (PARTITION BY q),"
    "count(d) OVER (PARTITION BY q) FROM window_types");
  compare_gpu_vs_cpu(
    "SELECT count(*) OVER (),count(v) OVER (),min(v) OVER (),max(v) OVER () FROM w WHERE id<0");
}

TEST_CASE_METHOD(WindowFixture,
                 "whole-partition SQL windows assemble fragmented input and buckets",
                 "[integration][gpu_execution][window_sql]")
{
  compare_window(
    "SELECT id,p,sum(v) OVER (PARTITION BY p),avg(d) OVER (PARTITION BY p) "
    "FROM (SELECT * FROM w UNION ALL SELECT * FROM w) t",
    2,
    {3});
  compare_window(
    "SELECT id,sum(v) OVER (),avg(d) OVER () "
    "FROM (SELECT * FROM w UNION ALL SELECT * FROM w) t",
    1,
    {2});
}

TEST_CASE_METHOD(WindowFixture,
                 "whole-partition SQL windows preserve wide integer and decimal sums",
                 "[integration][gpu_execution][window_sql]")
{
  run_ok("CREATE TABLE wide(v BIGINT, d DECIMAL(38,2))");
  run_ok(
    "INSERT INTO wide VALUES (9223372036854775807,123456789012345678901234567890.12), "
    "(9223372036854775807,0.03), (NULL,NULL)");
  run_ok("CHECKPOINT");
  // Native DuckDB scans do not decode DECIMAL128 yet. Exercise this admitted evaluator input
  // through Parquet, as well as through a grouped SUM below.
  run_ok("COPY wide TO " + files.file_literal("wide.parquet") + " (FORMAT PARQUET)");
  run_ok("CREATE VIEW wide_p AS SELECT * FROM read_parquet(" + files.file_literal("wide.parquet") +
         ")");
  compare_window("SELECT v,d,sum(v) OVER (),sum(d) OVER (),avg(d) OVER () FROM wide_p", 1, {4});
  compare_window(
    "SELECT p,q,sum(sum(d)) OVER (PARTITION BY p), "
    "avg(sum(d)) OVER (PARTITION BY p) FROM w GROUP BY p,q",
    1,
    {3});
}

TEST_CASE_METHOD(WindowFixture,
                 "decimal window SUM rejects precision overflow",
                 "[integration][gpu_execution][window_sql]")
{
  run_ok("CREATE TABLE overflow_values(v DECIMAL(38,0))");
  run_ok(
    "INSERT INTO overflow_values VALUES (60000000000000000000000000000000000000), "
    "(60000000000000000000000000000000000000)");
  run_ok("CHECKPOINT");
  run_ok("COPY overflow_values TO " + files.file_literal("overflow.parquet") + " (FORMAT PARQUET)");
  run_ok("CREATE VIEW overflow_p AS SELECT * FROM read_parquet(" +
         files.file_literal("overflow.parquet") + ")");
  compare_window("SELECT v,count(v) OVER (),min(v) OVER (),max(v) OVER () FROM overflow_p");
  // AVG may use a sum beyond DECIMAL(38), provided its 128-bit accumulator still fits.
  compare_window("SELECT v,avg(v) OVER () FROM overflow_p", 1, {1});
  auto const sql = "SELECT sum(v) OVER () FROM overflow_p";
  run_ok("SET gpu_execution=false");
  auto cpu = con->Query(sql);
  REQUIRE(cpu->HasError());
  run_ok("SET gpu_execution=true");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  window_task_observer observer(
    sirius::test::get_registered_sirius_context(*con)->get_event_publisher());
  auto gpu = con->Query(sql);
  REQUIRE(gpu->HasError());
  CHECK(gpu->GetError().find("DECIMAL(38)") != std::string::npos);
  observer.stop();
  REQUIRE_FALSE(observer.failed());
  REQUIRE(observer.window_tasks > 0);
  auto const after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.fallbacks == before.fallbacks);
  // The existing read-only replay policy retries the terminal execution error on CPU,
  // which also raises precision overflow. Do not change that policy for window failures.
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
}

TEST_CASE_METHOD(WindowFixture,
                 "empty whole-partition SQL windows return zero rows",
                 "[integration][gpu_execution][window_sql]")
{
  // DuckDB may eliminate this window entirely, so no task-deployment claim is made here.
  compare_gpu_vs_cpu("SELECT id,sum(v) OVER (),avg(d) OVER () FROM w WHERE id<0");
}

TEST_CASE_METHOD(WindowFixture,
                 "whole-partition SQL windows compose with downstream expressions",
                 "[integration][gpu_execution][window_sql]")
{
  compare_window(
    "SELECT id,d/total AS ratio FROM "
    "(SELECT id,d,sum(d) OVER (PARTITION BY p) total FROM w) t WHERE total>0");
  compare_window(
    "SELECT id,total FROM (SELECT id,sum(v) OVER (PARTITION BY p) total FROM w) t "
    "ORDER BY total,id LIMIT 25");
}

TEST_CASE_METHOD(WindowFixture,
                 "unsupported SQL window semantics retain CPU fallback",
                 "[integration][gpu_execution][window_sql]")
{
  expect_plan_fallback_matches_cpu("SELECT id,sum(v) OVER (ORDER BY id) FROM w");
  expect_plan_fallback_matches_cpu("SELECT id,sum(DISTINCT v) OVER () FROM w");
  expect_plan_fallback_matches_cpu("SELECT id,count(DISTINCT v) OVER () FROM w");
  expect_plan_fallback_matches_cpu("SELECT sum(v::DECIMAL(4,0)) OVER () FROM w");
  expect_plan_fallback_matches_cpu(
    "SELECT sum(s) OVER (PARTITION BY p) FROM "
    "(SELECT p,sum(d) OVER (PARTITION BY q) s FROM w) t");
  expect_plan_fallback_matches_cpu("SELECT sum(v) OVER () + 1 FROM w");
  expect_plan_fallback_matches_cpu(
    "SELECT s,count(*) FROM (SELECT sum(v) OVER () s FROM w) t GROUP BY s");
  expect_plan_fallback_matches_cpu(
    "SELECT id FROM (SELECT id,sum(v) OVER () s FROM w) t WHERE s>0");
}

TEST_CASE_METHOD(WindowFixture,
                 "original TPC-DS whole-partition query probes",
                 "[integration][gpu_execution][window_tpcds]")
{
  auto const number         = GENERATE(12, 20, 53, 63, 89, 98);
  auto const root           = std::filesystem::path(__FILE__).parent_path();
  auto const* override_path = std::getenv("SIRIUS_TPCDS_TEST_DB_PATH");
  auto const database =
    override_path ? std::filesystem::path(override_path) : root / "data/duckdb/tpcds.duckdb";
  REQUIRE(std::filesystem::exists(database));
  run_ok("ATTACH IF NOT EXISTS '" + database.string() + "' AS tpcds (READ_ONLY)");
  run_ok("USE tpcds");
  std::ifstream file(root / "../../../duckdb/extension/tpcds/dsdgen/queries" /
                     (std::to_string(number) + ".sql"));
  REQUIRE(file.good());
  std::ostringstream text;
  text << file.rdbuf();
  auto const sql = text.str();
  INFO("TPC-DS Q" << number);
  auto const preserve_window = GENERATE(false, true);
  run_ok("SET disabled_optimizers='" + (preserve_window ? window_optimizers : original_optimizers) +
         "'");
  run_ok("SET gpu_execution=false");
  auto explain = con->Query("EXPLAIN " + sql);
  REQUIRE_FALSE(explain->HasError());
  auto const retains_window = has_window(explain->ToString());
  UNSCOPED_INFO("optimizer retains WINDOW: " << retains_window);
  auto expected = con->Query(sql);
  REQUIRE_FALSE(expected->HasError());
  auto const rows = expected->RowCount();
  UNSCOPED_INFO("TPC-DS result rows: " << rows);
  run_ok("SET gpu_execution=true");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  window_task_observer observer(
    sirius::test::get_registered_sirius_context(*con)->get_event_publisher());
  auto actual = con->Query(sql);
  REQUIRE(actual);
  if (actual->HasError()) { UNSCOPED_INFO(actual->GetError()); }
  REQUIRE_FALSE(actual->HasError());
  auto const after     = sirius::test::get_transparent_execution_stats(*con);
  auto const fell_back = after.fallbacks != before.fallbacks;
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  REQUIRE(actual->GetTypes() == expected->GetTypes());
  auto gpu_rows = collect_rows(*actual);
  auto cpu_rows = collect_rows(*expected);
  REQUIRE(gpu_rows.size() == cpu_rows.size());
  // Approximate only the aggregate/ratio cells, keeping row-identifying leading keys exact.
  auto const approximate = (number == 53 || number == 63) ? std::set<size_t>{2}
                           : number == 89                 ? std::set<size_t>{7}
                                                          : std::set<size_t>{6};
  for (size_t row = 0; row < gpu_rows.size(); ++row) {
    for (size_t column = 0; column < gpu_rows[row].size(); ++column) {
      REQUIRE(cells_equal(
        gpu_rows[row][column], cpu_rows[row][column], approximate.count(column) ? 1e-9 : 0.0));
    }
  }
  observer.stop();
  REQUIRE_FALSE(observer.failed());
  UNSCOPED_INFO("CPU fallback: " << fell_back
                                 << "; GPU WINDOW tasks deployed: " << observer.window_tasks);
  if (!fell_back && retains_window && rows > 0) { REQUIRE(observer.window_tasks > 0); }
}
