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

#include <catch.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <optional>
#include <string>
#include <vector>

namespace {

class RequiredNullScanFixture : public sirius::test::GpuExecutionFixture {
 public:
  RequiredNullScanFixture()
  {
    run_ok("SET gpu_execution = false");
    run_ok(
      "CREATE TABLE required_null AS SELECT i::INTEGER AS id,"
      " CASE WHEN i % 3 = 0 THEN NULL WHEN i % 3 = 1 THEN '' ELSE 'x' END AS s,"
      " (i % 7)::INTEGER AS part, (i * 2)::INTEGER AS amount"
      " FROM range(6000) t(i)");
    run_ok(
      "CREATE TABLE all_null AS SELECT i::INTEGER AS id, NULL::VARCHAR AS s"
      " FROM range(32) t(i)");
    run_ok("COPY required_null TO " + sirius::test::sql_literal(dir.file("mixed.parquet")) +
           " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
    run_ok("COPY required_null TO " + sirius::test::sql_literal(dir.file("hive")) +
           " (FORMAT PARQUET, PARTITION_BY(part))");
    run_ok("COPY all_null TO " + sirius::test::sql_literal(dir.file("all_null.parquet")) +
           " (FORMAT PARQUET)");
    run_ok("CHECKPOINT");
    run_ok("SET gpu_execution = true");
  }

  ~RequiredNullScanFixture()
  {
    // Also clean up after a failed assertion, before the attached database is removed.
    if (pinned) { con->Query("CALL unpin_table('required_null')"); }
  }

  std::string scan(std::string const& backend)
  {
    if (backend == "parquet") {
      return "read_parquet(" + sirius::test::sql_literal(dir.file("mixed.parquet")) + ")";
    }
    if (backend == "hive") {
      return "read_parquet(" + sirius::test::sql_literal(dir.file("hive/part=*/*.parquet")) +
             ", hive_partitioning=true)";
    }
    if (backend == "host" || backend == "gpu") {
      run_ok("CALL pin_table(format='duckdb', name='required_null', tier='" + backend + "')");
      pinned = true;
    }
    return "required_null";
  }

  sirius::test::scratch_dir dir{"required_null"};
  bool pinned = false;
};

}  // namespace

TEST_CASE_METHOD(RequiredNullScanFixture,
                 "required NULL rejection survives scan filtering and projection",
                 "[integration][gpu_execution][scan][required_null]")
{
  std::string const backend = GENERATE("native", "parquet", "hive", "host", "gpu");
  // Multi-file hive scans can retain DuckDB's unsupported constant_or_null helper.
  // Keep the original string predicate executable there; ordinary native/parquet
  // scans below still exercise the optimizer's normal IS NOT NULL rewrite.
  std::optional<sirius::test::disabled_optimizers_guard> hive_predicates;
  if (backend == "hive") { hive_predicates.emplace(*con, "expression_rewriter"); }
  auto const source = scan(backend);
  for (std::string const predicate :
       {"s IS NOT NULL", "s LIKE '%'", "suffix(s, '')", "contains(s, '')"}) {
    CAPTURE(backend, predicate);
    auto const from = " FROM " + source + " WHERE " + predicate;
    // Every comparison asserts one actual GPU execution and zero CPU fallbacks.
    // Empty strings must survive; the mixed groups cannot be answered by pruning alone.
    compare_gpu_vs_cpu("SELECT s" + from);
    compare_gpu_vs_cpu("SELECT *" + from);
    compare_gpu_vs_cpu_ordered("SELECT amount, id" + from + " ORDER BY id");
    compare_gpu_vs_cpu("SELECT COUNT(*), SUM(amount)" + from);
    compare_gpu_vs_cpu("SELECT part, COUNT(*), SUM(amount)" + from + " GROUP BY part");
    compare_gpu_vs_cpu("SELECT id" + from + " AND amount >= 40 AND id < 90");
    compare_gpu_vs_cpu("SELECT amount, s, id" + from + " AND part = 2");
  }
}

TEST_CASE_METHOD(RequiredNullScanFixture,
                 "required NULL rejection handles all-NULL scan input",
                 "[integration][gpu_execution][scan][required_null]")
{
  // Keep an executable predicate instead of EMPTY_RESULT. With statistics propagation
  // disabled alone, the string rewrites leave an unsupported constant_or_null helper.
  // The mixed-input test above exercises the normal rewrites to IS NOT NULL.
  sirius::test::disabled_optimizers_guard const keep_scan{
    *con, "statistics_propagation,expression_rewriter"};
  std::string const backend = GENERATE("native", "parquet");
  auto const source =
    backend == "native"
      ? std::string("all_null")
      : "read_parquet(" + sirius::test::sql_literal(dir.file("all_null.parquet")) + ")";
  for (std::string const predicate :
       {"s IS NOT NULL", "s LIKE '%'", "suffix(s, '')", "contains(s, '')"}) {
    CAPTURE(backend, predicate);
    compare_gpu_vs_cpu("SELECT id FROM " + source + " WHERE " + predicate);
    compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + source + " WHERE " + predicate);
  }
}
