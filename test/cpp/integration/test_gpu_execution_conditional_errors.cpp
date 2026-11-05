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

// CASE, COALESCE, and AND and OR in filters must not raise errors for rows a branch or operand
// does not see (#2203, #2052), while DuckDB's errors for every row still raise.
// compare_gpu_vs_cpu requires one GPU execution and no fallback, which is disabled besides.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/types/timestamp.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <cstdint>
#include <limits>
#include <string>

namespace {

class ConditionalErrorsFixture : public sirius::test::GpuExecutionFixture {
 public:
  void strategy(std::string const& value)
  {
    run_ok("SET gpu_execution = false");
    run_ok("SET enable_duckdb_fallback = false");
    run_ok("SET expression_evaluator_strategy = '" + value + "'");
  }

  // Requires DuckDB and the GPU to both fail @p query with @p message, without a fallback.
  void expect_error(std::string const& query, std::string const& message)
  {
    INFO(query);
    for (bool gpu : {false, true}) {
      run_ok(std::string("SET gpu_execution = ") + (gpu ? "true" : "false"));
      auto const before = sirius::test::get_transparent_execution_stats(*con);
      auto result       = con->Query(query);
      auto const after  = sirius::test::get_transparent_execution_stats(*con);
      REQUIRE(result);
      REQUIRE(result->HasError());
      INFO(result->GetError());
      REQUIRE(result->GetError().find(message) != std::string::npos);
      REQUIRE(after.fallbacks == before.fallbacks);
      REQUIRE(after.runtime_fallbacks == before.runtime_fallbacks);
      if (gpu) { REQUIRE(after.successful_rebinds == before.successful_rebinds + 1); }
    }
  }
};

}  // namespace

TEST_CASE_METHOD(ConditionalErrorsFixture,
                 "CASE skips DATE arithmetic errors in unselected branches",
                 "[integration][gpu_execution][conditional_errors]")
{
  strategy(GENERATE("materialize", "ast_interpret", "ast_jit"));
  // The issue's reproduction, verbatim.
  run_ok("CREATE TABLE issue(a DATE, n INTEGER)");
  run_ok("INSERT INTO issue VALUES ('2000-01-01', 2147483647), ('2000-01-01', 1)");
  // Adding n to a overflows on id 1. Ids 4 and 5 hold NULLs.
  run_ok("CREATE TABLE dates(id INTEGER, a DATE, n INTEGER)");
  run_ok(
    "INSERT INTO dates VALUES (1, '2000-01-01', 2147483647), (2, '2000-01-01', 1), "
    "(3, '2000-01-01', -5), (4, '2000-01-01', NULL), (5, NULL, 1)");
  run_ok("CHECKPOINT");

  compare_gpu_vs_cpu("SELECT CASE WHEN n = 1 THEN a + n ELSE a END FROM issue");
  // Mixed rows; a NULL condition selects ELSE.
  compare_gpu_vs_cpu("SELECT id, CASE WHEN n < 100 THEN a + n ELSE a END FROM dates");
  // The first matching WHEN wins: later branches overflow on id 1.
  compare_gpu_vs_cpu(
    "SELECT id, CASE WHEN n = 2147483647 THEN a WHEN n > 0 THEN a + n ELSE a - n END FROM dates");
  // A later WHEN condition overflows only on a row an earlier WHEN matched.
  compare_gpu_vs_cpu(
    "SELECT id, CASE WHEN n = 2147483647 THEN 0 WHEN a + n > a THEN 1 ELSE 2 END FROM dates");
  // The outer WHEN excludes the row on which the inner WHEN selects the overflow.
  compare_gpu_vs_cpu(
    "SELECT id, CASE WHEN n < 100 THEN CASE WHEN n > 0 THEN a + n ELSE a END ELSE a END "
    "FROM dates");
  // CASE beneath an aggregate and in a filter.
  compare_gpu_vs_cpu(
    "SELECT count(CASE WHEN n < 100 THEN a + n END), max(CASE WHEN n < 100 THEN a + n ELSE a END) "
    "FROM dates");
  compare_gpu_vs_cpu(
    "SELECT id FROM dates WHERE CASE WHEN n < 100 THEN a + n ELSE a END > DATE '2000-01-01'");
}

TEST_CASE_METHOD(ConditionalErrorsFixture,
                 "CASE and COALESCE skip negation and abs errors in unselected rows",
                 "[integration][gpu_execution][conditional_errors]")
{
  strategy(GENERATE("materialize", "ast_interpret", "ast_jit"));
  // #2128: every INTEGER minimum has a non-NULL a.
  run_ok("CREATE TABLE ints(id INTEGER, i INTEGER, a INTEGER)");
  run_ok(
    "INSERT INTO ints VALUES (1, -2147483648, 5), (2, 1, NULL), (3, -2147483648, 7), "
    "(4, NULL, NULL), (5, 3, NULL)");
  run_ok("CHECKPOINT");

  compare_gpu_vs_cpu("SELECT id, CASE WHEN i = -2147483648 THEN NULL ELSE -i END FROM ints");
  compare_gpu_vs_cpu("SELECT id, COALESCE(a, -i) FROM ints");
  compare_gpu_vs_cpu("SELECT id, CASE WHEN i > -2147483648 THEN abs(i) ELSE 0 END FROM ints");
  compare_gpu_vs_cpu(
    "SELECT id, CASE WHEN i = -2147483648 THEN 'min' ELSE CAST(-i AS VARCHAR) END FROM ints");
  // DuckDB never evaluates a branch that no row reaches, here a hand-written error() that the GPU
  // cannot evaluate at all.
  compare_gpu_vs_cpu(
    "SELECT id, CASE WHEN i < 0 THEN 0 WHEN i = -2147483648 THEN error('boom') ELSE 1 END "
    "FROM ints");
}

TEST_CASE_METHOD(ConditionalErrorsFixture,
                 "CASE and COALESCE skip timestamp cast errors in unselected rows",
                 "[integration][gpu_execution][conditional_errors]")
{
  strategy(GENERATE("materialize", "ast_interpret", "ast_jit"));
  // Midnight of year 300000 overflows TIMESTAMP's microseconds.
  run_ok("CREATE TABLE far(id INTEGER, d DATE)");
  run_ok("INSERT INTO far VALUES (1, DATE '2000-01-01'), (2, DATE '300000-01-01')");
  // #2052's reproduction. A TIMESTAMP_S literal past TIMESTAMP's range does not parse, so the
  // second row is bound as a value just past it.
  run_ok("CREATE TABLE timestamps(id INTEGER, t TIMESTAMP_S, p TIMESTAMP)");
  run_ok("INSERT INTO timestamps VALUES (0, TIMESTAMP_S '2024-01-01 00:00:00', NULL)");
  auto statement =
    con->Prepare("INSERT INTO timestamps VALUES (1, ?, TIMESTAMP '2024-01-01 00:00:00')");
  REQUIRE_FALSE(statement->HasError());
  auto const seconds = std::numeric_limits<int64_t>::max() / 1000000 + 1;
  auto const inserted =
    statement->Execute(duckdb::Value::TIMESTAMPSEC(duckdb::timestamp_sec_t{seconds}));
  REQUIRE_FALSE(inserted->HasError());
  run_ok("CHECKPOINT");

  compare_gpu_vs_cpu("SELECT id, CASE WHEN id = 1 THEN CAST(d AS TIMESTAMP) END FROM far");
  compare_gpu_vs_cpu(
    "SELECT id, CASE WHEN id = 0 THEN CAST(t AS TIMESTAMP) ELSE NULL END FROM timestamps");
  compare_gpu_vs_cpu("SELECT id, COALESCE(p, CAST(t AS TIMESTAMP)) FROM timestamps");
}

TEST_CASE_METHOD(ConditionalErrorsFixture,
                 "CASE and COALESCE still raise errors for rows that select the branch",
                 "[integration][gpu_execution][conditional_errors]")
{
  strategy(GENERATE("materialize", "ast_interpret", "ast_jit"));
  run_ok("CREATE TABLE dates(id INTEGER, a DATE, n INTEGER)");
  run_ok("INSERT INTO dates VALUES (1, '2000-01-01', 2147483647), (2, '2000-01-01', 1)");
  run_ok("CREATE TABLE ints(id INTEGER, i INTEGER, a INTEGER)");
  run_ok("INSERT INTO ints VALUES (1, -2147483648, NULL), (2, 1, 5)");
  run_ok("CHECKPOINT");

  expect_error("SELECT CASE WHEN n > 0 THEN a + n ELSE a END FROM dates", "Date out of range");
  expect_error("SELECT CASE WHEN n = 1 THEN a ELSE a + n END FROM dates", "Date out of range");
  expect_error(
    "SELECT CASE WHEN n > 0 THEN CASE WHEN n > 1 THEN a + n ELSE a END ELSE a END FROM dates",
    "Date out of range");
  expect_error("SELECT COALESCE(a, -i) FROM ints", "Overflow in negation");
}

TEST_CASE_METHOD(ConditionalErrorsFixture,
                 "AND and OR in filters and WHENs skip errors on rows earlier operands decide",
                 "[integration][gpu_execution][conditional_errors]")
{
  strategy(GENERATE("materialize", "ast_interpret", "ast_jit"));
  // n // (b - c) overflows on id 1, where b - c = -1. Each guard reads two columns, so DuckDB keeps
  // it in the filter instead of pushing it into the scan.
  run_ok("CREATE TABLE t(id INTEGER, n INTEGER, b INTEGER, c INTEGER)");
  run_ok(
    "INSERT INTO t VALUES (1, -2147483648, 0, 1), (2, 12, 5, 2), (3, 7, NULL, 1), (4, 9, 4, 1), "
    "(5, 2, 3, 1)");
  run_ok("CHECKPOINT");

  compare_gpu_vs_cpu("SELECT id FROM t WHERE b - c <> -1 AND n // (b - c) > 1");
  compare_gpu_vs_cpu("SELECT id FROM t WHERE b - c = -1 OR n // (b - c) > 1");
  compare_gpu_vs_cpu("SELECT id FROM t WHERE id > 0 AND (b - c = -1 OR n // (b - c) > 1)");
  compare_gpu_vs_cpu(
    "SELECT id FROM t GROUP BY id HAVING max(b) - max(c) <> -1 AND max(n) // (max(b) - max(c)) > "
    "1");
  compare_gpu_vs_cpu(
    "SELECT id, CASE WHEN b - c <> -1 AND n // (b - c) > 1 THEN 'big' ELSE 'small' END FROM t");
}

TEST_CASE_METHOD(ConditionalErrorsFixture,
                 "aggregate FILTER and AND in a value raise errors for every row, as in DuckDB",
                 "[integration][gpu_execution][conditional_errors]")
{
  strategy(GENERATE("materialize", "ast_interpret", "ast_jit"));
  // DuckDB computes an aggregate's arguments and FILTER on every row before it filters. a + n
  // overflows on id 1.
  run_ok("CREATE TABLE dates(id INTEGER, a DATE, n INTEGER)");
  run_ok(
    "INSERT INTO dates VALUES (1, '2000-01-01', 2147483647), (2, '2000-01-01', 1), "
    "(3, '2000-01-01', -5)");
  run_ok("CHECKPOINT");

  expect_error("SELECT max(a + n) FILTER (WHERE n < 100) FROM dates", "Date out of range");
  expect_error("SELECT id % 2, max(a + n) FILTER (WHERE n < 100) FROM dates GROUP BY id % 2",
               "Date out of range");
  expect_error("SELECT count(*) FILTER (WHERE n < id + 100 AND a + n > a) FROM dates",
               "Date out of range");
  expect_error("SELECT n < id + 100 AND a + n > a FROM dates", "Date out of range");
  // A CASE in the argument still evaluates its branches only on their rows.
  compare_gpu_vs_cpu(
    "SELECT max(CASE WHEN n < 100 THEN a + n END) FILTER (WHERE id <> 2) FROM dates");
}
