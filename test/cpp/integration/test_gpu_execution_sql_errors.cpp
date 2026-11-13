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
 * @file test_gpu_execution_sql_errors.cpp
 * @brief SQL errors that the GPU detects are decided by CPU Replay, so every query returns DuckDB's
 * rows or DuckDB's error.
 *
 * The error contract in `docs/super-sirius/expression-executor.md` states what this guarantees. An
 * overflow in a branch that a Row-exact Guard (CASE, COALESCE) excludes returns DuckDB's value,
 * even where an evaluator computes that branch over every row; an evaluated overflow returns
 * DuckDB's error even where the row's value is NULL or another AND, OR or IN operand decides it,
 * unless a filter predicate or CASE WHEN condition skips that operand for the row as DuckDB does;
 * an unguarded conversion overflow returns DuckDB's error, as does an overflow in a TRY_CAST's
 * input; and a check that DuckDB does not share (a non-finite STDDEV_SAMP) returns DuckDB's value.
 * Each case asserts whether the query completed on the GPU or through CPU Replay, so a replay
 * cannot hide a broken GPU path. Overflow of abs and negation is covered beside their other cases
 * in `test_gpu_execution_abs.cpp` and `test_gpu_execution_negate.cpp`.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/expression_evaluator_strategies.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <cstdint>

using SqlErrorFixture = sirius::test::GpuExecutionFixture;

using sirius::test::execution_path;
using sirius::test::guarded_error_path;

TEST_CASE_METHOD(SqlErrorFixture,
                 "an overflow excluded by a Row-exact Guard returns DuckDB's value",
                 "[integration][gpu_execution][sql_errors]")
{
  auto const strategy = GENERATE(sirius::test::each_expression_evaluator_strategy());
  CAPTURE(strategy);
  run_ok(
    "CREATE TABLE guarded AS SELECT * FROM (VALUES"
    " (1, (-2147483648)::INTEGER, 0, DATE '2000-01-01'),"
    " (2, 5, NULL, DATE '2000-01-02'),"
    " (3, NULL, NULL, NULL)) v(id, x, y, d);");
  run_ok("CHECKPOINT;");
  sirius::test::scoped_expression_evaluator_strategy strategy_guard{*con, strategy};

  expect_result_matches_cpu(
    "SELECT id, CASE WHEN x = -2147483648 THEN 0 ELSE abs(x) END FROM guarded;",
    guarded_error_path(strategy));
  expect_result_matches_cpu("SELECT id, COALESCE(y, abs(x)) FROM guarded;",
                            guarded_error_path(strategy));
  // No strategy fuses DATE + INTEGER, so every strategy, `nvrtc_jit` included, evaluates the ELSE
  // branch over every row. Under `nvrtc_jit` the `abs(x)` inside it may still run as a kernel of
  // its own, whose overflow must recover through CPU Replay like any other.
  expect_result_matches_cpu(
    "SELECT id, CASE WHEN x = -2147483648 THEN d ELSE d + abs(x) END FROM guarded;",
    execution_path::cpu_replay);
}

TEST_CASE_METHOD(SqlErrorFixture,
                 "an evaluated overflow returns DuckDB's error even when the row's value is NULL",
                 "[integration][gpu_execution][sql_errors]")
{
  auto const strategy = GENERATE(sirius::test::each_expression_evaluator_strategy());
  CAPTURE(strategy);
  run_ok(
    "CREATE TABLE null_sums AS SELECT * FROM (VALUES"
    " (1, NULL::INTEGER, (-2147483648)::INTEGER),"
    " (2, 1, 5)) v(id, n, x);");
  run_ok("CHECKPOINT;");
  sirius::test::scoped_expression_evaluator_strategy strategy_guard{*con, strategy};

  expect_error_matches_cpu("SELECT id, n + abs(x) FROM null_sums;", execution_path::cpu_replay);
  expect_error_matches_cpu("SELECT id FROM null_sums WHERE n + abs(x) > 0;",
                           execution_path::cpu_replay);
  // No row selects the THEN branch, so DuckDB never evaluates the overflowing abs(x). The guard
  // depends on the row, so the optimizer cannot fold the branch away.
  expect_result_matches_cpu("SELECT id, CASE WHEN id < 0 THEN abs(x) ELSE 1 END FROM null_sums;",
                            guarded_error_path(strategy));
}

TEST_CASE_METHOD(SqlErrorFixture,
                 "a temporal conversion overflow returns DuckDB's value when guarded and DuckDB's "
                 "error otherwise",
                 "[integration][gpu_execution][sql_errors]")
{
  auto const strategy = GENERATE(sirius::test::each_expression_evaluator_strategy());
  CAPTURE(strategy);
  // DATE reaches past TIMESTAMP: midnight of year 300000 overflows its int64 microseconds.
  run_ok(
    "CREATE TABLE far_dates AS SELECT * FROM (VALUES (1, DATE '2000-01-01'),"
    " (2, DATE '300000-01-01'), (3, NULL)) v(id, d);");
  // SQL cannot spell a TIMESTAMP_MS past the microsecond range, so the row binds its raw ticks.
  run_ok("CREATE TABLE far_millis (id INTEGER, ms TIMESTAMP_MS);");
  run_ok("INSERT INTO far_millis VALUES (1, TIMESTAMP_MS '2000-01-01'), (3, NULL);");
  auto insert = con->Prepare("INSERT INTO far_millis VALUES (2, $1);");
  REQUIRE_FALSE(insert->HasError());
  auto constexpr past_microsecond_range = std::int64_t{9'223'372'036'854'776};
  REQUIRE_FALSE(
    insert->Execute(duckdb::Value::TIMESTAMPMS(duckdb::timestamp_ms_t{past_microsecond_range}))
      ->HasError());
  run_ok("CHECKPOINT;");
  sirius::test::scoped_expression_evaluator_strategy strategy_guard{*con, strategy};

  expect_result_matches_cpu(
    "SELECT id, CASE WHEN d < DATE '200000-01-01' THEN CAST(d AS TIMESTAMP) END FROM far_dates;",
    guarded_error_path(strategy));
  expect_result_matches_cpu(
    "SELECT id, CASE WHEN id <> 2 THEN CAST(ms AS TIMESTAMP) END FROM far_millis;",
    guarded_error_path(strategy));
  expect_error_matches_cpu("SELECT id, CAST(ms AS TIMESTAMP) FROM far_millis;",
                           execution_path::cpu_replay);
}

TEST_CASE_METHOD(SqlErrorFixture,
                 "an AND, OR or IN operand returns DuckDB's error wherever DuckDB evaluates it",
                 "[integration][gpu_execution][sql_errors]")
{
  auto const strategy = GENERATE(sirius::test::each_expression_evaluator_strategy());
  CAPTURE(strategy);
  // Row 0 holds the lowest INTEGER, whose abs overflows. `m` is a column rather than a constant so
  // that DuckDB cannot push `x <> m` into the scan, and `w` equals `x` on row 0 so that its IN list
  // matches before reaching abs(x).
  run_ok(
    "CREATE TABLE operands AS SELECT * FROM (VALUES"
    " (0, (-2147483648)::INTEGER, 1, (-2147483648)::INTEGER, (-2147483648)::INTEGER),"
    " (1, 3, -1, 0, 3), (2, -2, 0, 0, 0), (3, NULL, NULL, 0, NULL), (4, 4, -7, 0, 9),"
    " (5, 5, 9, 1, 5)) v(id, x, y, m, w);");
  run_ok("CHECKPOINT;");
  sirius::test::scoped_expression_evaluator_strategy strategy_guard{*con, strategy};

  // DuckDB evaluates every AND and OR operand of a select-list expression on every row, so the
  // overflow raises whichever side holds it and whatever the other side decides. DuckDB rewrites a
  // BETWEEN whose bound is not a constant into an AND of its two comparisons, so only native tests
  // reach Sirius's own BETWEEN with an overflowing bound.
  for (auto const* query : {"SELECT id, (abs(x) > 0) AND (y < 0) FROM operands;",
                            "SELECT id, (y < 0) AND (abs(x) > 0) FROM operands;",
                            "SELECT id, (abs(x) > 0) OR (y > 0) FROM operands;",
                            "SELECT id, (y > 0) OR (abs(x) > 0) FROM operands;",
                            "SELECT id, y BETWEEN abs(x) AND -id FROM operands;"}) {
    expect_error_matches_cpu(query, execution_path::cpu_replay);
  }

  // DuckDB evaluates every IN list value on every row, after a match too, in a select list and in a
  // filter alike.
  expect_error_matches_cpu("SELECT id, x IN (w, abs(x)) FROM operands;",
                           execution_path::cpu_replay);
  expect_error_matches_cpu("SELECT id FROM operands WHERE x IN (w, abs(x));",
                           execution_path::cpu_replay);

  // A filter predicate and a CASE WHEN condition run a later AND operand only on the rows every
  // earlier operand made TRUE, so DuckDB never computes abs(x) for row 0. Every strategy completes
  // the filter on the GPU; only `nvrtc_jit` also completes the CASE there, as the other strategies
  // compute its condition on every row.
  expect_result_matches_cpu("SELECT id FROM operands WHERE (x <> m) AND abs(x) > 0;",
                            execution_path::gpu);
  expect_result_matches_cpu(
    "SELECT id, CASE WHEN (x <> m) AND abs(x) > 0 THEN 1 ELSE 0 END FROM operands;",
    guarded_error_path(strategy));
}

TEST_CASE_METHOD(SqlErrorFixture,
                 "TRY_CAST returns NULL for its own conversion failure and DuckDB's error for an "
                 "overflow in its input",
                 "[integration][gpu_execution][sql_errors]")
{
  run_ok(
    "CREATE TABLE try_inputs AS SELECT * FROM (VALUES"
    " (0, 9223372036854775807::BIGINT), (1, 2), (2, NULL)) v(id, z);");
  run_ok("CHECKPOINT;");
  // Only `nvrtc_jit` checks BIGINT addition and narrowing integer casts: the other strategies wrap
  // both on the GPU, so they return wrong rows for these queries rather than NULL or an error.
  sirius::test::scoped_expression_evaluator_strategy strategy_guard{
    *con, sirius::expression_evaluator_strategy::NVRTC_JIT};

  expect_result_matches_cpu("SELECT id, TRY_CAST(z AS INTEGER) FROM try_inputs;",
                            execution_path::gpu);
  expect_error_matches_cpu("SELECT id, TRY_CAST(z + 1 AS INTEGER) FROM try_inputs;",
                           execution_path::cpu_replay);
}

TEST_CASE_METHOD(SqlErrorFixture,
                 "a non-finite STDDEV_SAMP returns DuckDB's value through CPU Replay",
                 "[integration][gpu_execution][sql_errors]")
{
  // The GPU rejects a non-finite sample deviation, which DuckDB returns as infinity.
  run_ok("CREATE TABLE wide AS SELECT * FROM (VALUES (1e200::DOUBLE), (-1e200::DOUBLE)) v(v);");
  run_ok("CHECKPOINT;");
  expect_result_matches_cpu("SELECT stddev_samp(v) FROM wide;", execution_path::cpu_replay);
}
