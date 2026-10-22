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
 * @file test_gpu_execution_integer_sum_overflow.cpp
 * @brief Verifies that SUM and AVG over integer columns never wrap around INT64.
 *
 * DuckDB accumulates SUM(INTEGER), SUM(BIGINT), AVG(INTEGER) and AVG(BIGINT) in 128 bits and
 * returns HUGEINT for SUM, unless its statistics prove the total fits in INT64 (sum_no_overflow).
 * The GPU computes such sums in INT64 while that is provably exact and over DECIMAL128 otherwise.
 * A SUM whose total does not fit BIGINT cannot be returned by the GPU and falls back to DuckDB at
 * runtime, which still returns the exact HUGEINT.
 *
 * An AVG that reaches the GPU is always computed there. But DuckDB's AGGREGATE_FUNCTION_REWRITER
 * turns every ungrouped AVG(x), and a grouped one that can share a SUM or COUNT, into
 * SUM(x) / COUNT(x). Such an AVG over a total past BIGINT is then that SUM and falls back too.
 *
 * AVG over a 128-bit sum is the sum converted to double divided by the count, while DuckDB
 * divides in long double. The two can differ in the last bit or two, so AVG columns compare
 * with a 1e-12 relative tolerance; every SUM compares exactly.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <set>
#include <string>

using IntegerSumFixture = sirius::test::GpuExecutionFixture;

namespace {

/// The column types of @p query's result with GPU execution on and off must agree: SUM over
/// INTEGER or BIGINT is HUGEINT and AVG is DOUBLE either way.
void require_same_result_types(IntegerSumFixture& fx, std::string const& query)
{
  fx.run_ok("SET gpu_execution = true;");
  auto gpu = fx.con->Query(query);
  fx.run_ok("SET gpu_execution = false;");
  auto cpu = fx.con->Query(query);
  fx.run_ok("SET gpu_execution = true;");
  REQUIRE_FALSE(gpu->HasError());
  REQUIRE_FALSE(cpu->HasError());
  REQUIRE(gpu->GetTypes() == cpu->GetTypes());
}

/// Whether DuckDB's optimized plan for @p query (CPU) contains @p needle.
bool cpu_plan_contains(IntegerSumFixture& fx, std::string const& query, std::string const& needle)
{
  fx.run_ok("SET gpu_execution = false;");
  auto plan = fx.con->Query("EXPLAIN " + query);
  fx.run_ok("SET gpu_execution = true;");
  REQUIRE_FALSE(plan->HasError());
  return plan->ToString().find(needle) != std::string::npos;
}

}  // namespace

// The repro: ten rows of 4e18 sum to 4e19, past 2^63 - 1 = 9.22e18. The GPU used to return the
// INT64 wrap-around 3106511852580896768 for SUM and 3.1e17 for AVG.

TEST_CASE_METHOD(IntegerSumFixture,
                 "integer sum overflow - BIGINT SUM and AVG past INT64, grouped and ungrouped",
                 "[integration][gpu_execution][aggregate][integer_sum_overflow]")
{
  run_ok(
    "CREATE TABLE big AS SELECT 4000000000000000000::BIGINT AS x, (i % 2)::INTEGER AS g "
    "FROM range(10) t(i);");
  run_ok("CHECKPOINT;");
  REQUIRE(cpu_plan_contains(*this, "SELECT SUM(x) FROM big", "sum("));
  REQUIRE_FALSE(cpu_plan_contains(*this, "SELECT SUM(x) FROM big", "sum_no_overflow"));

  expect_runtime_fallback_matches_cpu("SELECT AVG(x) FROM big;");
  compare_gpu_vs_cpu_approx("SELECT g, AVG(x) FROM big GROUP BY g;", std::set<size_t>{1}, 1e-12);
  expect_runtime_fallback_matches_cpu("SELECT SUM(x) FROM big;");
  expect_runtime_fallback_matches_cpu("SELECT g, SUM(x) FROM big GROUP BY g;");
  require_same_result_types(*this, "SELECT g, SUM(x), AVG(x) FROM big GROUP BY g;");
}

TEST_CASE_METHOD(IntegerSumFixture,
                 "integer sum overflow - totals exactly at and just past INT64_MAX and INT64_MIN",
                 "[integration][gpu_execution][aggregate][integer_sum_overflow]")
{
  // 2^62 + (2^62 - 1) = INT64_MAX, and -2^62 - 2^62 = INT64_MIN: both still fit, but two rows of
  // 2^62 already fail the per-batch bound, so they are summed over DECIMAL128 and narrowed back.
  run_ok(
    "CREATE TABLE at_max AS SELECT * FROM (VALUES (0, 4611686018427387904::BIGINT), "
    "(0, 4611686018427387903::BIGINT), (1, -4611686018427387904::BIGINT), "
    "(1, -4611686018427387904::BIGINT)) t(g, x);");
  run_ok(
    "CREATE TABLE past_max AS SELECT * FROM (VALUES (0, 4611686018427387904::BIGINT), "
    "(0, 4611686018427387904::BIGINT)) t(g, x);");
  run_ok(
    "CREATE TABLE past_min AS SELECT * FROM (VALUES (0, -4611686018427387904::BIGINT), "
    "(0, -4611686018427387904::BIGINT), (0, -1::BIGINT)) t(g, x);");
  run_ok("CHECKPOINT;");

  compare_gpu_vs_cpu("SELECT g, SUM(x) s FROM at_max GROUP BY g;");
  compare_gpu_vs_cpu("SELECT SUM(x) s FROM at_max WHERE g = 0;");
  compare_gpu_vs_cpu("SELECT SUM(x) s FROM at_max WHERE g = 1;");
  compare_gpu_vs_cpu_approx("SELECT g, AVG(x) a FROM at_max GROUP BY g;", {1}, 1e-12);

  expect_runtime_fallback_matches_cpu("SELECT SUM(x) FROM past_max;");
  expect_runtime_fallback_matches_cpu("SELECT g, SUM(x) FROM past_max GROUP BY g;");
  expect_runtime_fallback_matches_cpu("SELECT SUM(x) FROM past_min;");
  expect_runtime_fallback_matches_cpu("SELECT g, SUM(x) FROM past_min GROUP BY g;");
  // The grouped average of the same rows stays on the GPU; the ungrouped one is rewritten to a SUM.
  expect_runtime_fallback_matches_cpu("SELECT AVG(x) FROM past_max;");
  compare_gpu_vs_cpu_approx("SELECT g, AVG(x) FROM past_min GROUP BY g;", {1}, 1e-12);
}

TEST_CASE_METHOD(IntegerSumFixture,
                 "integer sum overflow - large values of both signs whose total fits",
                 "[integration][gpu_execution][aggregate][integer_sum_overflow]")
{
  // Alternating +-4e18: any prefix of three same-sign rows exceeds INT64, but each group's
  // total is small. The GPU sums over DECIMAL128 and narrows the exact total.
  run_ok(
    "CREATE TABLE mixed AS SELECT (i % 3)::INTEGER g, "
    "CASE WHEN i % 2 = 0 THEN 4000000000000000000 ELSE -3999999999999999999 END::BIGINT x "
    "FROM range(3000) t(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT g, SUM(x) s FROM mixed GROUP BY g;");
  compare_gpu_vs_cpu("SELECT SUM(x) s FROM mixed;");
  compare_gpu_vs_cpu_approx("SELECT g, AVG(x) a, SUM(x) s FROM mixed GROUP BY g;", {1}, 1e-12);
  compare_gpu_vs_cpu_approx("SELECT AVG(x) a FROM mixed;", {0}, 1e-12);
  // Grouping sets re-aggregate the partial sums for the coarser sets.
  compare_gpu_vs_cpu("SELECT g, SUM(x) s FROM mixed GROUP BY ROLLUP(g);");
}

TEST_CASE_METHOD(IntegerSumFixture,
                 "integer sum overflow - NULLs, all-NULL groups and empty input",
                 "[integration][gpu_execution][aggregate][integer_sum_overflow]")
{
  run_ok(
    "CREATE TABLE nulls AS SELECT (i % 3)::INTEGER g, "
    "CASE WHEN i % 3 = 2 THEN NULL ELSE 3000000000000000000 + i END::BIGINT x "
    "FROM range(9) t(i);");
  run_ok("CREATE TABLE allnull AS SELECT (i % 2)::INTEGER g, NULL::BIGINT x FROM range(10) t(i);");
  run_ok("CREATE TABLE empty_t (g INTEGER, x BIGINT);");
  run_ok("CHECKPOINT;");
  // Groups 0 and 1 hold three values of 3e18 (9e18 + small, within INT64); group 2 is all NULL.
  compare_gpu_vs_cpu("SELECT g, SUM(x) s, COUNT(x) c FROM nulls GROUP BY g;");
  compare_gpu_vs_cpu_approx("SELECT g, AVG(x) a FROM nulls GROUP BY g;", {1}, 1e-12);
  compare_gpu_vs_cpu("SELECT g, SUM(x) s FROM allnull GROUP BY g;");
  compare_gpu_vs_cpu("SELECT SUM(x) s, AVG(x) a FROM allnull;");
  compare_gpu_vs_cpu("SELECT SUM(x) s, AVG(x) a FROM empty_t;");
  compare_gpu_vs_cpu("SELECT g, SUM(x) s, AVG(x) a FROM empty_t GROUP BY g;");
  // All three non-NULL groups together exceed INT64: NULLs are skipped, not counted as zero.
  expect_runtime_fallback_matches_cpu("SELECT SUM(x) FROM nulls;");
  expect_runtime_fallback_matches_cpu("SELECT AVG(x) a FROM nulls;");
}

TEST_CASE_METHOD(IntegerSumFixture,
                 "integer sum overflow - narrower integer inputs match DuckDB",
                 "[integration][gpu_execution][aggregate][integer_sum_overflow]")
{
  run_ok(
    "CREATE TABLE narrow AS SELECT (i % 4)::INTEGER g, (2147483647 - i)::INTEGER i32, "
    "(-2147483648 + i)::INTEGER n32, (32767 - (i % 7))::SMALLINT i16, "
    "(-128 + (i % 5))::TINYINT i8, (4294967295 - i)::UINTEGER u32, "
    "(65535 - i % 3)::USMALLINT u16 FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu(
    "SELECT g, SUM(i32) a, SUM(n32) b, SUM(i16) c, SUM(i8) d, SUM(u32) e, SUM(u16) f "
    "FROM narrow GROUP BY g;");
  compare_gpu_vs_cpu(
    "SELECT SUM(i32) a, SUM(n32) b, SUM(i16) c, SUM(i8) d, SUM(u32) e, SUM(u16) f FROM narrow;");
  compare_gpu_vs_cpu_approx(
    "SELECT g, AVG(i32) a, AVG(n32) b, AVG(i16) c, AVG(i8) d, AVG(u32) e FROM narrow GROUP BY g;",
    {1, 2, 3, 4, 5},
    1e-12);
  compare_gpu_vs_cpu_approx(
    "SELECT AVG(i32) a, AVG(n32) b, AVG(i16) c, AVG(i8) d, AVG(u32) e FROM narrow;",
    {0, 1, 2, 3, 4},
    1e-12);
  require_same_result_types(*this,
                            "SELECT g, SUM(i32), SUM(i16), SUM(u32) FROM narrow GROUP BY g;");
}

TEST_CASE_METHOD(IntegerSumFixture,
                 "integer sum overflow - sums DuckDB proves fit keep the INT64 path",
                 "[integration][gpu_execution][aggregate][integer_sum_overflow]")
{
  // DuckDB's statistics bound 100000 rows of at most 999 well inside INT64 and plan
  // sum_no_overflow, which the GPU sums in INT64 exactly as before.
  run_ok(
    "CREATE TABLE small AS SELECT (i % 7)::INTEGER g, (i % 1000)::BIGINT x, "
    "(i % 1000)::INTEGER y FROM range(100000) t(i);");
  run_ok("CHECKPOINT;");
  REQUIRE(
    cpu_plan_contains(*this, "SELECT g, SUM(x), SUM(y) FROM small GROUP BY g", "sum_no_overflow"));
  compare_gpu_vs_cpu("SELECT g, SUM(x) s, SUM(y) t FROM small GROUP BY g;");
  compare_gpu_vs_cpu("SELECT SUM(x) s, SUM(y) t FROM small;");
  compare_gpu_vs_cpu_approx("SELECT g, AVG(x) a FROM small GROUP BY g;", {1}, 1e-12);
  require_same_result_types(*this, "SELECT g, SUM(x), AVG(x) FROM small GROUP BY g;");
}

// The ClickBench Q03 shape: AVG over a BIGINT id averaging ~2.5e18 across enough rows to span
// several scan batches, so the partial sums are merged. Every batch overflows INT64, so every
// partial is DECIMAL128. Q03's own AVG(UserID) is rewritten to SUM / COUNT and falls back; an AVG
// that keeps its own state (no COUNT to share) stays on the GPU.

TEST_CASE_METHOD(IntegerSumFixture,
                 "integer sum overflow - ClickBench Q03 AVG(UserID) over several batches",
                 "[integration][gpu_execution][aggregate][integer_sum_overflow]")
{
  run_ok(
    "CREATE TABLE hits AS SELECT (2500000000000000000 + (hash(i) % 1000000000000000000)::BIGINT "
    "- 500000000000000000)::BIGINT AS UserID, (i % 16)::INTEGER AS CounterID "
    "FROM range(30000000) t(i);");
  run_ok("CHECKPOINT;");
  expect_runtime_fallback_matches_cpu("SELECT AVG(UserID) FROM hits;");
  expect_runtime_fallback_matches_cpu(
    "SELECT CounterID, AVG(UserID) a, COUNT(*) c FROM hits GROUP BY CounterID;");
  compare_gpu_vs_cpu_approx(
    "SELECT CounterID, AVG(UserID) a FROM hits GROUP BY CounterID;", {1}, 1e-12);
  expect_runtime_fallback_matches_cpu("SELECT SUM(UserID) FROM hits;");
}

TEST_CASE_METHOD(IntegerSumFixture,
                 "integer sum overflow - partials that fit per batch but not when merged",
                 "[integration][gpu_execution][aggregate][integer_sum_overflow]")
{
  // 4e11 per row: a batch of a few million rows stays far below INT64, so each batch sums in
  // INT64, but 30M rows reach 1.2e19 in magnitude, beyond the merge's 2^62 bound (and beyond
  // what DuckDB's statistics can prove, so these are not sum_no_overflow). Alternating signs keep
  // every group's total of x in range, so the merge widens and the narrowed SUM stays on the GPU.
  run_ok(
    "CREATE TABLE wide_merge AS SELECT (i % 5)::INTEGER g, "
    "CASE WHEN i % 2 = 0 THEN 400000000000 ELSE -399999999999 END::BIGINT x, "
    "400000000000::BIGINT y FROM range(30000000) t(i);");
  run_ok("CHECKPOINT;");
  REQUIRE_FALSE(cpu_plan_contains(*this, "SELECT SUM(y) FROM wide_merge", "sum_no_overflow"));
  compare_gpu_vs_cpu("SELECT g, SUM(x) s FROM wide_merge GROUP BY g;");
  compare_gpu_vs_cpu("SELECT SUM(x) s FROM wide_merge;");
  compare_gpu_vs_cpu_approx("SELECT g, AVG(y) a FROM wide_merge GROUP BY g;", {1}, 1e-12);
  // y sums to 2.4e18 per group, which fits, but 1.2e19 in total, which falls back.
  compare_gpu_vs_cpu("SELECT g, SUM(y) s FROM wide_merge GROUP BY g;");
  expect_runtime_fallback_matches_cpu("SELECT SUM(y) FROM wide_merge;");
}

TEST_CASE_METHOD(IntegerSumFixture,
                 "integer sum overflow - SUM and AVG over an inner SUM (a HUGEINT input)",
                 "[integration][gpu_execution][aggregate][integer_sum_overflow]")
{
  // Each inner group sums to 8e18, which fits BIGINT; the outer sum over HUGEINT reaches 3.2e19.
  run_ok(
    "CREATE TABLE inner_t AS SELECT (i % 4)::INTEGER g, 4000000000000000000::BIGINT x "
    "FROM range(8) r(i);");
  run_ok("CHECKPOINT;");
  expect_runtime_fallback_matches_cpu(
    "SELECT AVG(s) FROM (SELECT g, SUM(x) s FROM inner_t GROUP BY g);");
  expect_runtime_fallback_matches_cpu(
    "SELECT SUM(s) FROM (SELECT g, SUM(x) s FROM inner_t GROUP BY g);");
  compare_gpu_vs_cpu(
    "SELECT SUM(s) FROM (SELECT g, SUM(x) s FROM inner_t GROUP BY g) WHERE g < 1;");
}
