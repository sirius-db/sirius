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
 * @file test_gpu_execution_hugeint.cpp
 * @brief HUGEINT results, carried on the GPU as DECIMAL128 with scale 0, flow through the
 * operators and expressions that consume them.
 *
 * DuckDB returns HUGEINT for SUM over INTEGER and BIGINT. Every table here holds groups whose sums
 * leave BIGINT, so each query only matches DuckDB if the 128-bit value survives end to end.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

struct HugeintFixture : sirius::test::GpuExecutionFixture {
  HugeintFixture()
  {
    // Groups 0-2 sum to 4e19 and group 3 to -4e19, all past BIGINT; y and u stay small.
    run_ok(
      "CREATE TABLE h AS SELECT (i % 4)::INTEGER k, "
      "CASE WHEN i % 4 = 3 THEN -4000000000000000000 ELSE 4000000000000000000 END::BIGINT x, "
      "(i % 7)::BIGINT y, (i % 5)::UBIGINT * 4000000000000000000::UBIGINT u "
      "FROM range(40) t(i);");
    run_ok("CREATE TABLE small AS SELECT (i * 3)::BIGINT v FROM range(20) t(i);");
    run_ok("CHECKPOINT;");
    run_ok("SET enable_duckdb_fallback = false;");
  }
};

constexpr char const* sums = "(SELECT k, SUM(x) s, SUM(y) t FROM h GROUP BY k)";

}  // namespace

TEST_CASE_METHOD(HugeintFixture,
                 "hugeint - sorting, top-N and HAVING over sums past BIGINT",
                 "[integration][gpu_execution][hugeint]")
{
  compare_gpu_vs_cpu_ordered("SELECT k, SUM(x) s FROM h GROUP BY k ORDER BY s DESC, k;");
  compare_gpu_vs_cpu_ordered("SELECT k, SUM(x) s FROM h GROUP BY k ORDER BY s, k LIMIT 2;");
  compare_gpu_vs_cpu("SELECT k, SUM(x) s FROM h GROUP BY k HAVING SUM(x) > 9223372036854775807;");
  compare_gpu_vs_cpu("SELECT k, SUM(x) s FROM h GROUP BY k HAVING SUM(x) = 40000000000000000000;");
}

TEST_CASE_METHOD(HugeintFixture,
                 "hugeint - arithmetic, casts and conditional expressions",
                 "[integration][gpu_execution][hugeint]")
{
  compare_gpu_vs_cpu("SELECT k, SUM(x) + SUM(y) a, SUM(x) - SUM(y) b FROM h GROUP BY k;");
  compare_gpu_vs_cpu("SELECT k, s * 2 FROM " + std::string{sums} + ";");
  compare_gpu_vs_cpu_approx("SELECT k, s::DOUBLE d FROM " + std::string{sums} + ";", {1}, 1e-15);
  compare_gpu_vs_cpu("SELECT k, s::VARCHAR v FROM " + std::string{sums} + ";");
  compare_gpu_vs_cpu("SELECT k, COALESCE(s, 0) c, CASE WHEN s > 0 THEN s ELSE t END e FROM " +
                     std::string{sums} + ";");
  compare_gpu_vs_cpu("SELECT k, s IN (40000000000000000000, 1) i FROM " + std::string{sums} + ";");
  compare_gpu_vs_cpu("SELECT k, s // 3 q, s % 3 r FROM " + std::string{sums} + ";");
}

TEST_CASE_METHOD(HugeintFixture,
                 "hugeint - implicit DECIMAL casts stay on the GPU",
                 "[integration][gpu_execution][hugeint]")
{
  // DuckDB casts the sum to DECIMAL(38, s) to meet a decimal.
  compare_gpu_vs_cpu("SELECT k, s * 1.5 m FROM " + std::string{sums} + ";");
  compare_gpu_vs_cpu("SELECT s FROM " + std::string{sums} + " UNION ALL SELECT 1.5;");
  compare_gpu_vs_cpu("SELECT k, COALESCE(s, 2.25) c FROM " + std::string{sums} + ";");
}

TEST_CASE_METHOD(HugeintFixture,
                 "hugeint - narrowing casts check the range",
                 "[integration][gpu_execution][hugeint]")
{
  // In range: the GPU narrows.
  compare_gpu_vs_cpu("SELECT k, t::BIGINT a, t::INTEGER b, t::DECIMAL(10, 2) c FROM " +
                     std::string{sums} + ";");

  run_ok("SET enable_duckdb_fallback = true;");
  // Out of range: the GPU must not wrap. CAST raises DuckDB's error and TRY_CAST returns NULL.
  auto result = con->Query("SELECT s::BIGINT FROM " + std::string{sums} + ";");
  REQUIRE(result->HasError());
  CHECK_THAT(result->GetError(), Catch::Matchers::ContainsSubstring("out of range"));
  expect_runtime_fallback_matches_cpu("SELECT k, TRY_CAST(s AS BIGINT) b FROM " +
                                      std::string{sums} + ";");
  expect_runtime_fallback_matches_cpu("SELECT k, TRY_CAST(s AS DECIMAL(18, 2)) b FROM " +
                                      std::string{sums} + ";");
}

TEST_CASE_METHOD(HugeintFixture,
                 "hugeint - division by a column stays on the CPU",
                 "[integration][gpu_execution][hugeint]")
{
  // cuDF leaves a zero divisor's row undefined, so only a nonzero constant divisor runs on the GPU.
  run_ok("SET enable_duckdb_fallback = true;");
  expect_plan_fallback_matches_cpu("SELECT k, SUM(x) // COUNT(*) a FROM h GROUP BY k;");
  expect_plan_fallback_matches_cpu("SELECT k, SUM(x) % COUNT(*) a FROM h GROUP BY k;");
}

TEST_CASE_METHOD(HugeintFixture,
                 "hugeint - joins, DISTINCT, UNION ALL and grouping on sums",
                 "[integration][gpu_execution][hugeint]")
{
  // HUGEINT = HUGEINT, and HUGEINT = BIGINT through DuckDB's cast.
  compare_gpu_vs_cpu("SELECT a.k, b.k FROM " + std::string{sums} + " a JOIN " + std::string{sums} +
                     " b ON a.s = b.s;");
  compare_gpu_vs_cpu("SELECT a.k, small.v FROM " + std::string{sums} +
                     " a JOIN small ON a.t = small.v;");
  compare_gpu_vs_cpu("SELECT DISTINCT s FROM " + std::string{sums} + ";");
  compare_gpu_vs_cpu("SELECT SUM(x) FROM h WHERE k < 2 UNION ALL SELECT SUM(x) FROM h;");
  compare_gpu_vs_cpu("SELECT SUM(x) FROM h WHERE k < 2 UNION SELECT SUM(x) FROM h WHERE k = 1;");
  compare_gpu_vs_cpu("SELECT s, COUNT(*) c FROM " + std::string{sums} + " GROUP BY s;");
  compare_gpu_vs_cpu("SELECT k, SUM(x) s FROM h GROUP BY ROLLUP(k);");
}

TEST_CASE_METHOD(HugeintFixture,
                 "hugeint - aggregates over a HUGEINT column",
                 "[integration][gpu_execution][hugeint]")
{
  compare_gpu_vs_cpu("SELECT MIN(s), MAX(s), COUNT(DISTINCT s), SUM(s) FROM " + std::string{sums} +
                     ";");
  compare_gpu_vs_cpu("SELECT s > 0 p, MIN(s), MAX(s), SUM(s) FROM " + std::string{sums} +
                     " GROUP BY p;");
  compare_gpu_vs_cpu_approx("SELECT AVG(s) FROM " + std::string{sums} + ";", {0}, 1e-12);
  // SUM over UBIGINT is HUGEINT too, and its values need all 64 unsigned bits.
  compare_gpu_vs_cpu("SELECT k, SUM(u) s FROM h GROUP BY k;");
  compare_gpu_vs_cpu("SELECT SUM(u) s FROM h;");
}
