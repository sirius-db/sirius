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
 * @file test_gpu_execution_semantic_cast_fallback.cpp
 * @brief Verifies that GPU planning rejects semantic temporal-numeric casts.
 *
 * DuckDB's CAST and TRY_CAST behavior for these pairs cannot be implemented by the physical DATE
 * carrier tunnel. The tests cover configured CPU fallback, fallback-disabled plan rejection, and
 * positive GPU-planning controls. A file-backed database exercises the DuckDB-native scan path.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

using SemanticCastFixture = sirius::test::GpuExecutionFixture;

namespace {

/// Rejection wording of the projection translation site (sirius_plan_projection.cpp).
constexpr char kProjectionRejection[] = "Unsupported expression in projection";
/// Shared prefix of both filter rejection sites: LogicalFilter translation (sirius_plan_filter.cpp,
/// "Unsupported filter predicate (falling back to CPU)") and pushed-down table filters
/// (sirius_plan_get.cpp, "Unsupported filter predicate on column ..."). DuckDB pushes the tested
/// predicate into table filters, so the second site is the one that fires today.
constexpr char kFilterRejection[] = "Unsupported filter predicate";

/// Run @p query with gpu_execution on and CPU fallback disabled: the plan rejection must surface as
/// an error carrying @p expected_detail, the rejecting site's own wording, so the assertion cannot
/// be satisfied by an unrelated plan failure -- tunneled (bit-reinterpreted) values must never come
/// back.
void expect_gpu_only_rejection(SemanticCastFixture& fx,
                               std::string const& query,
                               std::string const& expected_detail)
{
  fx.run_ok("SET gpu_execution = true;");
  fx.run_ok("SET enable_duckdb_fallback = false;");
  auto const before = sirius::test::get_transparent_execution_stats(*fx.con);
  auto result       = fx.con->Query(query);
  auto const after  = sirius::test::get_transparent_execution_stats(*fx.con);
  fx.run_ok("SET enable_duckdb_fallback = true;");
  REQUIRE(result);
  if (!result->HasError()) {
    UNSCOPED_INFO("expected a plan rejection, got rows: " << result->ToString());
  }
  REQUIRE(result->HasError());
  REQUIRE(result->GetError().find("GPU plan generation failed") != std::string::npos);
  if (result->GetError().find(expected_detail) == std::string::npos) {
    UNSCOPED_INFO("rejection came from the wrong site: " << result->GetError());
  }
  REQUIRE(result->GetError().find(expected_detail) != std::string::npos);
  // Plan-time rejection, not a mid-execution failure: the execution counter must not move.
  REQUIRE(after.executions == before.executions);
}

void create_cast_table(SemanticCastFixture& fx)
{
  fx.run_ok("CREATE TABLE cast_t(d DATE, s SMALLINT, ts TIMESTAMP, b BIGINT, dec DECIMAL(15,2));");
  // DATE '1970-01-02' is epoch day 1: exactly the value the carrier tunnel would misreport as
  // SMALLINT 1 (and s = 1 the value it would misreport as DATE '1970-01-02'). The extra columns
  // feed the family-pair coverage below.
  fx.run_ok(
    "INSERT INTO cast_t VALUES"
    " (DATE '1970-01-02', 1, TIMESTAMP '1970-01-02 00:00:00', 1, 1.00),"
    " (DATE '2024-06-15', 123, TIMESTAMP '2024-06-15 12:34:56', 123, 123.45),"
    " (NULL, NULL, NULL, NULL, NULL);");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(SemanticCastFixture,
                 "semantic cast - TRY_CAST(DATE AS SMALLINT) falls back and returns NULLs",
                 "[integration][gpu_execution][semantic_cast_fallback]")
{
  create_cast_table(*this);
  // DuckDB null-cast semantics: every row is NULL -- never the int16 epoch-day value 1.
  expect_plan_fallback_matches_cpu("SELECT TRY_CAST(d AS SMALLINT) FROM cast_t;");
}

TEST_CASE_METHOD(SemanticCastFixture,
                 "semantic cast - TRY_CAST(SMALLINT AS DATE) falls back and returns NULLs",
                 "[integration][gpu_execution][semantic_cast_fallback]")
{
  create_cast_table(*this);
  // Every row NULL -- never DATE '1970-01-02' retagged from s = 1.
  expect_plan_fallback_matches_cpu("SELECT TRY_CAST(s AS DATE) FROM cast_t;");
}

TEST_CASE_METHOD(SemanticCastFixture,
                 "semantic cast - CAST(SMALLINT AS DATE) falls back and errors like the CPU",
                 "[integration][gpu_execution][semantic_cast_fallback]")
{
  create_cast_table(*this);
  // The non-try null-cast raises a ConversionException on the CPU; after the plan-time fallback
  // the GPU session must reproduce that error rather than return retagged dates.
  run_ok("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto result       = con->Query("SELECT CAST(s AS DATE) FROM cast_t;");
  auto const after  = sirius::test::get_transparent_execution_stats(*con);
  REQUIRE(result);
  REQUIRE(result->HasError());
  REQUIRE(after.fallbacks == before.fallbacks + 1);
  REQUIRE(after.executions == before.executions);

  run_ok("SET gpu_execution = false;");
  auto cpu_result = con->Query("SELECT CAST(s AS DATE) FROM cast_t;");
  run_ok("SET gpu_execution = true;");
  REQUIRE(cpu_result);
  REQUIRE(cpu_result->HasError());
}

TEST_CASE_METHOD(SemanticCastFixture,
                 "semantic cast - CAST(DATE AS SMALLINT) falls back and errors like the CPU",
                 "[integration][gpu_execution][semantic_cast_fallback]")
{
  create_cast_table(*this);
  // The other direction of the non-try pair: the null-cast errors on the CPU; after the plan-time
  // fallback the GPU session must reproduce that error rather than return epoch days as SMALLINT.
  run_ok("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto result       = con->Query("SELECT CAST(d AS SMALLINT) FROM cast_t;");
  auto const after  = sirius::test::get_transparent_execution_stats(*con);
  REQUIRE(result);
  REQUIRE(result->HasError());
  REQUIRE(after.fallbacks == before.fallbacks + 1);
  REQUIRE(after.executions == before.executions);

  run_ok("SET gpu_execution = false;");
  auto cpu_result = con->Query("SELECT CAST(d AS SMALLINT) FROM cast_t;");
  run_ok("SET gpu_execution = true;");
  REQUIRE(cpu_result);
  REQUIRE(cpu_result->HasError());
}

TEST_CASE_METHOD(SemanticCastFixture,
                 "semantic cast - a filter over TRY_CAST(DATE AS SMALLINT) falls back",
                 "[integration][gpu_execution][semantic_cast_fallback]")
{
  create_cast_table(*this);
  // Filter position: DuckDB pushes this predicate into the scan's table filters, so the same
  // translation declines it from reject_untranslatable_table_filter (sirius_plan_get.cpp) rather
  // than the LogicalFilter site. On the CPU the null-cast makes the predicate true for every row.
  expect_plan_fallback_matches_cpu(
    "SELECT count(*) FROM cast_t WHERE TRY_CAST(d AS SMALLINT) IS NULL;");
}

TEST_CASE_METHOD(SemanticCastFixture,
                 "semantic cast - with fallback disabled the plan is rejected, never tunneled",
                 "[integration][gpu_execution][semantic_cast_fallback]")
{
  create_cast_table(*this);
  expect_gpu_only_rejection(
    *this, "SELECT TRY_CAST(d AS SMALLINT) FROM cast_t;", kProjectionRejection);
  expect_gpu_only_rejection(*this, "SELECT TRY_CAST(s AS DATE) FROM cast_t;", kProjectionRejection);
  expect_gpu_only_rejection(*this, "SELECT CAST(s AS DATE) FROM cast_t;", kProjectionRejection);
  expect_gpu_only_rejection(*this, "SELECT CAST(d AS SMALLINT) FROM cast_t;", kProjectionRejection);
  expect_gpu_only_rejection(
    *this, "SELECT count(*) FROM cast_t WHERE TRY_CAST(d AS SMALLINT) IS NULL;", kFilterRejection);
}

TEST_CASE_METHOD(SemanticCastFixture,
                 "semantic cast - positive control: cast_t stays GPU-plannable",
                 "[integration][gpu_execution][semantic_cast_fallback]")
{
  create_cast_table(*this);
  // Guards this file against going tautological: the fallback and rejection cases above only
  // prove the CAST GATE fired if the same table is otherwise GPU-plannable. compare_gpu_vs_cpu
  // asserts executions +1 / fallbacks +0 for each query.
  compare_gpu_vs_cpu("SELECT d FROM cast_t WHERE d IS NOT NULL;");
  // A supported (temporal -> temporal) cast on the same column translates and runs on the GPU.
  compare_gpu_vs_cpu("SELECT CAST(d AS TIMESTAMP) FROM cast_t;");
}

TEST_CASE_METHOD(SemanticCastFixture,
                 "semantic cast - the gate covers every temporal/numeric family pair, not VARCHAR",
                 "[integration][gpu_execution][semantic_cast_fallback]")
{
  create_cast_table(*this);
  SECTION("TIMESTAMP <-> BIGINT is rejected")
  {
    expect_gpu_only_rejection(
      *this, "SELECT TRY_CAST(ts AS BIGINT) FROM cast_t;", kProjectionRejection);
    expect_gpu_only_rejection(
      *this, "SELECT TRY_CAST(b AS TIMESTAMP) FROM cast_t;", kProjectionRejection);
  }
  SECTION("DECIMAL <-> temporal is rejected")
  {
    expect_gpu_only_rejection(
      *this, "SELECT TRY_CAST(dec AS DATE) FROM cast_t;", kProjectionRejection);
  }
  SECTION("temporal -> VARCHAR is not gated at translation")
  {
    // VARCHAR is neither temporal nor numeric: the gate must not creep beyond the null-cast pairs
    // it exists for, so this query must never take the PLAN-time fallback. It does not run purely
    // on the GPU today -- cuDF's unary cast is fixed-width only, so execution completes via a
    // runtime fallback -- which is why this locks the plan-time permit and result correctness
    // rather than a GPU-only execution.
    run_ok("SET gpu_execution = true;");
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto result       = con->Query("SELECT CAST(d AS VARCHAR) FROM cast_t;");
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    REQUIRE(after.fallbacks == before.fallbacks);

    run_ok("SET gpu_execution = false;");
    auto cpu_result = con->Query("SELECT CAST(d AS VARCHAR) FROM cast_t;");
    run_ok("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    REQUIRE_FALSE(cpu_result->HasError());
    auto rows = SemanticCastFixture::collect_rows(result->Cast<duckdb::MaterializedQueryResult>());
    auto cpu_rows =
      SemanticCastFixture::collect_rows(cpu_result->Cast<duckdb::MaterializedQueryResult>());
    REQUIRE(rows == cpu_rows);
  }
}
