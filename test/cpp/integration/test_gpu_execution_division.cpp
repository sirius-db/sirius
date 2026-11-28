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
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

struct DivisionFixture : sirius::test::GpuExecutionFixture {
  DivisionFixture()
  {
    run_ok("SET threads = 1;");
    run_ok(
      "CREATE TABLE div_t AS SELECT id::INTEGER id, a::BIGINT a, b::BIGINT b, "
      "u::BIGINT u, v::BIGINT v FROM (VALUES "
      "(0, 12, 3, 12, 3), (1, -12, 5, 12, 5), (2, 7, -3, 7, 3), "
      "(3, NULL, 0, NULL, 0), (4, 9, NULL, 9, NULL), (5, 0, -1, 0, 1)) "
      "t(id, a, b, u, v);");
    run_ok(
      "CREATE TABLE div_zero AS SELECT id::INTEGER id, a::BIGINT a, b::BIGINT b "
      "FROM (VALUES (0, 12, 3), (1, 9, 0), (2, NULL, 0)) t(id, a, b);");
    run_ok("CHECKPOINT;");
  }

  void expect_error(std::string const& query, std::string const& message)
  {
    INFO(query);
    run_ok("SET gpu_execution = false;");
    auto cpu = con->Query(query);
    REQUIRE(cpu->HasError());
    CHECK_THAT(cpu->GetError(), Catch::Matchers::ContainsSubstring(message));
    run_ok("SET gpu_execution = true;");
    run_ok("SET enable_duckdb_fallback = true;");
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto gpu          = con->Query(query);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(gpu->HasError());
    CHECK_THAT(gpu->GetError(), Catch::Matchers::ContainsSubstring(message));
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
  }
};

}  // namespace

TEST_CASE_METHOD(DivisionFixture,
                 "checked division - safe numeric column pairs stay on the GPU",
                 "[integration][gpu_execution][checked_division]")
{
  auto const strategy = GENERATE("materialize", "ast_interpret", "ast_jit");
  run_ok("SET expression_evaluator_strategy = '" + std::string(strategy) + "';");
  run_ok("SET enable_duckdb_fallback = false;");
  for (auto const* type :
       {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "FLOAT", "DOUBLE"}) {
    CAPTURE(strategy, type);
    auto const a = "a::" + std::string(type);
    auto const b = "b::" + std::string(type);
    compare_gpu_vs_cpu("SELECT id, " + a + " // " + b + " q, " + a + " % " + b + " r FROM div_t");
  }
  for (auto const* type : {"UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"}) {
    CAPTURE(strategy, type);
    auto const u = "u::" + std::string(type);
    auto const v = "v::" + std::string(type);
    compare_gpu_vs_cpu("SELECT id, " + u + " // " + v + " q, " + u + " % " + v + " r FROM div_t");
  }
  for (auto const* type : {"DECIMAL(9,2)", "DECIMAL(18,2)", "DECIMAL(38,2)"}) {
    CAPTURE(strategy, type);
    compare_gpu_vs_cpu("SELECT id, a::" + std::string(type) + " % b::" + type + " r FROM div_t");
  }
  // Compose checked arithmetic inside an AST parent, a filter, an aggregate and a join key.
  compare_gpu_vs_cpu("SELECT id, (a // b) + 2, 100 % b FROM div_t WHERE id <> 3");
  compare_gpu_vs_cpu("SELECT id FROM div_t WHERE a % b = 2");
  compare_gpu_vs_cpu("SELECT sum(a // b) FROM div_t");
  compare_gpu_vs_cpu("SELECT x.id, y.id FROM div_t x JOIN div_t y ON x.a // x.b = y.a");
}

TEST_CASE_METHOD(DivisionFixture,
                 "checked division - zero divisors use DuckDB error semantics",
                 "[integration][gpu_execution][checked_division]")
{
  auto const strategy = GENERATE("materialize", "ast_interpret", "ast_jit");
  run_ok("SET expression_evaluator_strategy = '" + std::string(strategy) + "';");
  run_ok("SET error_on_division_by_zero = true;");
  run_ok("SET ieee_floating_point_ops = false;");
  for (auto const* op : {"//", "%"}) {
    for (auto const* type : {"BIGINT", "HUGEINT", "UBIGINT", "DOUBLE"}) {
      expect_error("SELECT a::" + std::string(type) + op + "b::" + type + " FROM div_zero",
                   "Division by zero");
    }
    expect_error("SELECT a " + std::string(op) + " 0 FROM div_zero", "Division by zero");
    auto const query = "SELECT a " + std::string(op) + " b FROM div_zero";
    run_ok("SET enable_duckdb_fallback = false;");
    auto result = con->Query(query);
    REQUIRE(result->HasError());
    CHECK_THAT(result->GetError(), Catch::Matchers::ContainsSubstring("Division by zero"));
    run_ok("SET enable_duckdb_fallback = true;");
  }
  expect_error("SELECT a::DECIMAL(18,2) % b::DECIMAL(18,2) FROM div_zero", "Division by zero");
}

TEST_CASE_METHOD(DivisionFixture,
                 "checked division - NULL setting TRY and guarded CASE preserve CPU results",
                 "[integration][gpu_execution][checked_division]")
{
  auto const strategy = GENERATE("materialize", "ast_interpret", "ast_jit");
  run_ok("SET expression_evaluator_strategy = '" + std::string(strategy) + "';");
  run_ok("SET enable_duckdb_fallback = true;");
  for (auto const* op : {"//", "%"}) {
    run_ok("SET error_on_division_by_zero = false;");
    expect_runtime_fallback_matches_cpu("SELECT id, a " + std::string(op) + " b FROM div_zero");
    run_ok("SET error_on_division_by_zero = true;");
    // The zero divisors select THEN, so the GPU evaluates the division only on other rows.
    compare_gpu_vs_cpu("SELECT id, CASE WHEN b = 0 THEN NULL ELSE a " + std::string(op) +
                       " b END FROM div_zero");
    expect_plan_fallback_matches_cpu("SELECT id, TRY(a " + std::string(op) + " b) FROM div_zero");
  }
}

TEST_CASE_METHOD(DivisionFixture,
                 "checked division - signed minima overflow by minus one with remainder zero",
                 "[integration][gpu_execution][checked_division]")
{
  run_ok(
    "CREATE TABLE div_min AS SELECT id::INTEGER id, a::BIGINT a, b::BIGINT b FROM "
    "(VALUES (0, -128, -1), (1, -32768, -1), (2, -2147483648, -1), "
    "(3, -9223372036854775808, -1)) t(id, a, b);");
  run_ok("CHECKPOINT;");
  auto const strategy = GENERATE("materialize", "ast_interpret", "ast_jit");
  run_ok("SET expression_evaluator_strategy = '" + std::string(strategy) + "';");
  // Only the quotient overflows: // raises DuckDB's error through CPU replay, while % returns
  // DuckDB's 0 on the GPU for a column divisor and for a constant one.
  int id = 0;
  for (auto const* type : {"TINYINT", "SMALLINT", "INTEGER", "BIGINT"}) {
    CAPTURE(strategy, type);
    auto const a    = "a::" + std::string(type);
    auto const b    = "b::" + std::string(type);
    auto const rows = " FROM div_min WHERE id = " + std::to_string(id);
    expect_error("SELECT " + a + " // " + b + rows, "Overflow in division");
    run_ok("SET enable_duckdb_fallback = false;");
    compare_gpu_vs_cpu("SELECT " + a + " % " + b + rows);
    compare_gpu_vs_cpu("SELECT " + a + " % -1" + rows);
    ++id;
  }
  // The GPU cannot scan a HUGEINT column, so the HUGEINT minimum is a constant numerator.
  auto const minimum = std::string("(-170141183460469231731687303715884105728)::HUGEINT");
  auto const rows    = std::string(" FROM div_min WHERE id = 0");
  expect_error("SELECT " + minimum + " // b::HUGEINT" + rows, "Overflow in division");
  run_ok("SET enable_duckdb_fallback = false;");
  compare_gpu_vs_cpu("SELECT " + minimum + " % b::HUGEINT" + rows);
}

TEST_CASE_METHOD(DivisionFixture,
                 "checked division - a zero divisor beside a signed minimum by minus one",
                 "[integration][gpu_execution][checked_division]")
{
  // Each group g holds one type's minimum over -1 and a zero divisor.
  run_ok(
    "CREATE TABLE div_mixed AS SELECT id::INTEGER id, g::INTEGER g, a::BIGINT a, b::BIGINT b "
    "FROM (VALUES (0, 0, -128, -1), (1, 0, 9, 0), (2, 1, -32768, -1), (3, 1, 9, 0), "
    "(4, 2, -2147483648, -1), (5, 2, 9, 0), (6, 3, -9223372036854775808, -1), (7, 3, 9, 0)) "
    "t(id, g, a, b);");
  run_ok("CHECKPOINT;");
  auto const strategy = GENERATE("materialize", "ast_interpret", "ast_jit");
  run_ok("SET expression_evaluator_strategy = '" + std::string(strategy) + "';");
  // The zero divisor sends an unguarded % to CPU replay, which returns NULL for it and 0 for the
  // minimum. Under a CASE that excludes it, % is retried on the other row and stays on the GPU.
  auto const check = [&](std::string const& mod, std::string const& rows) {
    run_ok("SET enable_duckdb_fallback = true;");
    run_ok("SET error_on_division_by_zero = false;");
    expect_runtime_fallback_matches_cpu("SELECT id, " + mod + rows);
    run_ok("SET error_on_division_by_zero = true;");
    run_ok("SET enable_duckdb_fallback = false;");
    compare_gpu_vs_cpu("SELECT id, CASE WHEN b = 0 THEN NULL ELSE " + mod + " END" + rows);
  };
  int group = 0;
  for (auto const* type : {"TINYINT", "SMALLINT", "INTEGER", "BIGINT"}) {
    CAPTURE(strategy, type);
    check("a::" + std::string(type) + " % b::" + type,
          " FROM div_mixed WHERE g = " + std::to_string(group));
    ++group;
  }
  // The GPU cannot scan a HUGEINT column, so the HUGEINT minimum is a constant numerator.
  check("(-170141183460469231731687303715884105728)::HUGEINT % b::HUGEINT",
        " FROM div_mixed WHERE g = 0");
}
