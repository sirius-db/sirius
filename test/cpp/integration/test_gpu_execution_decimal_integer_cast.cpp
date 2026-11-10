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
#include <duckdb.hpp>
#include <utils/expression_evaluator_strategies.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <string>

struct DecimalIntegerCastFixture : sirius::test::GpuExecutionFixture {
  DecimalIntegerCastFixture() { run_ok("SET threads = 1;"); }
};

namespace {
void create_cast_table(DecimalIntegerCastFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE dec_cast AS SELECT id, v::DECIMAL(9,2) AS d9, v::DECIMAL(18,2) AS d18 "
    "FROM (VALUES (1,1.49), (2,1.50), (3,1.51), (4,-1.49), (5,-1.50), (6,-1.51), "
    "(7,0.49), (8,0.50), (9,-0.49), (10,-0.50), (11,NULL), (12,0.00)) t(id,v);");
  fx.run_ok("CHECKPOINT;");
}
}  // namespace

TEST_CASE_METHOD(DecimalIntegerCastFixture,
                 "decimal integer casts match DuckDB across strategies and destinations",
                 "[integration][gpu_execution][decimal_integer_cast]")
{
  auto const strategy = GENERATE(sirius::test::each_expression_evaluator_strategy());
  CAPTURE(strategy);
  create_cast_table(*this);
  sirius::test::scoped_expression_evaluator_strategy strategy_guard{*con, strategy};
  for (auto const* column : {"d9", "d18"}) {
    for (auto const* target : {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT"}) {
      compare_gpu_vs_cpu("SELECT id, CAST(" + std::string(column) + " AS " + target +
                         ") FROM dec_cast;");
    }
    for (auto const* target : {"UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT"}) {
      compare_gpu_vs_cpu("SELECT id, TRY_CAST(" + std::string(column) + " AS " + target +
                         ") FROM dec_cast;");
    }
  }
  compare_gpu_vs_cpu("SELECT id, d18::BIGINT + 10 FROM dec_cast WHERE d9::INTEGER > 0;");
  compare_gpu_vs_cpu("SELECT d18::INTEGER k, SUM(d9::BIGINT) FROM dec_cast GROUP BY k;");
  compare_gpu_vs_cpu(
    "SELECT a.id,b.id FROM dec_cast a JOIN dec_cast b ON a.d18::BIGINT IS NOT DISTINCT FROM "
    "b.d9::BIGINT;");
}

TEST_CASE_METHOD(DecimalIntegerCastFixture,
                 "decimal integer casts check overflow after rounding",
                 "[integration][gpu_execution][decimal_integer_cast]")
{
  auto const strategy = GENERATE(sirius::test::each_expression_evaluator_strategy());
  CAPTURE(strategy);
  run_ok(
    "CREATE TABLE dec_bounds AS SELECT id, v::DECIMAL(18,2) AS v FROM "
    "(VALUES (1,127.49), (2,127.50), (3,-128.49), (4,-128.50), (5,255.49), (6,255.50), "
    "(7,-0.49), (8,-0.50), (9,NULL)) t(id,v);");
  run_ok("CHECKPOINT;");
  sirius::test::scoped_expression_evaluator_strategy strategy_guard{*con, strategy};
  compare_gpu_vs_cpu("SELECT id, TRY_CAST(v AS TINYINT), TRY_CAST(v AS UTINYINT) FROM dec_bounds;");
  run_ok("SET enable_duckdb_fallback = true;");
  auto const out_of_range = "SELECT CAST(v AS TINYINT) FROM dec_bounds;";
  if (strategy == "nvrtc_jit") {
    // TODO(kevkrist/sirius#30): the NVRTC overflow error bypasses CPU Replay, so DuckDB's error
    // never surfaces. Expect CPU Replay under every strategy once that ticket lands.
    REQUIRE(run_on_path(out_of_range, sirius::test::execution_path::gpu)->HasError());
  } else {
    expect_error_matches_cpu(out_of_range, sirius::test::execution_path::cpu_replay);
  }
  // The cast's invalid rows belong to the unselected branch, so the query returns a value. The
  // fused nvrtc_jit kernel skips those rows itself; the other strategies evaluate the cast over
  // every row and rely on CPU Replay.
  expect_result_matches_cpu(
    "SELECT CASE WHEN v BETWEEN -128.49 AND 127.49 THEN CAST(v AS TINYINT) ELSE 0::TINYINT END "
    "FROM dec_bounds;",
    strategy == "nvrtc_jit" ? sirius::test::execution_path::gpu
                            : sirius::test::execution_path::cpu_replay);
}

TEST_CASE_METHOD(DecimalIntegerCastFixture,
                 "decimal128 integer casts retain precision past BIGINT",
                 "[integration][gpu_execution][parquet][decimal_integer_cast]")
{
  auto const strategy = GENERATE(sirius::test::each_expression_evaluator_strategy());
  CAPTURE(strategy);
  sirius::test::scratch_dir scratch{"decimal_integer_cast"};
  auto const path = scratch.file_literal("decimal_cast.parquet");
  run_ok(
    "COPY (SELECT id, v::DECIMAL(38,2) AS v FROM (VALUES "
    "(1,'18446744073709551615.49'), (2,'18446744073709551615.50'), "
    "(3,'9223372036854775807.49'), (4,'9223372036854775807.50'), "
    "(5,'-9223372036854775808.49'), (6,'-9223372036854775808.50'), (7,NULL)) t(id,v)) "
    "TO " +
    path + " (FORMAT PARQUET);");
  auto const source = "read_parquet(" + path + ")";
  sirius::test::scoped_expression_evaluator_strategy strategy_guard{*con, strategy};
  compare_gpu_vs_cpu("SELECT id, v::HUGEINT, TRY_CAST(v AS BIGINT), TRY_CAST(v AS UBIGINT) FROM " +
                     source);
  run_ok("SET enable_duckdb_fallback = true;");
  expect_runtime_fallback_matches_cpu("SELECT id, TRY_CAST(v AS UHUGEINT) FROM " + source);
}
