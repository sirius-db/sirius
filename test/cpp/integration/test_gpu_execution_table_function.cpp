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
 * @file test_gpu_execution_table_function.cpp
 * @brief Argument handling of the explicit gpu_execution(sql, ...) table function: the SQL must
 *        hold exactly one statement, and the enable_optimizer option controls whether DuckDB's
 *        optimizer runs on the plan Sirius executes.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/main/settings.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <string>
#include <vector>

namespace {

using rows = std::vector<std::vector<std::string>>;

/// Error text of a plan the Sirius projection translator rejects (sirius_plan_projection.cpp).
constexpr char kProjectionRejection[] = "Unsupported expression in projection";

/// A table with a DATE column: TRY_CAST(DATE AS SMALLINT) is a projection Sirius rejects at plan
/// time. Placed in an unused subquery column, only DuckDB's optimizer removes it.
class GpuExecutionTableFunctionFixture : public sirius::test::GpuExecutionFixture {
 public:
  GpuExecutionTableFunctionFixture()
  {
    run_ok("CREATE TABLE tf_t (x INTEGER, v INTEGER, d DATE);");
    run_ok(
      "INSERT INTO tf_t VALUES (1, 10, DATE '1970-01-02'), (2, 20, DATE '2000-01-01'), "
      "(3, 30, DATE '2024-02-29'), (1, 40, NULL);");
    run_ok("CHECKPOINT;");
    // Keep the outer SELECT over gpu_execution(...) out of transparent interception.
    run_ok("SET gpu_execution = false;");
  }

  /// Fully qualified, so the CPU fallback's separate connection resolves it too.
  std::string table() const { return attach_alias + ".tf_t"; }

  /// Correct with or without the optimizer; GPU-plannable only once the unused column is gone.
  std::string unused_cast_query() const
  {
    return "SELECT count(*), sum(v) FROM (SELECT v, TRY_CAST(d AS SMALLINT) AS unused FROM " +
           table() + ")";
  }

  /// GPU-plannable with or without the optimizer.
  std::string group_query() const
  {
    return "SELECT x, count(*), sum(v) FROM " + table() + " WHERE v > 15 GROUP BY x";
  }

  /// `SELECT * FROM gpu_execution('<sql>'<options>)`.
  static std::string gpu_execution(std::string const& sql, std::string const& options = "")
  {
    return "SELECT * FROM gpu_execution(" + sirius::test::sql_literal(sql) + options + ")";
  }

  rows query_rows(std::string const& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    return collect_rows(*result);
  }

  std::string query_error(std::string const& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    if (!result->HasError()) { UNSCOPED_INFO("expected an error, got: " << result->ToString()); }
    REQUIRE(result->HasError());
    return result->GetError();
  }

  bool session_optimizer_enabled() const
  {
    return duckdb::Settings::Get<duckdb::EnableOptimizerSetting>(*con->context);
  }
};

}  // namespace

TEST_CASE_METHOD(GpuExecutionTableFunctionFixture,
                 "gpu_execution rejects SQL that is not exactly one statement",
                 "[integration][gpu_execution][table_function]")
{
  for (std::string const sql : {"", "   "}) {
    INFO("sql: '" << sql << "'");
    auto const error = query_error(gpu_execution(sql));
    CHECK(error.find("gpu_execution expects exactly one SQL statement, got 0") !=
          std::string::npos);
  }

  auto const error = query_error(gpu_execution("SELECT 1; SELECT 2"));
  CHECK(error.find("gpu_execution expects exactly one SQL statement, got 2") != std::string::npos);

  // The session stays usable after a rejected bind.
  REQUIRE(query_rows("SELECT 42") == rows{{"42"}});
}

TEST_CASE_METHOD(GpuExecutionTableFunctionFixture,
                 "gpu_execution honors enable_optimizer",
                 "[integration][gpu_execution][table_function]")
{
  auto const expected = query_rows(unused_cast_query());
  REQUIRE(expected == rows{{"4", "100"}});

  run_ok("SET enable_duckdb_fallback = false;");

  // Optimized (the default): the unused TRY_CAST column is removed and the plan runs on the GPU.
  CHECK(query_rows(gpu_execution(unused_cast_query())) == expected);
  CHECK(query_rows(gpu_execution(unused_cast_query(), ", enable_optimizer = true")) == expected);
  // NULL means the default.
  CHECK(query_rows(gpu_execution(unused_cast_query(), ", enable_optimizer = NULL")) == expected);

  // Unoptimized: the TRY_CAST projection reaches Sirius, which rejects it at plan time.
  auto const error = query_error(gpu_execution(unused_cast_query(), ", enable_optimizer = false"));
  CHECK(error.find(kProjectionRejection) != std::string::npos);

  run_ok("SET enable_duckdb_fallback = true;");
}

TEST_CASE_METHOD(GpuExecutionTableFunctionFixture,
                 "gpu_execution enable_optimizer = false falls back to CPU for unsupported plans",
                 "[integration][gpu_execution][table_function]")
{
  run_ok("SET enable_duckdb_fallback = true;");
  auto const expected = query_rows(unused_cast_query());
  CHECK(query_rows(gpu_execution(unused_cast_query(), ", enable_optimizer = false")) == expected);
}

TEST_CASE_METHOD(GpuExecutionTableFunctionFixture,
                 "gpu_execution enable_optimizer = false runs a supported plan on the GPU",
                 "[integration][gpu_execution][table_function]")
{
  auto const expected = query_rows(group_query());
  REQUIRE(expected == rows{{"1", "1", "40"}, {"2", "1", "20"}, {"3", "1", "30"}});

  run_ok("SET enable_duckdb_fallback = false;");
  CHECK(query_rows(gpu_execution(group_query(), ", enable_optimizer = false")) == expected);
  run_ok("SET enable_duckdb_fallback = true;");
}

TEST_CASE_METHOD(GpuExecutionTableFunctionFixture,
                 "gpu_execution enable_optimizer leaves the session optimizer setting unchanged",
                 "[integration][gpu_execution][table_function]")
{
  auto const expected = query_rows(group_query());

  REQUIRE(session_optimizer_enabled());
  CHECK(query_rows(gpu_execution(group_query(), ", enable_optimizer = false")) == expected);
  CHECK(session_optimizer_enabled());

  run_ok("PRAGMA disable_optimizer;");
  CHECK(query_rows(gpu_execution(group_query(), ", enable_optimizer = true")) == expected);
  CHECK_FALSE(session_optimizer_enabled());
  run_ok("PRAGMA enable_optimizer;");
}
