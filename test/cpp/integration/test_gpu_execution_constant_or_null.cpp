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

// DuckDB's optimizer rewrites expressions into constant_or_null(c, a1, ..., an) when column
// statistics decide them, for example `x <= 200` on a column whose max is 99. These tests check
// that the rewritten queries run on the GPU and match the CPU, including rows where an argument
// is NULL.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <string>

namespace {

class ConstantOrNullFixture : public sirius::test::GpuExecutionFixture {
 public:
  ConstantOrNullFixture()
  {
    // x is in [0, 99] and d in [0.25, 199.25], both with NULLs.
    run_ok(
      "CREATE TABLE t AS SELECT i::INTEGER AS i,"
      " CASE WHEN i % 7 = 0 THEN NULL ELSE (i % 100)::INTEGER END AS x,"
      " CASE WHEN i % 11 = 0 THEN NULL ELSE ((i % 200) + 0.25)::DECIMAL(7, 2) END AS d"
      " FROM range(3000) r(i);");
    run_ok("CHECKPOINT;");
    auto const pq = dir_.file_literal("t.parquet");
    run_ok("COPY t TO " + pq + " (FORMAT PARQUET);");
    run_ok("CREATE VIEW p AS SELECT * FROM read_parquet(" + pq + ");");
  }

  /// Requires that DuckDB's optimized plan for @p query contains constant_or_null, so the
  /// comparison below exercises it.
  void require_constant_or_null(std::string const& query)
  {
    auto result = con->Query("EXPLAIN " + query);
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    REQUIRE(result->ToString().find("constant_or_null") != std::string::npos);
  }

  void compare(std::string const& query)
  {
    CAPTURE(query);
    require_constant_or_null(query);
    compare_gpu_vs_cpu(query);
  }

 private:
  sirius::test::scratch_dir dir_{"constant_or_null"};
};

}  // namespace

TEST_CASE_METHOD(ConstantOrNullFixture,
                 "constant_or_null in a filter pushed into the scan",
                 "[integration][gpu_execution][constant_or_null]")
{
  for (auto const* table : {"t", "p"}) {
    compare(std::string("SELECT i FROM ") + table +
            " WHERE (d BETWEEN 10 AND 20) OR (d BETWEEN 150 AND 250);");
  }
}

TEST_CASE_METHOD(ConstantOrNullFixture,
                 "constant_or_null in a filter over several columns",
                 "[integration][gpu_execution][constant_or_null]")
{
  for (auto const* table : {"t", "p"}) {
    compare(std::string("SELECT i FROM ") + table + " WHERE x > 500 OR i < 5;");
    compare(std::string("SELECT i FROM ") + table + " WHERE x <= 200 OR i < 5;");
  }
}

TEST_CASE_METHOD(ConstantOrNullFixture,
                 "constant_or_null in a projection keeps NULL rows",
                 "[integration][gpu_execution][constant_or_null]")
{
  compare("SELECT i, x <= 200 FROM t;");
  compare("SELECT i, x * 0 FROM t;");
}
