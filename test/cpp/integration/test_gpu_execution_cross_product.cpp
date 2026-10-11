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
 * @file test_gpu_execution_cross_product.cpp
 * @brief Verifies that DuckDB cross products run on the GPU and match the CPU.
 *
 * A LOGICAL_CROSS_PRODUCT is planned as a nested loop join without conditions. Cross products
 * whose estimated output, the product of the estimated input rows, exceeds the row limit of a
 * cuDF column are refused at plan time. A pair of input batches whose cross join exceeds the GPU
 * memory limit fails at runtime before allocating its output.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>
#include <string_view>

using CrossProductFixture = sirius::test::GpuExecutionFixture;

namespace {

constexpr std::string_view kRejection = "Cross product of an estimated";

void create_cross_tables(CrossProductFixture& fx)
{
  fx.run_ok("CREATE TABLE cp_a AS SELECT i::INTEGER x FROM range(4) r(i);");
  fx.run_ok(
    "CREATE TABLE cp_b AS SELECT i::BIGINT y, (i * 1.5)::DOUBLE z, 'b' || i::VARCHAR s "
    "FROM range(3) r(i);");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(CrossProductFixture,
                 "cross product - two tables run on the GPU",
                 "[integration][gpu_execution][cross_product]")
{
  create_cross_tables(*this);
  SECTION("CROSS JOIN") { compare_gpu_vs_cpu("SELECT * FROM cp_a CROSS JOIN cp_b;"); }
  SECTION("comma join") { compare_gpu_vs_cpu("SELECT x, s FROM cp_a, cp_b;"); }
  SECTION("JOIN ON true") { compare_gpu_vs_cpu("SELECT y, x FROM cp_a JOIN cp_b ON true;"); }
}

TEST_CASE_METHOD(CrossProductFixture,
                 "cross product - scalar aggregate subqueries run on the GPU",
                 "[integration][gpu_execution][cross_product]")
{
  // The shape of TPC-DS Q88 and Q90: single-row aggregates combined by a cross product.
  create_cross_tables(*this);
  compare_gpu_vs_cpu(
    "SELECT * FROM (SELECT count(*) c1 FROM cp_a WHERE x < 2) s1, "
    "(SELECT sum(y) c2 FROM cp_b WHERE y >= 1) s2, "
    "(SELECT max(z) c3 FROM cp_b) s3;");
}

TEST_CASE_METHOD(CrossProductFixture,
                 "cross product - count(*) without projected columns runs on the GPU",
                 "[integration][gpu_execution][cross_product]")
{
  create_cross_tables(*this);
  compare_gpu_vs_cpu("SELECT count(*) FROM cp_a, cp_b;");
}

TEST_CASE_METHOD(CrossProductFixture,
                 "cross product - an empty side yields no rows",
                 "[integration][gpu_execution][cross_product]")
{
  create_cross_tables(*this);
  compare_gpu_vs_cpu("SELECT * FROM cp_a, (SELECT * FROM cp_b WHERE y > 100) e;");
}

TEST_CASE_METHOD(CrossProductFixture,
                 "cross product - uncorrelated EXISTS runs on the GPU",
                 "[integration][gpu_execution][cross_product]")
{
  // DuckDB plans an uncorrelated EXISTS as a cross product with a count over LIMIT 1.
  create_cross_tables(*this);
  SECTION("EXISTS, rows")
  {
    compare_gpu_vs_cpu("SELECT * FROM cp_a WHERE EXISTS (SELECT 1 FROM cp_b);");
  }
  SECTION("EXISTS, no rows")
  {
    compare_gpu_vs_cpu("SELECT * FROM cp_a WHERE EXISTS (SELECT 1 FROM cp_b WHERE y > 100);");
  }
  SECTION("NOT EXISTS, rows")
  {
    compare_gpu_vs_cpu("SELECT * FROM cp_a WHERE NOT EXISTS (SELECT 1 FROM cp_b);");
  }
  SECTION("NOT EXISTS, no rows")
  {
    compare_gpu_vs_cpu("SELECT * FROM cp_a WHERE NOT EXISTS (SELECT 1 FROM cp_b WHERE y > 100);");
  }
}

TEST_CASE_METHOD(CrossProductFixture,
                 "cross product - feeds a grouped aggregate",
                 "[integration][gpu_execution][cross_product]")
{
  create_cross_tables(*this);
  compare_gpu_vs_cpu("SELECT x, sum(y) s, count(*) c FROM cp_a, cp_b GROUP BY x;");
}

TEST_CASE_METHOD(CrossProductFixture,
                 "cross product - sides split into several batches",
                 "[integration][gpu_execution][cross_product]")
{
  run_ok("CREATE TABLE cp_big AS SELECT i::INTEGER x FROM range(200000) r(i);");
  run_ok("CREATE TABLE cp_small AS SELECT i::INTEGER y FROM range(50) r(i);");
  run_ok("CHECKPOINT;");
  sirius::test::scoped_setting const scan_batch{*con, "scan_task_batch_size", 65536};
  sirius::test::scoped_setting const concat_batch{*con, "concat_batch_bytes", 65536};
  compare_gpu_vs_cpu("SELECT count(*) c, sum(x + y) s, min(x - y) m FROM cp_big, cp_small;");
}

TEST_CASE_METHOD(CrossProductFixture,
                 "cross product - an estimated output beyond the cuDF row limit is refused",
                 "[integration][gpu_execution][cross_product]")
{
  // 100000 x 100000 rows exceeds cudf::size_type. With the fallback disabled the query must fail
  // at plan time, so nothing is executed.
  run_ok("CREATE TABLE cp_l AS SELECT i::INTEGER x FROM range(100000) r(i);");
  run_ok("CREATE TABLE cp_r AS SELECT i::INTEGER y FROM range(100000) r(i);");
  run_ok("CHECKPOINT;");
  run_ok("SET gpu_execution = true;");
  run_ok("SET enable_duckdb_fallback = false;");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto result       = con->Query("SELECT count(*) FROM cp_l, cp_r;");
  auto const after  = sirius::test::get_transparent_execution_stats(*con);
  run_ok("SET enable_duckdb_fallback = true;");
  REQUIRE(result);
  INFO(result->ToString());
  REQUIRE(result->HasError());
  REQUIRE(result->GetError().find("GPU plan generation failed") != std::string::npos);
  REQUIRE(result->GetError().find(kRejection) != std::string::npos);
  REQUIRE(after.executions == before.executions);
}

TEST_CASE_METHOD(CrossProductFixture,
                 "cross product - an output beyond GPU memory fails without retrying",
                 "[integration][gpu_execution][cross_product]")
{
  // DuckDB estimates each range filter at 20% of the table, so the plan estimate of 20000 x 20000
  // rows passes the plan-time guard. The real 46000 x 46000 rows of eight BIGINT columns need
  // about 135 GB, which no GPU memory space holds.
  run_ok(
    "CREATE TABLE cp_wide_l AS SELECT i::BIGINT a1, i::BIGINT a2, i::BIGINT a3, i::BIGINT a4 "
    "FROM range(100000) r(i);");
  run_ok(
    "CREATE TABLE cp_wide_r AS SELECT i::BIGINT b1, i::BIGINT b2, i::BIGINT b3, i::BIGINT b4 "
    "FROM range(100000) r(i);");
  run_ok("CHECKPOINT;");
  run_ok("SET gpu_execution = true;");
  run_ok("SET enable_duckdb_fallback = false;");
  auto result = con->Query(
    "SELECT max(a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4) "
    "FROM (SELECT * FROM cp_wide_l WHERE a1 < 46000), (SELECT * FROM cp_wide_r WHERE b1 < 46000);");
  run_ok("SET enable_duckdb_fallback = true;");
  REQUIRE(result);
  REQUIRE(result->HasError());
  INFO(result->GetError());
  REQUIRE(result->GetError().find("Cross join of 46000 x 46000 rows needs") != std::string::npos);
}
