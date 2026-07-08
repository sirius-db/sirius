/*
 * Copyright 2025, Sirius Contributors.
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

// Companion to test/cpp/operator/test_orders_count_two_pipelines.cpp.
//
// That test hand-builds the plan (no parser) and verifies the two-pipeline SPLIT the course
// diagram shows, but stops short of executing it. This test closes the loop end to end: it runs
// the actual `SELECT count(*) FROM orders WHERE amount > 100` over a generated Parquet file on the
// GPU (fallback disabled, so a silent CPU fallback cannot mask a GPU failure) and asserts the
// scalar result is 3 — the same two-pipeline machine, driven the normal way, producing the answer.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/sirius_test_env.hpp>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

// A small `orders(o_orderkey BIGINT, amount DOUBLE)` Parquet; exactly 3 of 6 rows have amount > 100
// (100.01, 250.0, 500.0), and the boundary row amount == 100.0 is excluded by the strict `>`.
// SIRIUS_DISABLE=1 keeps the extension from building a SiriusContext on the throwaway writer DB.
void generate_orders_parquet(const fs::path& path)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(nullptr);
    duckdb::Connection gen(gen_db);
    auto r = gen.Query(
      "COPY (SELECT * FROM (VALUES "
      "  (1::BIGINT,  50.0::DOUBLE), (2::BIGINT, 100.0::DOUBLE), (3::BIGINT, 100.01::DOUBLE), "
      "  (4::BIGINT, 250.0::DOUBLE), (5::BIGINT,  99.99::DOUBLE), (6::BIGINT, 500.0::DOUBLE)) "
      "  AS t(o_orderkey, amount)) TO '" +
      path.string() + "' (FORMAT PARQUET);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }
  unsetenv("SIRIUS_DISABLE");
}

}  // namespace

TEST_CASE("SELECT count(*) FROM orders WHERE amount > 100 over Parquet runs on GPU and returns 3",
          "[integration][orders_count]")
{
  auto tmp = fs::temp_directory_path() / ("sirius-orders-count-int-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);
  auto parquet_path = tmp / "orders.parquet";
  generate_orders_parquet(parquet_path);

  // [integration]-tagged tests run with the shared integration env active (see the shared_env
  // listener in test/cpp/unittest.cpp). Guard defensively for direct/standalone runs.
  if (!(sirius::test::g_integration_env && sirius::test::g_integration_env->is_active())) {
    WARN("integration env not active; skipping GPU count test");
    fs::remove_all(tmp, ec);
    return;
  }
  auto con_owner =
    std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
  auto& con = *con_owner;

  const std::string query =
    "SELECT count(*) FROM read_parquet('" + parquet_path.string() + "') WHERE amount > 100;";

  // Force GPU execution: with fallback off, a query Sirius cannot run on the GPU errors loudly
  // instead of silently returning DuckDB's CPU answer — so a passing assertion means the GPU
  // two-pipeline path (scan -> filter -> partial count | FULL | merge -> collector) produced it.
  REQUIRE_FALSE(con.Query("SET gpu_execution = true;")->HasError());
  REQUIRE_FALSE(con.Query("SET enable_duckdb_fallback = false;")->HasError());

  auto gpu_result = con.Query(query);
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) { UNSCOPED_INFO("GPU execution error: " << gpu_result->GetError()); }
  REQUIRE_FALSE(gpu_result->HasError());
  REQUIRE(gpu_result->GetValue(0, 0).ToString() == "3");

  // Cross-check against DuckDB's CPU result for the same query.
  REQUIRE_FALSE(con.Query("SET enable_duckdb_fallback = true;")->HasError());
  REQUIRE_FALSE(con.Query("SET gpu_execution = false;")->HasError());
  auto cpu_result = con.Query(query);
  REQUIRE(cpu_result);
  REQUIRE_FALSE(cpu_result->HasError());
  REQUIRE(cpu_result->GetValue(0, 0).ToString() == "3");

  fs::remove_all(tmp, ec);
}
