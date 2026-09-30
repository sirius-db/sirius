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

// The reset_sirius_cache() table function, exercised through the catalog the way
// a user reaches it.  What the reset does to the caches themselves is covered by
// test/cpp/scan_manager/test_reset_caches.cpp; this is about the DuckDB surface
// -- that it is registered, binds with no arguments, runs under the execution
// slot, and reports success.
//
// The integration config carries no `cache:` section, so caching is off and this
// runs the no-op branch: the reset must still succeed rather than fault on a
// context that has no cache to drop.

#include "catch.hpp"
#include "utils/gpu_execution_fixture.hpp"

#include <string>

namespace {

/// Run @p sql expecting a single BOOLEAN `true` row -- the shape every Sirius
/// control table function returns.
void expect_success_row(duckdb::Connection& con, std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());

  auto& materialized = result->Cast<duckdb::MaterializedQueryResult>();
  REQUIRE(materialized.RowCount() == 1);
  REQUIRE(materialized.GetValue(0, 0) == duckdb::Value::BOOLEAN(true));
}

}  // namespace

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "reset_sirius_cache is registered and reports success",
                 "[integration][reset_sirius_cache]")
{
  expect_success_row(*con, "CALL reset_sirius_cache();");
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "reset_sirius_cache can be called repeatedly",
                 "[integration][reset_sirius_cache]")
{
  // Nothing about a reset is one-shot: a benchmark loop calls it once per
  // iteration, so back-to-back calls must behave the same as the first.
  expect_success_row(*con, "CALL reset_sirius_cache();");
  expect_success_row(*con, "CALL reset_sirius_cache();");
  expect_success_row(*con, "CALL reset_sirius_cache();");
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "reset_sirius_cache leaves the connection able to query",
                 "[integration][reset_sirius_cache]")
{
  run_ok("CREATE TABLE reset_probe AS SELECT range AS i FROM range(1000);");
  expect_success_row(*con, "CALL reset_sirius_cache();");

  // The reset drops IO-layer state, not catalog or query state: a query after
  // it must still work.  This is the check that would catch a reset that tore
  // down something a live connection depends on.
  auto result = con->Query("SELECT count(*) FROM reset_probe;");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  auto& materialized = result->Cast<duckdb::MaterializedQueryResult>();
  REQUIRE(materialized.RowCount() == 1);
  CHECK(materialized.GetValue(0, 0).GetValue<std::int64_t>() == 1000);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "reset_sirius_cache takes no arguments",
                 "[integration][reset_sirius_cache]")
{
  auto result = con->Query("CALL reset_sirius_cache('some-argument');");
  REQUIRE(result);
  CHECK(result->HasError());
}
