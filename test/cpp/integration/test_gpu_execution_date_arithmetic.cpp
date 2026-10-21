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
#include <duckdb/common/types/date.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace {
class DateArithmeticFixture : public sirius::test::GpuExecutionFixture {
 public:
  void strategy(std::string const& value)
  {
    run_ok("SET gpu_execution = false");
    run_ok("SET enable_duckdb_fallback = false");
    run_ok("SET expression_evaluator_strategy = '" + value + "'");
  }
  void seed()
  {
    run_ok("CREATE TABLE dates(id INTEGER, a DATE, b DATE, n INTEGER)");
    run_ok(
      "INSERT INTO dates VALUES (1,'1969-12-31','1970-01-05',-7), "
      "(2,'2000-02-29','1999-12-31',5), (3,'infinity','-infinity',2147483647), "
      "(4,'-infinity','infinity',-2147483648), (5,NULL,'2001-01-01',3), "
      "(6,'2001-01-01',NULL,NULL), (7,'infinity',NULL,NULL), "
      "(8,'-infinity','-infinity',0), (9,NULL,'2001-01-01',2147483647)");
    run_ok("CHECKPOINT");
  }
};
}  // namespace

TEST_CASE_METHOD(DateArithmeticFixture,
                 "DATE arithmetic preserves values NULLs and infinities",
                 "[integration][gpu_execution][date_arithmetic]")
{
  strategy(GENERATE("materialize", "ast_interpret", "ast_jit"));
  seed();
  compare_gpu_vs_cpu("SELECT id,a+n,n+a,a-n,a-b FROM dates");
  compare_gpu_vs_cpu("SELECT id,a+5,5+a,a-5,a-DATE '1970-01-01',DATE '1970-01-01'-a FROM dates");
  compare_gpu_vs_cpu(
    "SELECT id,DATE '2000-01-01'+n,n+DATE '2000-01-01',DATE '2000-01-01'-n "
    "FROM dates WHERE id IN (1,2,5,6,7,8)");
  compare_gpu_vs_cpu("SELECT id,a+n FROM dates WHERE (a+n)>b AND (a-n)<DATE '2001-01-01'");
  compare_gpu_vs_cpu("SELECT l.id,r.id FROM dates l JOIN dates r ON l.a+5=r.b-2");
  run_ok("SET gpu_execution = false");
  run_ok("CREATE TABLE empty_dates AS SELECT * FROM dates WHERE false");
  run_ok("CHECKPOINT");
  compare_gpu_vs_cpu("SELECT a+n,n+a,a-n,a-b FROM empty_dates");
}

TEST_CASE_METHOD(DateArithmeticFixture,
                 "DATE difference widens before subtraction",
                 "[integration][gpu_execution][date_arithmetic]")
{
  strategy(GENERATE("materialize", "ast_interpret", "ast_jit"));
  run_ok("CREATE TABLE dates(a DATE,b DATE,n INTEGER)");
  auto const high = duckdb::Value::DATE(duckdb::date_t{std::numeric_limits<int32_t>::max() - 1});
  auto const low  = duckdb::Value::DATE(duckdb::date_t{std::numeric_limits<int32_t>::min()});
  auto statement  = con->Prepare("INSERT INTO dates VALUES (?, ?, 0)");
  REQUIRE_FALSE(statement->HasError());
  auto forward = statement->Execute(high, low);
  REQUIRE_FALSE(forward->HasError());
  auto reverse = statement->Execute(low, high);
  REQUIRE_FALSE(reverse->HasError());
  run_ok("CHECKPOINT");
  compare_gpu_vs_cpu("SELECT a-b,b-a,a+n,n+a,a-n FROM dates");
}

TEST_CASE_METHOD(DateArithmeticFixture,
                 "DATE offsets reject overflow and sentinel collision",
                 "[integration][gpu_execution][date_arithmetic]")
{
  auto const operation = GENERATE("a+n", "n+a", "a-n");
  auto const edge      = GENERATE(0, 1, 2, 3);
  strategy(GENERATE("materialize", "ast_interpret", "ast_jit"));
  bool const subtract = std::string(operation) == "a-n";
  int32_t const days =
    edge < 2 ? std::numeric_limits<int32_t>::max() - 1 : std::numeric_limits<int32_t>::min();
  int const delta = edge == 0 ? 1 : edge == 1 ? 2 : edge == 2 ? -1 : 1;
  run_ok("CREATE TABLE dates(a DATE,n INTEGER)");
  run_ok("INSERT INTO dates SELECT DATE '2000-01-01',1 FROM range(5000)");
  // DATE's INT32_MIN carrier is finite but does not round-trip via its text parser.
  auto statement = con->Prepare("INSERT INTO dates VALUES (?, ?)");
  REQUIRE_FALSE(statement->HasError());
  auto inserted = statement->Execute(duckdb::Value::DATE(duckdb::date_t{days}),
                                     duckdb::Value::INTEGER(subtract ? -delta : delta));
  REQUIRE_FALSE(inserted->HasError());
  run_ok("CHECKPOINT");
  auto const query = "SELECT " + std::string(operation) + " FROM dates";
  for (bool gpu : {false, true}) {
    run_ok(std::string("SET gpu_execution = ") + (gpu ? "true" : "false"));
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto result = con->Query(query);
    REQUIRE(result);
    REQUIRE(result->HasError());
    REQUIRE(result->GetError().find("Date out of range") != std::string::npos);
    auto after = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(after.fallbacks == before.fallbacks);
    REQUIRE(after.runtime_fallbacks == before.runtime_fallbacks);
    if (gpu) { REQUIRE(after.successful_rebinds == before.successful_rebinds + 1); }
  }
}
