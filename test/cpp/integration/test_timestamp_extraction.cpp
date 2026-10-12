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
#include <config.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>
#include <vector>

namespace {

class TimestampExtractionFixture : public sirius::test::GpuExecutionFixture {
 public:
  ~TimestampExtractionFixture()
  {
    duckdb::Config::EXPRESSION_EVALUATOR_STRATEGY = original_strategy_;
  }

  void check_extraction(std::string const& query)
  {
    INFO(query);
    run_ok("SET gpu_execution = false");
    auto cpu = con->Query(query);
    REQUIRE(cpu);
    REQUIRE_FALSE(cpu->HasError());
    auto expected = collect_rows(cpu->Cast<duckdb::MaterializedQueryResult>());
    for (auto const& type : cpu->types) {
      REQUIRE(type == duckdb::LogicalType::BIGINT);
    }

    run_ok("SET gpu_execution = true");
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto gpu    = con->Query(query);
    auto after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(gpu);
    if (gpu->HasError()) { UNSCOPED_INFO(gpu->GetError()); }
    REQUIRE_FALSE(gpu->HasError());
    // A CPU fallback must not turn this regression green.
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    REQUIRE(gpu->types == cpu->types);
    CHECK(collect_rows(gpu->Cast<duckdb::MaterializedQueryResult>()) == expected);
  }

 private:
  sirius::expression_evaluator_strategy original_strategy_ =
    duckdb::Config::EXPRESSION_EVALUATOR_STRATEGY;
};

}  // namespace

TEST_CASE_METHOD(TimestampExtractionFixture,
                 "timestamp millisecond and microsecond extraction matches DuckDB",
                 "[integration][gpu_execution][timestamp_extraction]")
{
  auto const type     = GENERATE("TIMESTAMP_S", "TIMESTAMP_MS", "TIMESTAMP", "TIMESTAMP_NS");
  auto const strategy = GENERATE("materialize", "ast_interpret", "ast_jit");
  CAPTURE(type, strategy);
  run_ok("SET gpu_execution = false");
  run_ok(std::string("SET expression_evaluator_strategy = '") + strategy + "'");
  run_ok(std::string("CREATE TABLE timestamps (id BIGINT, t ") + type + ")");

  // Materialized table columns prevent constant folding of the extraction.
  // Keep sub-microsecond digits for NS: its implicit cast to TIMESTAMP truncates
  // toward zero, including just before the epoch and before a minute boundary.
  std::vector<std::string> const values = {
    "1970-01-01 00:00:23.040000",
    "1970-01-01 00:00:07.739523",
    "1970-01-01 00:00:59.999999999",
    "1970-01-01 00:01:00",
    "1970-01-01 00:01:00.000001001",
    "1970-01-01 00:00:00.000000001",
    "1970-01-01 00:00:00",
    "1969-12-31 23:59:59.999999999",
    "1969-12-31 23:59:59.999999000",
    "1969-12-31 23:59:59.999998999",
    "1969-12-31 23:59:59.999000001",
    "1969-12-31 23:58:59.999999999",
    "1969-12-31 23:59:00",
    "1969-12-31 23:59:00.000000001",
    "1969-12-31 23:59:07.739523",
    "1900-01-01 12:34:23.040523",
    "2000-02-29 12:34:59.999999",
    "infinity",
    "-infinity",
  };
  for (size_t i = 0; i < values.size(); ++i) {
    run_ok("INSERT INTO timestamps VALUES (" + std::to_string(i) + ", '" + values[i] + "')");
  }
  run_ok("INSERT INTO timestamps VALUES (19, NULL), (20, NULL)");
  run_ok("CHECKPOINT");

  check_extraction("SELECT id, millisecond(t), microsecond(t) FROM timestamps");
  check_extraction(
    "SELECT id, millisecond(CASE WHEN id % 2 = 0 THEN t ELSE NULL END), "
    "microsecond(CASE WHEN id % 2 = 1 THEN t ELSE NULL END) FROM timestamps");
  // Exercise extraction as an AST breaker beneath arithmetic and a filter.
  check_extraction(
    "SELECT id, millisecond(t) * 1000 + microsecond(t), microsecond(t) + 1 "
    "FROM timestamps");
  check_extraction(
    "SELECT id, millisecond(t), microsecond(t) FROM timestamps "
    "WHERE millisecond(t) >= 23000 AND microsecond(t) < 59999999");
}
