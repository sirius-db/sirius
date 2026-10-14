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

#include "planner/sirius_physical_plan_generator.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/expression_executor.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/table_filter_test_utils.hpp>

TEST_CASE("unpushed scan filters follow the emitted projection",
          "[planner][scan_filter_projection]")
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  duckdb::TableFunction function;
  function.supports_pushdown_type = [](duckdb::FunctionData const&, duckdb::idx_t primary) {
    return primary != 2;
  };
  duckdb::LogicalGet scan(
    duckdb::TableIndex(0),
    duckdb::BoundTableFunction(function),
    duckdb::make_uniq<duckdb::TableFunctionData>(),
    {duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER},
    {"a", "b", "c"});
  // Scan slots [c, a, b]; explicit projection emits [b, c].
  scan.AddColumnId(2);
  scan.AddColumnId(0);
  scan.AddColumnId(1);
  auto const reordered = GENERATE(false, true);
  if (reordered) { scan.projection_ids = {duckdb::ProjectionIndex(2), duckdb::ProjectionIndex(0)}; }

  duckdb::TableFilterSet filters;
  filters.PushFilter(duckdb::ProjectionIndex(0),
                     sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_EQUAL,
                                                   duckdb::Value::INTEGER(10)));
  filters.PushFilter(duckdb::ProjectionIndex(2),
                     sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHAN,
                                                   duckdb::Value::INTEGER(0)));
  auto predicates = sirius::planner::extract_unpushed_scan_filters(scan, filters);
  REQUIRE(predicates.size() == 1);
  REQUIRE(filters.FilterCount() == 1);
  REQUIRE(filters.HasFilter(duckdb::ProjectionIndex(2)));

  duckdb::idx_t const width = reordered ? 2 : 3;
  duckdb::idx_t const c     = reordered ? 1 : 0;
  duckdb::DataChunk input;
  input.Initialize(duckdb::Allocator::DefaultAllocator(),
                   duckdb::vector<duckdb::LogicalType>(width, duckdb::LogicalType::INTEGER));
  input.SetChildCardinality(3);
  for (duckdb::idx_t col = 0; col < width; ++col) {
    input.data[col].SetValue(0, duckdb::Value::INTEGER(11));
    input.data[col].SetValue(1, duckdb::Value::INTEGER(10));
    input.data[col].SetValue(2, duckdb::Value::INTEGER(99));
  }
  input.data[c].SetValue(0, duckdb::Value::INTEGER(10));
  input.data[c].SetValue(1, duckdb::Value::INTEGER(20));
  input.data[c].SetValue(2, duckdb::Value(duckdb::LogicalType::INTEGER));
  duckdb::DataChunk result;
  result.Initialize(duckdb::Allocator::DefaultAllocator(), {duckdb::LogicalType::BOOLEAN});
  duckdb::ExpressionExecutor executor(*con.context, predicates);
  executor.Execute(input, result);
  REQUIRE(result.size() == 3);
  CHECK(result.GetValue(0, 0) == duckdb::Value::BOOLEAN(true));
  CHECK(result.GetValue(0, 1) == duckdb::Value::BOOLEAN(false));
  CHECK(result.GetValue(0, 2).IsNull());
}
