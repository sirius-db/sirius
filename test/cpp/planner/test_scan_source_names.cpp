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

#include "planner/scan_source_names.hpp"

#include <catch.hpp>
#include <duckdb/common/multi_file/multi_file_function.hpp>

#include <string>
#include <utility>

TEST_CASE("scan source names fall back to LogicalGet::names when the bind's column count differs",
          "[planner][scan_source_names]")
{
  // Either bind is built without names, so it cannot name the scan's one column. Without the count
  // check the helper would return that empty list.
  auto const names_for = [](duckdb::unique_ptr<duckdb::FunctionData> bind) {
    duckdb::LogicalGet const get(duckdb::TableIndex(0),
                                 duckdb::BoundTableFunction(duckdb::TableFunction()),
                                 std::move(bind),
                                 duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT},
                                 duckdb::vector<duckdb::Identifier>{"k"});
    return sirius::planner::scan_source_names(get);
  };
  duckdb::vector<std::string> const expected{"k"};
  CHECK(names_for(duckdb::make_uniq<duckdb::MultiFileBindData>()) == expected);
  CHECK(names_for(duckdb::make_uniq<duckdb::SiriusReadParquetBindData>("s3://bucket/a.parquet",
                                                                       1)) == expected);
}
