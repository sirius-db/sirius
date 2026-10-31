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

#include "planner/scan_returned_types.hpp"

#include <catch.hpp>
#include <duckdb/common/optional_idx.hpp>

TEST_CASE("scan schema conversion ignores only unread unsupported types",
          "[planner][scan_returned_types]")
{
  using duckdb::LogicalType;
  auto const unsupported = GENERATE(Catch::Generators::as<LogicalType>{},
                                    LogicalType::BLOB,
                                    LogicalType::UUID,
                                    LogicalType::VARIANT(),
                                    LogicalType::TIME,
                                    LogicalType::TIMESTAMP_TZ,
                                    LogicalType::INTERVAL,
                                    LogicalType::ARRAY(LogicalType::BLOB, duckdb::optional_idx(2)));
  duckdb::vector<LogicalType> types{
    unsupported, LogicalType::INTEGER, unsupported, LogicalType::VARCHAR, LogicalType::SMALLINT};
  // Reordered scan slots include a column used only by a filter.
  duckdb::vector<duckdb::ColumnIndex> columns{
    duckdb::ColumnIndex(3), duckdb::ColumnIndex(1), duckdb::ColumnIndex(4)};
  auto converted = sirius::planner::scan_returned_types(types, columns);
  REQUIRE(converted.size() == types.size());
  CHECK(converted[0].id() == sirius::type_id::INVALID);
  CHECK(converted[1].id() == sirius::type_id::INTEGER);
  CHECK(converted[2].id() == sirius::type_id::INVALID);
  CHECK(converted[3].id() == sirius::type_id::VARCHAR);
  CHECK(converted[4].id() == sirius::type_id::SMALLINT);

  columns.emplace_back(2);
  CHECK_THROWS_AS(sirius::planner::scan_returned_types(types, columns),
                  duckdb::InvalidInputException);
  // The non-projection-pushdown path must still validate the whole schema.
  CHECK_THROWS_AS(sirius::from_duckdb_vec(types), duckdb::InvalidInputException);
}

TEST_CASE("scan schema conversion retains row count carriers and rejects invalid indices",
          "[planner][scan_returned_types]")
{
  using duckdb::LogicalType;
  duckdb::vector<LogicalType> types{
    LogicalType::BLOB, LogicalType::SMALLINT, LogicalType::VARCHAR, LogicalType::DECIMAL(10, 2)};
  auto converted = sirius::planner::scan_returned_types(
    types, {duckdb::ColumnIndex(duckdb::COLUMN_IDENTIFIER_ROW_ID)});
  CHECK_FALSE(converted[0].is_fixed_width());
  CHECK(converted[1].fixed_width_byte_size() == 2);
  CHECK(converted[2].id() == sirius::type_id::VARCHAR);
  CHECK(converted[3].decimal_precision() == 10);
  CHECK(converted[3].decimal_scale() == 2);
  CHECK_THROWS_AS(sirius::planner::scan_returned_types(types, {duckdb::ColumnIndex(types.size())}),
                  duckdb::InternalException);
}
