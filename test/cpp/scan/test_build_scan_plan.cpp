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

// Unit tests for build_scan_plan's options surface, focused on the
// duckdb-native extras (rowid routing + per-column types) added on top of the
// parquet behavior. The parquet defaults must keep dropping rowid and leaving
// types unset.

#include <catch.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/constants.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <helper/logical_type.hpp>
#include <op/scan/scan_plan.hpp>
#include <sirius/exception.hpp>

#include <optional>
#include <string>

using namespace sirius;
using namespace sirius::op::scan;

namespace {

duckdb::ColumnIndex rowid_index() { return duckdb::ColumnIndex(duckdb::COLUMN_IDENTIFIER_ROW_ID); }

logical_type integer() { return logical_type::make(type_id::INTEGER); }
logical_type bigint() { return logical_type::make(type_id::BIGINT); }

// Resolves a column's type the way the pipeline converter does for the native
// path: rowid → BIGINT, everything else from the schema's returned_types.
build_scan_plan_options native_options(duckdb::vector<logical_type> const& returned_types)
{
  build_scan_plan_options opts;
  opts.decode_rowid_columns = true;
  opts.type_for = [&returned_types](duckdb::ColumnIndex const& ci) -> std::optional<logical_type> {
    if (ci.IsRowIdColumn()) { return bigint(); }
    return returned_types[ci.GetPrimaryIndex()];
  };
  return opts;
}

}  // namespace

TEST_CASE("build_scan_plan native routes rowid into data_columns with a type",
          "[scan][build_scan_plan]")
{
  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), rowid_index()};
  duckdb::vector<duckdb::idx_t> projection_ids;  // read all, both emitted
  duckdb::vector<std::string> names{"a"};
  duckdb::vector<logical_type> returned_types{integer()};
  duckdb::vector<duckdb::HivePartitioningIndex> no_partitions;

  auto plan = build_scan_plan(column_ids,
                              projection_ids,
                              names,
                              returned_types,
                              /*output_types_size=*/2,
                              no_partitions,
                              native_options(returned_types));

  REQUIRE(plan.data_columns.size() == 2);
  REQUIRE(plan.data_columns[0].reader_info.has_value());
  REQUIRE_FALSE(plan.data_columns[0].reader_info->is_rowid);
  REQUIRE(plan.data_columns[0].primary_idx == 0);
  REQUIRE(plan.data_columns[0].reader_info->type.id() == type_id::INTEGER);

  REQUIRE(plan.data_columns[1].reader_info.has_value());
  REQUIRE(plan.data_columns[1].reader_info->is_rowid);
  REQUIRE(plan.data_columns[1].reader_info->type.id() == type_id::BIGINT);

  REQUIRE(plan.output_layout.size() == 2);
  REQUIRE(plan.output_layout[0].source == scan_plan::output_entry::DATA);
  REQUIRE(plan.output_layout[0].idx == 0);
  REQUIRE(plan.output_layout[1].source == scan_plan::output_entry::DATA);
  REQUIRE(plan.output_layout[1].idx == 1);
}

TEST_CASE("build_scan_plan parquet defaults drop rowid and leave types unset",
          "[scan][build_scan_plan]")
{
  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), rowid_index()};
  duckdb::vector<duckdb::idx_t> projection_ids;
  duckdb::vector<std::string> names{"a"};
  duckdb::vector<logical_type> returned_types{integer()};
  duckdb::vector<duckdb::HivePartitioningIndex> no_partitions;

  // Default options: decode_rowid_columns == false, type_for empty.
  auto plan = build_scan_plan(
    column_ids, projection_ids, names, returned_types, /*output_types_size=*/2, no_partitions);

  REQUIRE(plan.data_columns.size() == 1);  // rowid dropped
  REQUIRE(plan.data_columns[0].primary_idx == 0);
  // parquet leaves reader_info unset and derives types from the footer.
  REQUIRE_FALSE(plan.data_columns[0].reader_info.has_value());

  REQUIRE(plan.output_layout.size() == 1);
  REQUIRE(plan.output_layout[0].source == scan_plan::output_entry::DATA);
  REQUIRE(plan.output_layout[0].idx == 0);
}

TEST_CASE("build_scan_plan native throws when type_for is missing", "[scan][build_scan_plan]")
{
  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0)};
  duckdb::vector<duckdb::idx_t> projection_ids;
  duckdb::vector<std::string> names{"a"};
  duckdb::vector<logical_type> returned_types{integer()};
  duckdb::vector<duckdb::HivePartitioningIndex> no_partitions;

  // decode_rowid_columns opts into the reader-typed path but no type_for is
  // supplied, so every data column ends up with no reader_decode_info —
  // build_scan_plan must reject this at build time rather than defer a null
  // deref to the decoder.
  build_scan_plan_options opts;
  opts.decode_rowid_columns = true;

  REQUIRE_THROWS_AS(build_scan_plan(column_ids,
                                    projection_ids,
                                    names,
                                    returned_types,
                                    /*output_types_size=*/1,
                                    no_partitions,
                                    opts),
                    sirius::internal_exception);
}

TEST_CASE("build_scan_plan keeps a pure-filter column out of the output layout",
          "[scan][build_scan_plan]")
{
  // projection_ids = {0, 1} with output_types_size = 1: column 0 is emitted,
  // column 1 is read for a filter only. Both land in data_columns; only column 0
  // is in output_layout.
  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  duckdb::vector<duckdb::idx_t> projection_ids{0, 1};
  duckdb::vector<std::string> names{"a", "b"};
  duckdb::vector<logical_type> returned_types{integer(), integer()};
  duckdb::vector<duckdb::HivePartitioningIndex> no_partitions;

  auto plan = build_scan_plan(column_ids,
                              projection_ids,
                              names,
                              returned_types,
                              /*output_types_size=*/1,
                              no_partitions,
                              native_options(returned_types));

  REQUIRE(plan.data_columns.size() == 2);
  REQUIRE(plan.output_layout.size() == 1);
  REQUIRE(plan.output_layout[0].source == scan_plan::output_entry::DATA);
  REQUIRE(plan.output_layout[0].idx == 0);

  auto const pure_filter = plan.pure_filter_batch_positions();
  REQUIRE(pure_filter.size() == 1);
  REQUIRE(pure_filter.count(1) == 1);  // batch position 1 (column b) is filter-only
}

TEST_CASE("build_scan_plan reorders data_columns and output to match projection order",
          "[scan][build_scan_plan]")
{
  // projection_ids = {2, 0, 1} reorders the output. data_columns end up in
  // projection order (D space); output_layout is the identity over that order;
  // batch_position_by_column_id maps each column_ids slot to its D position.
  duckdb::vector<duckdb::ColumnIndex> column_ids{
    duckdb::ColumnIndex(0), duckdb::ColumnIndex(1), duckdb::ColumnIndex(2)};
  duckdb::vector<duckdb::idx_t> projection_ids{2, 0, 1};
  duckdb::vector<std::string> names{"a", "b", "c"};
  duckdb::vector<logical_type> returned_types{integer(), integer(), integer()};
  duckdb::vector<duckdb::HivePartitioningIndex> no_partitions;

  auto plan = build_scan_plan(column_ids,
                              projection_ids,
                              names,
                              returned_types,
                              /*output_types_size=*/3,
                              no_partitions,
                              native_options(returned_types));

  REQUIRE(plan.data_columns.size() == 3);
  REQUIRE(plan.data_columns[0].primary_idx == 2);
  REQUIRE(plan.data_columns[1].primary_idx == 0);
  REQUIRE(plan.data_columns[2].primary_idx == 1);

  REQUIRE(plan.output_layout.size() == 3);
  for (std::size_t i = 0; i < 3; ++i) {
    REQUIRE(plan.output_layout[i].source == scan_plan::output_entry::DATA);
    REQUIRE(plan.output_layout[i].idx == i);
  }

  REQUIRE(plan.batch_position_by_column_id.size() == 3);
  REQUIRE(plan.batch_position_by_column_id[0] == std::optional<std::size_t>{1});
  REQUIRE(plan.batch_position_by_column_id[1] == std::optional<std::size_t>{2});
  REQUIRE(plan.batch_position_by_column_id[2] == std::optional<std::size_t>{0});
}
