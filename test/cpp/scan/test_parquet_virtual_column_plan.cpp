/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include <catch.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <op/scan/scan_plan.hpp>

namespace {

namespace scan = sirius::op::scan;

auto physical_types()
{
  return duckdb::vector<sirius::logical_type>{sirius::logical_type::make(sirius::type_id::INTEGER),
                                              sirius::logical_type::make(sirius::type_id::VARCHAR)};
}

std::vector<scan::bound_virtual_column> parquet_virtuals()
{
  return {
    {duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME,
     "filename",
     sirius::logical_type::make(sirius::type_id::VARCHAR)},
    {duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX,
     "file_index",
     sirius::logical_type::make(sirius::type_id::UBIGINT)},
    {duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER,
     "file_row_number",
     sirius::logical_type::make(sirius::type_id::BIGINT)},
  };
}

}  // namespace

TEST_CASE("parquet virtual scan plan assigns stable D M C and output positions",
          "[scan][parquet][virtual_columns][scan_plan]")
{
  auto const filename = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME;
  auto const file_idx = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX;
  auto const row_num  = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;

  // C = [physical x, filename, file_index, file_row_number]. The output asks
  // for filename twice and leaves file_row_number as a filter-only column.
  auto const plan = scan::build_scan_plan({duckdb::ColumnIndex(0),
                                           duckdb::ColumnIndex(filename),
                                           duckdb::ColumnIndex(file_idx),
                                           duckdb::ColumnIndex(row_num)},
                                          {1, 0, 2, 1, 3},
                                          {"x", "text"},
                                          physical_types(),
                                          /*output_types_size=*/4,
                                          {},
                                          parquet_virtuals());

  REQUIRE(plan.data_columns.size() == 1);
  CHECK(plan.data_columns[0].primary_idx == 0);
  REQUIRE(plan.virtual_columns.size() == 3);
  CHECK(plan.virtual_columns[0].materialized_idx == 1);
  CHECK(plan.virtual_columns[1].materialized_idx == 2);
  CHECK(plan.virtual_columns[2].materialized_idx == 3);

  REQUIRE(plan.batch_position_by_column_id.size() == 4);
  CHECK(plan.batch_position_by_column_id[0] == 0);
  CHECK(plan.batch_position_by_column_id[1] == 1);
  CHECK(plan.batch_position_by_column_id[2] == 2);
  CHECK(plan.batch_position_by_column_id[3] == 3);

  REQUIRE(plan.output_layout.size() == 4);
  CHECK(plan.output_layout[0].idx == 1);
  CHECK(plan.output_layout[1].idx == 0);
  CHECK(plan.output_layout[2].idx == 2);
  CHECK(plan.output_layout[3].idx == 1);
}

TEST_CASE("parquet virtual-only plan chooses a VARCHAR carrier before assigning M",
          "[scan][parquet][virtual_columns][scan_plan][carrier]")
{
  auto const filename = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME;
  auto const row_num  = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
  auto const file_idx = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX;
  duckdb::vector<sirius::logical_type> types{sirius::logical_type::make(sirius::type_id::VARCHAR)};

  auto const plan = scan::build_scan_plan(
    {duckdb::ColumnIndex(filename), duckdb::ColumnIndex(file_idx), duckdb::ColumnIndex(row_num)},
    {},
    {"only_text"},
    types,
    3,
    {},
    parquet_virtuals());

  REQUIRE(plan.data_columns.size() == 1);
  CHECK(plan.carrier_batch_index == 0);
  CHECK(plan.data_columns[0].primary_idx == 0);
  REQUIRE(plan.virtual_columns.size() == 3);
  CHECK(plan.virtual_columns[0].materialized_idx == 1);
  CHECK(plan.virtual_columns[1].materialized_idx == 2);
  CHECK(plan.virtual_columns[2].materialized_idx == 3);
  REQUIRE(plan.output_layout.size() == 3);
  CHECK(plan.output_layout[0].idx == 1);
  CHECK(plan.output_layout[1].idx == 2);
  CHECK(plan.output_layout[2].idx == 3);
}

TEST_CASE("parquet scan plan rejects an unknown bound virtual id",
          "[scan][parquet][virtual_columns][scan_plan]")
{
  auto const unknown = duckdb::VIRTUAL_COLUMN_START + 99;
  std::vector<scan::bound_virtual_column> definitions{
    {unknown, "future_metadata", sirius::logical_type::make(sirius::type_id::BIGINT)}};

  CHECK_THROWS_AS(scan::build_scan_plan({duckdb::ColumnIndex(unknown)},
                                        {},
                                        {"x"},
                                        {sirius::logical_type::make(sirius::type_id::INTEGER)},
                                        1,
                                        {},
                                        definitions),
                  duckdb::NotImplementedException);
}
