/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <catch.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <op/scan/scan_plan.hpp>
#include <sirius/exception.hpp>

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

TEST_CASE("parquet legacy virtual columns map ordinary primary indexes into M",
          "[scan][parquet][virtual_columns][scan_plan][virtual_review]")
{
  auto definitions = parquet_virtuals();
  definitions.push_back({2,
                         "source_path",
                         sirius::logical_type::make(sirius::type_id::VARCHAR),
                         scan::scan_plan::parquet_virtual_column_kind::FILENAME});
  definitions.push_back({3,
                         "file_row_number",
                         sirius::logical_type::make(sirius::type_id::BIGINT),
                         scan::scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER});
  auto types = physical_types();
  types.push_back(sirius::logical_type::make(sirius::type_id::VARCHAR));
  types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
  auto const plan =
    scan::build_scan_plan({duckdb::ColumnIndex(3), duckdb::ColumnIndex(0), duckdb::ColumnIndex(2)},
                          {1, 0, 2},
                          {"x", "text", "source_path", "file_row_number"},
                          types,
                          2,
                          {},
                          definitions);

  REQUIRE(plan.data_columns.size() == 1);
  REQUIRE(plan.virtual_columns.size() == 2);
  CHECK(plan.batch_position_by_column_id[0] == 1);
  CHECK(plan.batch_position_by_column_id[1] == 0);
  CHECK(plan.batch_position_by_column_id[2] == 2);
  REQUIRE(plan.output_layout.size() == 2);
  CHECK(plan.output_layout[0].idx == 0);
  CHECK(plan.output_layout[1].idx == 1);
}

TEST_CASE("parquet carrier selection excludes legacy synthetic schema entries",
          "[scan][parquet][virtual_columns][scan_plan][carrier][virtual_review]")
{
  auto definitions = parquet_virtuals();
  definitions.push_back({0,
                         "source_path",
                         sirius::logical_type::make(sirius::type_id::VARCHAR),
                         scan::scan_plan::parquet_virtual_column_kind::FILENAME});
  definitions.push_back({2,
                         "file_row_number",
                         sirius::logical_type::make(sirius::type_id::BIGINT),
                         scan::scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER});
  auto const text   = sirius::logical_type::make(sirius::type_id::VARCHAR);
  auto const bigint = sirius::logical_type::make(sirius::type_id::BIGINT);
  for (auto const requested :
       {duckdb::column_t{2}, duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX}) {
    auto const plan = scan::build_scan_plan({duckdb::ColumnIndex(requested)},
                                            {},
                                            {"source_path", "payload", "file_row_number"},
                                            {text, text, bigint},
                                            1,
                                            {},
                                            definitions);
    REQUIRE(plan.data_columns.size() == 1);
    CHECK(plan.carrier_batch_index == 0);
    CHECK(plan.data_columns[0].primary_idx == 1);
    CHECK(plan.virtual_columns[0].materialized_idx == 1);
  }

  definitions.pop_back();
  auto const plan = scan::build_scan_plan(
    {duckdb::ColumnIndex(duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX)},
    {},
    {"source_path", "payload"},
    {text, text},
    1,
    {},
    definitions);
  REQUIRE(plan.data_columns.size() == 1);
  CHECK(plan.data_columns[0].primary_idx == 1);
}

TEST_CASE("parquet virtual-only scans reject schemas without a usable carrier",
          "[scan][parquet][virtual_columns][scan_plan][carrier][virtual_review]")
{
  CHECK_THROWS_AS(
    scan::build_scan_plan(
      {duckdb::ColumnIndex(duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER)},
      {},
      {"items"},
      {sirius::logical_type::make(sirius::type_id::LIST)},
      1,
      {},
      parquet_virtuals()),
    duckdb::NotImplementedException);
}

TEST_CASE("parquet virtual synthesis rejects an unexpected decoded width",
          "[scan][parquet][virtual_columns][virtual_review]")
{
  auto const plan = scan::build_scan_plan(
    {duckdb::ColumnIndex(duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER)},
    {},
    {"x", "text"},
    physical_types(),
    1,
    {},
    parquet_virtuals());
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}));
  columns.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}));
  CHECK_THROWS_AS(
    scan::append_parquet_virtual_columns(std::make_unique<cudf::table>(std::move(columns)),
                                         plan,
                                         "a.parquet",
                                         0,
                                         0,
                                         cudf::get_default_stream(),
                                         cudf::get_current_device_resource_ref()),
    sirius::internal_exception);
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

TEST_CASE("parquet scan plan classifies execution sentinels before bound virtual metadata",
          "[scan][parquet][virtual_columns][scan_plan][sentinel]")
{
  for (auto const sentinel : {duckdb::COLUMN_IDENTIFIER_EMPTY, duckdb::COLUMN_IDENTIFIER_ROW_ID}) {
    auto const plan = scan::build_scan_plan(
      {duckdb::ColumnIndex(sentinel)}, {}, {"x", "text"}, physical_types(), 1, {}, {});

    CHECK(plan.virtual_columns.empty());
    CHECK(plan.output_layout.empty());
    REQUIRE(plan.batch_position_by_column_id.size() == 1);
    CHECK_FALSE(plan.batch_position_by_column_id[0].has_value());
    REQUIRE(plan.carrier_batch_index.has_value());
    CHECK(plan.data_columns[*plan.carrier_batch_index].primary_idx == 0);
  }
}

TEST_CASE("parquet scan plan does not classify same-named physical columns as virtual",
          "[scan][parquet][virtual_columns][scan_plan][identity]")
{
  auto const plan = scan::build_scan_plan({duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)},
                                          {},
                                          {"filename", "file_index"},
                                          physical_types(),
                                          2,
                                          {},
                                          parquet_virtuals());

  CHECK(plan.virtual_columns.empty());
  REQUIRE(plan.data_columns.size() == 2);
  CHECK(plan.data_columns[0].primary_idx == 0);
  CHECK(plan.data_columns[1].primary_idx == 1);
  REQUIRE(plan.output_layout.size() == 2);
  CHECK(plan.output_layout[0].idx == 0);
  CHECK(plan.output_layout[1].idx == 1);
}

TEST_CASE("parquet scan plan requires bound metadata with the exact virtual type",
          "[scan][parquet][virtual_columns][scan_plan][identity]")
{
  auto const filename = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME;

  SECTION("missing bound metadata")
  {
    CHECK_THROWS_AS(scan::build_scan_plan({duckdb::ColumnIndex(filename)},
                                          {},
                                          {"x"},
                                          {sirius::logical_type::make(sirius::type_id::INTEGER)},
                                          1,
                                          {},
                                          {}),
                    duckdb::NotImplementedException);
  }
  SECTION("wrong bound type")
  {
    std::vector<scan::bound_virtual_column> definitions{
      {filename, "filename", sirius::logical_type::make(sirius::type_id::BIGINT)}};
    CHECK_THROWS_AS(scan::build_scan_plan({duckdb::ColumnIndex(filename)},
                                          {},
                                          {"x"},
                                          {sirius::logical_type::make(sirius::type_id::INTEGER)},
                                          1,
                                          {},
                                          definitions),
                    duckdb::NotImplementedException);
  }
}

TEST_CASE("parquet scan plan preserves duplicate physical outputs without duplicate reads",
          "[scan][parquet][virtual_columns][scan_plan][assembly]")
{
  auto const plan = scan::build_scan_plan(
    {duckdb::ColumnIndex(0)}, {0, 0}, {"x", "text"}, physical_types(), 2, {}, {});

  REQUIRE(plan.data_columns.size() == 1);
  REQUIRE(plan.output_layout.size() == 2);
  CHECK(plan.output_layout[0].idx == 0);
  CHECK(plan.output_layout[1].idx == 0);
}
