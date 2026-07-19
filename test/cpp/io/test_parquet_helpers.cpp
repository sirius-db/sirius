/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "io/parquet_helpers.hpp"
#include "utils/utils.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/span.hpp>

#include <duckdb.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

void require_ok(std::unique_ptr<duckdb::QueryResult> result)
{
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
}

fs::path fresh_tmp_dir(std::string const& tag)
{
  auto dir = fs::temp_directory_path() / ("sirius_pr6_schema_" + tag);
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir);
  return dir;
}

fs::path parquet_fixture(std::string_view file_name)
{
  return fs::path{SIRIUS_PROJECT_ROOT} / "test" / "cpp" / "integration" / "data" / "parquet" /
         file_name;
}

fs::path write_parquet(duckdb::Connection& con,
                       fs::path const& dir,
                       std::string const& table,
                       std::string const& create_sql)
{
  require_ok(con.Query("DROP TABLE IF EXISTS " + table));
  require_ok(con.Query(create_sql));

  auto path = dir / (table + ".parquet");
  require_ok(con.Query("COPY " + table + " TO '" + path.string() + "' (FORMAT PARQUET)"));
  return path;
}

std::unique_ptr<cudf::io::datasource::buffer> read_parquet_footer(cudf::io::datasource& source)
{
  auto constexpr footer_tail_size = sizeof(cudf::io::parquet::file_ender_s);
  auto const file_size            = source.size();
  REQUIRE(file_size >= footer_tail_size);

  auto tail = source.host_read(file_size - footer_tail_size, footer_tail_size);

  std::uint32_t footer_size = 0;
  std::memcpy(&footer_size, tail->data(), sizeof(footer_size));
  REQUIRE(file_size >= footer_tail_size + footer_size);

  return source.host_read(file_size - footer_tail_size - footer_size, footer_size);
}

cudf::io::parquet::FileMetaData read_metadata(fs::path const& path)
{
  auto source = cudf::io::datasource::create(path.string());
  auto footer = read_parquet_footer(*source);
  auto opts   = cudf::io::parquet_reader_options::builder().build();
  cudf::io::parquet::experimental::hybrid_scan_reader reader{
    cudf::host_span<std::uint8_t const>(footer->data(), footer->size()), opts};
  return reader.parquet_metadata();
}

std::vector<duckdb::LogicalType> expected_flat_types()
{
  return {duckdb::LogicalType::INTEGER,
          duckdb::LogicalType::BIGINT,
          duckdb::LogicalType::DOUBLE,
          duckdb::LogicalType::BOOLEAN,
          duckdb::LogicalType::VARCHAR};
}

void require_struct_child(duckdb::LogicalType const& type,
                          duckdb::idx_t index,
                          std::string const& name,
                          duckdb::LogicalType const& child_type)
{
  REQUIRE(type.id() == duckdb::LogicalTypeId::STRUCT);
  REQUIRE(duckdb::StructType::GetChildCount(type) > index);
  CHECK(duckdb::StructType::GetChildName(type, index) == name);
  CHECK(duckdb::StructType::GetChildType(type, index) == child_type);
}

}  // namespace

TEST_CASE("parquet_helpers extract_schema maps flat parquet leaves", "[parquet_helpers][schema]")
{
  auto const dir       = fresh_tmp_dir("flat");
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  auto const path      = write_parquet(con,
                                  dir,
                                  "flat_types",
                                  "CREATE TABLE flat_types AS SELECT "
                                       "42::INTEGER AS i32_col, "
                                       "9000000000::BIGINT AS i64_col, "
                                       "1.25::DOUBLE AS double_col, "
                                       "true::BOOLEAN AS bool_col, "
                                       "'hello'::VARCHAR AS utf8_col");

  auto meta = read_metadata(path);
  auto info = sirius::io::parquet_helpers::extract_schema(meta);

  CHECK(info.names ==
        std::vector<std::string>{"i32_col", "i64_col", "double_col", "bool_col", "utf8_col"});
  REQUIRE(info.types.size() == expected_flat_types().size());
  auto expected = expected_flat_types();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(info.types[i] == expected[i]);
  }
}

TEST_CASE("parquet_helpers extract_schema maps decimal date and timestamp annotations",
          "[parquet_helpers][schema]")
{
  auto const dir       = fresh_tmp_dir("annotations");
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  auto const path      = write_parquet(con,
                                  dir,
                                  "annotated_types",
                                  "CREATE TABLE annotated_types AS SELECT "
                                       "12.34::DECIMAL(12,2) AS amount, "
                                       "DATE '2024-01-02' AS day_value, "
                                       "TIMESTAMP '2024-01-02 03:04:05' AS ts_value");

  auto meta = read_metadata(path);
  auto info = sirius::io::parquet_helpers::extract_schema(meta);

  CHECK(info.names == std::vector<std::string>{"amount", "day_value", "ts_value"});
  REQUIRE(info.types.size() == 3);
  CHECK(info.types[0] == duckdb::LogicalType::DECIMAL(12, 2));
  CHECK(info.types[1] == duckdb::LogicalType::DATE);
  CHECK(info.types[2] == duckdb::LogicalType::TIMESTAMP);
}

TEST_CASE("parquet_helpers extract_schema maps top-level struct columns",
          "[parquet_helpers][schema][nested]")
{
  auto meta = read_metadata(parquet_fixture("nested_struct.parquet"));
  auto info = sirius::io::parquet_helpers::extract_schema(meta);

  REQUIRE(info.names == std::vector<std::string>{"id", "payload"});
  REQUIRE(info.types.size() == 2);
  CHECK(info.types[0] == duckdb::LogicalType::INTEGER);
  REQUIRE(info.types[1].id() == duckdb::LogicalTypeId::STRUCT);
  REQUIRE(duckdb::StructType::GetChildCount(info.types[1]) == 2);
  require_struct_child(info.types[1], 0, "a", duckdb::LogicalType::INTEGER);
  require_struct_child(info.types[1], 1, "b", duckdb::LogicalType::VARCHAR);
}

TEST_CASE("parquet_helpers extract_schema maps top-level list columns",
          "[parquet_helpers][schema][nested]")
{
  auto meta = read_metadata(parquet_fixture("nested_list.parquet"));
  auto info = sirius::io::parquet_helpers::extract_schema(meta);

  REQUIRE(info.names == std::vector<std::string>{"id", "items"});
  REQUIRE(info.types.size() == 2);
  CHECK(info.types[0] == duckdb::LogicalType::INTEGER);
  REQUIRE(info.types[1].id() == duckdb::LogicalTypeId::LIST);
  CHECK(duckdb::ListType::GetChildType(info.types[1]) == duckdb::LogicalType::BIGINT);
}

TEST_CASE("parquet_helpers extract_schema maps parquet map columns to DuckDB MAP",
          "[parquet_helpers][schema][nested]")
{
  auto meta = read_metadata(parquet_fixture("nested_map.parquet"));
  auto info = sirius::io::parquet_helpers::extract_schema(meta);

  REQUIRE(info.names == std::vector<std::string>{"id", "attrs"});
  REQUIRE(info.types.size() == 2);
  CHECK(info.types[0] == duckdb::LogicalType::INTEGER);
  REQUIRE(info.types[1].id() == duckdb::LogicalTypeId::MAP);
  CHECK(duckdb::MapType::KeyType(info.types[1]) == duckdb::LogicalType::VARCHAR);
  CHECK(duckdb::MapType::ValueType(info.types[1]) == duckdb::LogicalType::INTEGER);
}

TEST_CASE("parquet_helpers extract_schema maps deep nested columns and resumes at next scalar",
          "[parquet_helpers][schema][nested]")
{
  auto meta = read_metadata(parquet_fixture("nested_deep.parquet"));
  auto info = sirius::io::parquet_helpers::extract_schema(meta);

  REQUIRE(info.names == std::vector<std::string>{"id", "struct_of_list", "list_of_struct", "tail"});
  REQUIRE(info.types.size() == 4);
  CHECK(info.types[0] == duckdb::LogicalType::INTEGER);
  CHECK(info.types[3] == duckdb::LogicalType::INTEGER);

  REQUIRE(info.types[1].id() == duckdb::LogicalTypeId::STRUCT);
  REQUIRE(duckdb::StructType::GetChildCount(info.types[1]) == 1);
  CHECK(duckdb::StructType::GetChildName(info.types[1], 0) == "s");
  auto const& struct_list_child = duckdb::StructType::GetChildType(info.types[1], 0);
  REQUIRE(struct_list_child.id() == duckdb::LogicalTypeId::LIST);
  CHECK(duckdb::ListType::GetChildType(struct_list_child) == duckdb::LogicalType::INTEGER);

  REQUIRE(info.types[2].id() == duckdb::LogicalTypeId::LIST);
  auto const& list_struct_child = duckdb::ListType::GetChildType(info.types[2]);
  REQUIRE(list_struct_child.id() == duckdb::LogicalTypeId::STRUCT);
  REQUIRE(duckdb::StructType::GetChildCount(list_struct_child) == 1);
  require_struct_child(list_struct_child, 0, "x", duckdb::LogicalType::DOUBLE);
}
