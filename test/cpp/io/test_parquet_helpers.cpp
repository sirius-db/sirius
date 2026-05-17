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

TEST_CASE("parquet_helpers extract_schema rejects nested top-level columns cleanly",
          "[parquet_helpers][schema]")
{
  auto const dir       = fresh_tmp_dir("nested");
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  auto const path      = write_parquet(con,
                                  dir,
                                  "nested_types",
                                  "CREATE TABLE nested_types AS SELECT "
                                       "struct_pack(child := 7) AS payload");

  auto meta = read_metadata(path);
  REQUIRE_THROWS_WITH(sirius::io::parquet_helpers::extract_schema(meta),
                      Catch::Contains("nested parquet types not yet supported"));
}
