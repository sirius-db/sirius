/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "io/parquet_helpers.hpp"
#include "io/templated_ioctx.hpp"
#include "op/scan/parquet_materialize.hpp"
#include "utils/utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <duckdb.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
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

namespace {

/// One file's column chunks staged on the device, which is the shape the bulk
/// materialize route hands the hybrid scan readers.
struct staged_chunks {
  std::vector<rmm::device_buffer> buffers;
  std::vector<cudf::device_span<std::uint8_t const>> spans;
};

staged_chunks stage_column_chunks(cudf::io::datasource& source,
                                  std::span<cudf::io::text::byte_range_info const> ranges,
                                  rmm::cuda_stream_view stream)
{
  staged_chunks staged;
  staged.buffers.reserve(ranges.size());
  for (auto const& range : ranges) {
    auto const host = source.host_read(static_cast<std::size_t>(range.offset()),
                                       static_cast<std::size_t>(range.size()));
    staged.buffers.emplace_back(host->data(), host->size(), stream);
  }
  staged.spans.reserve(staged.buffers.size());
  for (auto const& buffer : staged.buffers) {
    staged.spans.emplace_back(static_cast<std::uint8_t const*>(buffer.data()), buffer.size());
  }
  return staged;
}

/// `l_orderkey < 100000` over lineitem.parquet: 100382 of 600572 rows, spread
/// over all 5 row groups, so an equal count proves row-level filtering.
struct orderkey_filter {
  cudf::numeric_scalar<std::int64_t> limit{100000};
  cudf::ast::literal value{limit};
  cudf::ast::column_name_reference column{"l_orderkey"};
  cudf::ast::operation expression{cudf::ast::ast_operator::LESS, column, value};
};

cudf::io::parquet_reader_options filtered_options(std::vector<std::string> const& paths,
                                                  std::vector<std::string> columns,
                                                  cudf::ast::operation const& filter)
{
  auto options =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{paths}).filter(filter).build();
  options.set_column_names(std::move(columns));
  return options;
}

/// Rows the single-file bulk route produces for @p options.
cudf::size_type bulk_row_count(fs::path const& path,
                               cudf::io::parquet_reader_options const& options,
                               rmm::cuda_stream_view stream)
{
  auto source = cudf::io::datasource::create(path.string());
  auto footer = read_parquet_footer(*source);
  cudf::io::parquet::experimental::hybrid_scan_reader reader{
    cudf::host_span<std::uint8_t const>(footer->data(), footer->size()), options};

  auto const row_groups = reader.all_row_groups(options);
  auto const ranges     = reader.all_column_chunks_byte_ranges(row_groups, options);
  auto const staged     = stage_column_chunks(*source, ranges, stream);

  return reader
    .materialize_all_columns(
      row_groups, staged.spans, options, stream, cudf::get_current_device_resource_ref())
    .tbl->num_rows();
}

/// A backend that prefers bulk IO -- the one thing `prefers_bulk_materialize`
/// asks a source's datasource.  Nothing reads through it.
class bulk_object final : public sirius::io::io_object {
 public:
  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return 0; }

 private:
  std::string _path{"bulk"};
};

class bulk_reactor {
 public:
  struct config {
    [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return 1; }
    [[nodiscard]] std::size_t merge_gap_size() const noexcept { return 0; }
    std::size_t n_max_concurrent_scans{0};
  };

  using io_object_type                  = bulk_object;
  using reactor_config_type             = config;
  static constexpr bool prefers_bulk_io = true;

  [[nodiscard]] config const& get_config() const noexcept { return _config; }
  void enqueue(std::unique_ptr<sirius::io::grouped_io_request>) noexcept {}
  [[nodiscard]] std::size_t queued_bytes() const noexcept { return 0; }
  [[nodiscard]] std::size_t staging_block_size() const noexcept { return 0; }
  std::size_t host_read(bulk_object const&, std::size_t, std::size_t size, std::uint8_t*) const
  {
    return size;
  }
  void start() {}
  void shutdown() {}
  void interrupt() {}
  [[nodiscard]] static std::unique_ptr<bulk_object> create_io_object(std::string)
  {
    return std::make_unique<bulk_object>();
  }
  [[nodiscard]] static bool supports(std::string_view) { return true; }
  [[nodiscard]] static std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<cudf::io::text::byte_range_info const> ranges, std::optional<std::size_t>)
  {
    return {ranges.begin(), ranges.end()};
  }

 private:
  config _config;
};

class bulk_context final : public sirius::io::templated_ioctx<bulk_reactor> {
 public:
  using templated_ioctx::templated_ioctx;

  [[nodiscard]] sirius::io::io_context_type type() const noexcept override
  {
    return sirius::io::io_context_type::restful;
  }
};

}  // namespace

TEST_CASE("filtered options still take the bulk materialize route", "[scan][parquet][bulk_filter]")
{
  auto const path = parquet_fixture("lineitem.parquet");
  orderkey_filter const filter;
  auto const options =
    filtered_options({path.string()}, {"l_orderkey", "l_quantity"}, filter.expression);

  auto source = cudf::io::datasource::create(path.string());
  auto footer = read_parquet_footer(*source);
  cudf::io::parquet::experimental::hybrid_scan_reader reader{
    cudf::host_span<std::uint8_t const>(footer->data(), footer->size()), options};

  auto ioctx = std::make_shared<bulk_context>(std::vector<std::unique_ptr<bulk_reactor>>{});
  std::vector<sirius::op::scan::parquet_source> sources;
  sources.push_back(sirius::op::scan::parquet_source{
    std::make_shared<sirius::io::sirius_datasource>(ioctx, std::make_shared<bulk_object>()),
    std::make_shared<cudf::io::parquet::FileMetaData const>(reader.parquet_metadata()),
    reader.all_row_groups(options)});

  CHECK(sirius::op::scan::prefers_bulk_materialize(sources, options));
}

TEST_CASE("hybrid scan bulk materialize applies the reader filter like read_parquet",
          "[scan][parquet][bulk_filter]")
{
  auto const path   = parquet_fixture("lineitem.parquet");
  auto const stream = cudf::get_default_stream();
  orderkey_filter const filter;

  SECTION("filter column inside the projection")
  {
    auto const options =
      filtered_options({path.string()}, {"l_orderkey", "l_quantity"}, filter.expression);
    CHECK(cudf::io::read_parquet(options).tbl->num_rows() == 100382);
    CHECK(bulk_row_count(path, options, stream) == 100382);
  }

  SECTION("filter column outside the projection")
  {
    auto const options =
      filtered_options({path.string()}, {"l_quantity", "l_shipdate"}, filter.expression);
    CHECK(cudf::io::read_parquet(options).tbl->num_rows() == 100382);
    CHECK(bulk_row_count(path, options, stream) == 100382);
  }

  SECTION("several sources through hybrid_scan_multifile")
  {
    auto const options = filtered_options(
      {path.string(), path.string()}, {"l_orderkey", "l_quantity"}, filter.expression);

    auto source = cudf::io::datasource::create(path.string());
    auto footer = read_parquet_footer(*source);
    std::vector<cudf::host_span<std::uint8_t const>> footers(
      2, cudf::host_span<std::uint8_t const>(footer->data(), footer->size()));
    cudf::io::parquet::experimental::hybrid_scan_multifile reader{footers, options};

    auto const row_groups     = reader.all_row_groups(options);
    auto const [ranges, srcs] = reader.all_column_chunks_byte_ranges(row_groups, options);
    // Both entries are the same file, so every range reads from the one source.
    auto const staged = stage_column_chunks(*source, ranges, stream);

    CHECK(cudf::io::read_parquet(options).tbl->num_rows() == 200764);
    CHECK(reader
            .materialize_all_columns(
              row_groups, staged.spans, options, stream, cudf::get_current_device_resource_ref())
            .tbl->num_rows() == 200764);
  }
}
