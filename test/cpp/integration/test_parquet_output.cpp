// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "output/parquet_writer.hpp"
#include "utils/gpu_execution_fixture.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <unistd.h>

#include <filesystem>
#include <fstream>

namespace {
struct export_directory {
  std::filesystem::path path;
  export_directory()
  {
    std::string pattern =
      (std::filesystem::temp_directory_path() / "sirius-copy-sql-XXXXXX").string();
    if (!::mkdtemp(pattern.data())) throw std::runtime_error("mkdtemp failed");
    path = pattern;
  }
  ~export_directory() { std::filesystem::remove_all(path); }
};
std::string quote(std::filesystem::path const& path) { return "'" + path.string() + "'"; }
}  // namespace

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU COPY exports ordered batch-sized Parquet parts",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  std::filesystem::create_directory(root.path / "ordered");
  run_ok("SET gpu_execution=false");
  run_ok(
    "CREATE TABLE export_input AS SELECT i::INTEGER k, (i % 7)::INTEGER v FROM range(10000) t(i)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=false");
  auto result = con->Query("COPY (SELECT * FROM export_input ORDER BY k DESC) TO " +
                           quote(root.path / "ordered") +
                           " (FORMAT PARQUET, PER_THREAD_OUTPUT true, COMPRESSION 'snappy')");
  if (result->HasError()) UNSCOPED_INFO(result->GetError());
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 10000);
  run_ok("SET gpu_execution=false");
  std::vector<std::filesystem::path> parts;
  for (auto const& entry : std::filesystem::directory_iterator(root.path / "ordered"))
    parts.push_back(entry.path());
  std::sort(parts.begin(), parts.end());
  int64_t expected = 9999;
  for (auto const& file : parts) {
    auto rows = con->Query("SELECT k FROM read_parquet(" + quote(file) + ")");
    REQUIRE_FALSE(rows->HasError());
    auto metadata = con->Query("SELECT created_by FROM parquet_file_metadata(" + quote(file) + ")");
    REQUIRE_FALSE(metadata->HasError());
    REQUIRE(metadata->Collection().GetRows().GetValue(0, 0).ToString().find("cudf") !=
            std::string::npos);
    for (auto const& row : sirius::test::collect_rows(*rows, false))
      REQUIRE(std::stoll(row[0]) == expected--);
  }
  REQUIRE(expected == -1);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "COPY unsupported options use the existing fallback policy",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_input AS SELECT 42::INTEGER k");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=false");
  auto sql = "COPY (SELECT * FROM export_input) TO " + quote(root.path / "unsupported") +
             " (FORMAT PARQUET, PER_THREAD_OUTPUT true, ROW_GROUP_SIZE 2048)";
  auto refused = con->Query(sql);
  REQUIRE(refused->HasError());
  REQUIRE_FALSE(std::filesystem::exists(root.path / "unsupported"));
  run_ok("SET enable_duckdb_fallback=true");
  run_ok(sql);
  REQUIRE(std::filesystem::is_directory(root.path / "unsupported"));
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU COPY creates a schema-only part for no rows",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_empty(k INTEGER, d DECIMAL(9,2))");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=false");
  run_ok("COPY (SELECT * FROM export_empty) TO " + quote(root.path / "empty") +
         " (FORMAT PARQUET, PER_THREAD_OUTPUT true)");
  run_ok("SET gpu_execution=false");
  auto result =
    con->Query("SELECT count(*) FROM read_parquet(" + quote(root.path / "empty/*.parquet") + ")");
  if (result->HasError()) UNSCOPED_INFO(result->GetError());
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 0);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU COPY round trips nullable scalars and decimal widths",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok(
    "CREATE TABLE export_types AS SELECT i::INTEGER k, "
    "CASE WHEN i % 3 = 0 THEN NULL ELSE (i / 10)::DECIMAL(9,2) END small_decimal, "
    "(i / 100)::DECIMAL(18,3) medium_decimal, "
    "CASE WHEN i % 5 = 0 THEN NULL ELSE 'value-' || i END text_value "
    "FROM range(100) t(i)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=false");
  run_ok(
    "COPY (SELECT k, small_decimal::DECIMAL(9,2) small_decimal, medium_decimal, text_value FROM "
    "export_types) TO " +
    quote(root.path / "types") + " (FORMAT PARQUET, PER_THREAD_OUTPUT true, COMPRESSION 'zstd')");
  run_ok("SET gpu_execution=false");
  auto result = con->Query(
    "SELECT count(*) FROM ((SELECT k, small_decimal::DECIMAL(9,2), medium_decimal, text_value FROM "
    "export_types EXCEPT ALL SELECT * FROM read_parquet(" +
    quote(root.path / "types/*.parquet") + ")) UNION ALL (SELECT * FROM read_parquet(" +
    quote(root.path / "types/*.parquet") +
    ") EXCEPT ALL SELECT k, small_decimal::DECIMAL(9,2), medium_decimal, text_value FROM "
    "export_types)) differences");
  if (result->HasError()) UNSCOPED_INFO(result->GetError());
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 0);
  auto schema =
    con->Query("DESCRIBE SELECT * FROM read_parquet(" + quote(root.path / "types/*.parquet") + ")");
  REQUIRE_FALSE(schema->HasError());
  REQUIRE(schema->Collection().GetRows().GetValue(1, 1).ToString() == "DECIMAL(9,2)");
  REQUIRE(schema->Collection().GetRows().GetValue(1, 2).ToString() == "DECIMAL(18,3)");
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU COPY honors the installed cuDF codec capabilities",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_codecs AS SELECT i::INTEGER k FROM range(100) t(i)");
  run_ok("CHECKPOINT");
  run_ok("SET enable_duckdb_fallback=false");
  for (auto const* codec : {"uncompressed", "snappy", "gzip", "brotli", "zstd", "lz4_raw"}) {
    INFO("codec=" << codec);
    run_ok("SET gpu_execution=true");
    auto directory = root.path / codec;
    auto result =
      con->Query("COPY (SELECT * FROM export_codecs) TO " + quote(directory) +
                 " (FORMAT PARQUET, PER_THREAD_OUTPUT true, COMPRESSION '" + codec + "')");
    if (!sirius::output::supported_parquet_compression(sirius::output::parse_compression(codec))) {
      REQUIRE(result->HasError());
      REQUIRE_FALSE(std::filesystem::exists(directory));
      continue;
    }
    if (result->HasError()) UNSCOPED_INFO(result->GetError());
    REQUIRE_FALSE(result->HasError());
    REQUIRE(result->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 100);
    run_ok("SET gpu_execution=false");
    auto check = con->Query("SELECT count(*), sum(k) FROM read_parquet(" +
                            quote(directory / "*.parquet") + ")");
    REQUIRE_FALSE(check->HasError());
    REQUIRE(check->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 100);
    REQUIRE(check->Collection().GetRows().GetValue(1, 0).ToString() == "4950");
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU schema-only writer preserves nested names and decimal precision",
                 "[parquet_export_gpu][integration]")
{
  using namespace sirius;
  export_directory root;
  output::parquet_export_spec spec;
  output::parquet_field decimal;
  decimal.name              = "small_decimal";
  decimal.type              = logical_type::make_decimal(4, 2);
  decimal.decimal_precision = 4;
  output::parquet_field nested;
  nested.name = "nested";
  nested.type = logical_type::make(type_id::STRUCT);
  nested.children.push_back(decimal);
  output::parquet_field list;
  list.name    = "values";
  list.type    = logical_type::make(type_id::LIST);
  decimal.name = "element";
  list.children.push_back(decimal);
  spec.fields = {nested, list};
  output::write_empty_parquet(spec, root.path / "nested.parquet");
  run_ok("SET gpu_execution=false");
  auto schema =
    con->Query("DESCRIBE SELECT * FROM read_parquet(" + quote(root.path / "nested.parquet") + ")");
  if (schema->HasError()) UNSCOPED_INFO(schema->GetError());
  REQUIRE_FALSE(schema->HasError());
  REQUIRE(schema->Collection().GetRows().GetValue(1, 0).ToString() ==
          "STRUCT(small_decimal DECIMAL(4,2))");
  REQUIRE(schema->Collection().GetRows().GetValue(1, 1).ToString() == "DECIMAL(4,2)[]");
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU writer widens the small-decimal carrier without rescaling",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  sirius::output::parquet_export_spec spec;
  sirius::output::parquet_field decimal;
  decimal.name              = "d";
  decimal.type              = sirius::logical_type::make_decimal(4, 2);
  decimal.decimal_precision = 4;
  spec.fields.push_back(decimal);
  auto column = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT16}, 3, cudf::mask_state::UNALLOCATED);
  int16_t const values[] = {123, -456, 0};
  REQUIRE(cudaMemcpy(column->mutable_view().data<int16_t>(),
                     values,
                     sizeof(values),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  sirius::output::write_parquet_part(spec,
                                     cudf::table_view({column->view()}),
                                     root.path / "decimal.parquet",
                                     cudf::get_default_stream());
  run_ok("SET gpu_execution=false");
  auto rows =
    con->Query("SELECT * FROM read_parquet(" + quote(root.path / "decimal.parquet") + ")");
  REQUIRE_FALSE(rows->HasError());
  auto actual = sirius::test::collect_rows(*rows, false);
  REQUIRE(actual == std::vector<std::vector<std::string>>{{"1.23"}, {"-4.56"}, {"0.00"}});
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU COPY rejects nonempty destinations without CPU replay",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  auto target = root.path / "existing";
  std::filesystem::create_directories(target / "nested");
  auto original = target / "nested" / "original.txt";
  std::ofstream(original) << "preserve";
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_input AS SELECT 42::INTEGER k");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=true");
  auto result = con->Query("COPY (SELECT * FROM export_input) TO " + quote(target) +
                           " (FORMAT PARQUET, PER_THREAD_OUTPUT true)");
  REQUIRE(result->HasError());
  REQUIRE(result->GetError().find("CPU replay disabled") != std::string::npos);
  REQUIRE(std::filesystem::file_size(original) == 8);
  REQUIRE(std::distance(std::filesystem::directory_iterator(target),
                        std::filesystem::directory_iterator{}) == 1);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU COPY overwrite and append options remain unsupported",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_input AS SELECT 42::INTEGER k");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=false");
  for (auto const* option : {"OVERWRITE", "OVERWRITE_OR_IGNORE", "APPEND"}) {
    INFO(option);
    auto target = root.path / option;
    std::filesystem::create_directory(target);
    auto original = target / "original.txt";
    std::ofstream(original) << "preserve";
    auto result = con->Query("COPY (SELECT * FROM export_input) TO " + quote(target) +
                             " (FORMAT PARQUET, PER_THREAD_OUTPUT true, " + option + " true)");
    REQUIRE(result->HasError());
    REQUIRE(std::filesystem::file_size(original) == 8);
    REQUIRE(std::distance(std::filesystem::directory_iterator(target),
                          std::filesystem::directory_iterator{}) == 1);
  }
}
