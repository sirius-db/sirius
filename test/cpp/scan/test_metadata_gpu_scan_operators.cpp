/*
 * Copyright 2025, Sirius Contributors.
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

// test
#include <catch.hpp>
#include <scan/test_utils.hpp>
#include <utils/utils.hpp>

// sirius
#include <data/data_batch_utils.hpp>
#include <helper/type_conversions.hpp>
#include <op/scan/parquet_scan_operator_data.hpp>
#include <op/scan/sirius_gpu_parquet_scan_operator.hpp>
#include <op/scan/sirius_parquet_metadata_scan_operator.hpp>

// cucascade
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

// cudf
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

// duckdb
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>

// standard library
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace sirius::scan_test_utils;
using namespace cucascade::memory;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Write a DuckDB table to a parquet file.  Returns the path.
std::filesystem::path write_parquet(duckdb::Connection& con,
                                    std::string const& table_name,
                                    std::size_t row_group_size = 0)
{
  auto path = std::filesystem::temp_directory_path() / (table_name + ".parquet");
  std::string sql;
  if (row_group_size != 0) {
    sql = "COPY " + table_name + " TO '" + path.string() +
          "' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE " + std::to_string(row_group_size) +
          ")";
  } else {
    sql = "COPY " + table_name + " TO '" + path.string() + "' (FORMAT PARQUET, COMPRESSION zstd)";
  }
  auto result = con.Query(sql);
  REQUIRE(result);
  REQUIRE(!result->HasError());
  return path;
}

/// RAII cleanup for parquet files.
struct parquet_file_cleanup {
  std::vector<std::filesystem::path> paths;
  ~parquet_file_cleanup()
  {
    for (auto const& p : paths) {
      std::filesystem::remove(p);
    }
  }
};

/// Create a table with diverse column types:
///   id INTEGER, value BIGINT, price DECIMAL(12,2), label VARCHAR, created DATE
void create_diverse_table(duckdb::Connection& con,
                          std::string const& table_name,
                          std::size_t num_rows)
{
  // clang-format off
  auto result = con.Query(
    "CREATE TABLE " + table_name + " ("
    "id INTEGER, "
    "value BIGINT, "
    "price DECIMAL(12,2), "
    "label VARCHAR, "
    "created DATE"
    ");");
  // clang-format on
  REQUIRE(result);
  REQUIRE(!result->HasError());

  constexpr std::size_t BATCH = 1000;
  for (std::size_t start = 0; start < num_rows; start += BATCH) {
    auto end           = std::min(start + BATCH, num_rows);
    std::string insert = "INSERT INTO " + table_name + " VALUES ";
    for (std::size_t i = start; i < end; ++i) {
      if (i > start) { insert += ", "; }
      auto id = static_cast<int32_t>(i);
      // price: 0.01, 1.01, 2.01, ...
      auto price_cents = static_cast<int64_t>(i * 100 + 1);
      // date: 2020-01-01 + i days
      insert += "(" + std::to_string(id) + ", " + std::to_string(static_cast<int64_t>(i) * 100) +
                ", " + std::to_string(price_cents / 100) + "." +
                (price_cents % 100 < 10 ? "0" : "") + std::to_string(price_cents % 100) + ", " +
                "'label_" + std::to_string(i) + "', " + "'2020-01-01'::DATE + INTERVAL '" +
                std::to_string(i) + " days')";
    }
    result = con.Query(insert);
    REQUIRE(result);
    REQUIRE(!result->HasError());
  }
}

/// Schema info for the synthetic table (id INTEGER, value BIGINT, price DOUBLE, name VARCHAR).
struct schema_info {
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<std::string> names;
  duckdb::vector<duckdb::LogicalType> types;
};

schema_info synthetic_table_schema()
{
  schema_info info;
  info.column_ids = {
    duckdb::ColumnIndex(0), duckdb::ColumnIndex(1), duckdb::ColumnIndex(2), duckdb::ColumnIndex(3)};
  info.names = {"id", "value", "price", "name"};
  info.types = {duckdb::LogicalType::INTEGER,
                duckdb::LogicalType::BIGINT,
                duckdb::LogicalType::DOUBLE,
                duckdb::LogicalType::VARCHAR};
  return info;
}

/// Schema info for the diverse table
/// (id INTEGER, value BIGINT, price DECIMAL(12,2), label VARCHAR, created DATE).
schema_info diverse_table_schema()
{
  schema_info info;
  info.column_ids = {duckdb::ColumnIndex(0),
                     duckdb::ColumnIndex(1),
                     duckdb::ColumnIndex(2),
                     duckdb::ColumnIndex(3),
                     duckdb::ColumnIndex(4)};
  info.names      = {"id", "value", "price", "label", "created"};
  info.types      = {duckdb::LogicalType::INTEGER,
                     duckdb::LogicalType::BIGINT,
                     duckdb::LogicalType::DECIMAL(12, 2),
                     duckdb::LogicalType::VARCHAR,
                     duckdb::LogicalType::DATE};
  return info;
}

/// Run the full two-pipeline scan: metadata scan → sink → finalize → GPU scan.
/// Returns all output data_batches.
std::vector<std::shared_ptr<cucascade::data_batch>> run_two_pipeline_scan(
  std::vector<std::string> const& file_paths,
  const duckdb::vector<duckdb::LogicalType>& output_types,
  const duckdb::vector<duckdb::LogicalType>& returned_types,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<duckdb::idx_t>& projection_ids,
  const duckdb::vector<std::string>& names,
  std::size_t approximate_batch_size,
  cucascade::memory::memory_space& gpu_space,
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters = nullptr,
  rmm::cuda_stream_view stream                             = cudf::get_default_stream())
{
  sirius::op::scan::sirius_gpu_parquet_scan_operator gpu_op(sirius::from_duckdb_vec(output_types),
                                                            0);

  // --- Pipeline 1: metadata scan ---
  sirius::op::scan::sirius_parquet_metadata_scan_operator metadata_op(
    &gpu_op,
    sirius::from_duckdb_vec(output_types),
    sirius::from_duckdb_vec(returned_types),
    0,
    file_paths,
    column_ids,
    projection_ids,
    names,
    std::move(table_filters),
    approximate_batch_size);

  // Execute all metadata tasks and sink results into the GPU operator via metadata_op.sink().
  while (!metadata_op.all_ports_empty()) {
    auto input = metadata_op.get_next_task_input_data();
    if (!input) { break; }
    auto output = metadata_op.execute(*input, stream);
    REQUIRE(output);
    metadata_op.sink(*output, stream);
  }

  // --- Pipeline 1 → Pipeline 2 transition ---
  metadata_op.finalize_operator();

  // --- Pipeline 2: GPU scan ---
  std::vector<std::shared_ptr<cucascade::data_batch>> all_batches;
  while (!gpu_op.all_ports_empty()) {
    auto hint = gpu_op.get_next_task_hint();
    if (!hint) { break; }
    auto input = gpu_op.get_next_task_input_data();
    if (!input) { break; }
    input->prepare_for_processing(&gpu_space, stream);
    auto output = gpu_op.execute(*input, stream);
    REQUIRE(output);
    auto* pipelineable = dynamic_cast<sirius::op::pipelineable_operator_data*>(output.get());
    REQUIRE(pipelineable);
    for (auto& batch : pipelineable->get_data_batches()) {
      all_batches.push_back(batch);
    }
  }

  return all_batches;
}

/// Copy a column of int32 values from GPU to host.
std::vector<int32_t> copy_int32_column(cudf::column_view const& col)
{
  std::vector<int32_t> host(col.size());
  cudaMemcpy(
    host.data(), col.data<int32_t>(), sizeof(int32_t) * col.size(), cudaMemcpyDeviceToHost);
  return host;
}

/// Copy a column of int64 values from GPU to host.
std::vector<int64_t> copy_int64_column(cudf::column_view const& col)
{
  std::vector<int64_t> host(col.size());
  cudaMemcpy(
    host.data(), col.data<int64_t>(), sizeof(int64_t) * col.size(), cudaMemcpyDeviceToHost);
  return host;
}

/// Copy a string column from GPU to host.
std::vector<std::string> copy_string_column(cudf::column_view const& col)
{
  cudf::strings_column_view str_col(col);
  auto offsets = copy_string_offsets(str_col.offsets());
  std::vector<char> chars;
  if (!offsets.empty() && offsets.back() > 0) {
    chars.resize(static_cast<std::size_t>(offsets.back()));
    cudaMemcpy(chars.data(),
               str_col.chars_begin(cudf::get_default_stream()),
               chars.size(),
               cudaMemcpyDeviceToHost);
  }
  std::vector<std::string> result;
  result.reserve(col.size());
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    auto start = static_cast<std::size_t>(offsets[i]);
    auto end   = static_cast<std::size_t>(offsets[i + 1]);
    result.emplace_back(chars.data() + start, chars.data() + end);
  }
  return result;
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST_CASE("metadata_scan_operator - source interface dispatches all files",
          "[metadata_scan][shared_context]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  create_synthetic_table(con, "src_test", 1000);
  auto path = write_parquet(con, "src_test", 200);
  parquet_file_cleanup cleanup{{path}};

  auto schema                    = synthetic_table_schema();
  std::vector<std::string> files = {path.string()};
  duckdb::vector<duckdb::idx_t> no_projection;

  sirius::op::scan::sirius_gpu_parquet_scan_operator gpu_op(sirius::from_duckdb_vec(schema.types),
                                                            0);
  sirius::op::scan::sirius_parquet_metadata_scan_operator op(&gpu_op,
                                                             sirius::from_duckdb_vec(schema.types),
                                                             sirius::from_duckdb_vec(schema.types),
                                                             0,
                                                             files,
                                                             schema.column_ids,
                                                             no_projection,
                                                             schema.names);

  REQUIRE(op.is_source());
  REQUIRE_FALSE(op.all_ports_empty());

  std::size_t task_count = 0;
  while (!op.all_ports_empty()) {
    auto hint = op.get_next_task_hint();
    if (!hint) { break; }
    auto input = op.get_next_task_input_data();
    if (!input) { break; }
    ++task_count;
  }

  REQUIRE(task_count > 0);
  REQUIRE(op.all_ports_empty());
  REQUIRE(op.get_next_task_input_data() == nullptr);
}

TEST_CASE("metadata_scan_operator - execute produces partitioned metadata",
          "[metadata_scan][shared_context]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  create_synthetic_table(con, "meta_exec_test", 2000);
  auto path = write_parquet(con, "meta_exec_test", 500);
  parquet_file_cleanup cleanup{{path}};

  auto schema                    = synthetic_table_schema();
  std::vector<std::string> files = {path.string()};
  duckdb::vector<duckdb::idx_t> no_projection;

  sirius::op::scan::sirius_gpu_parquet_scan_operator gpu_op(sirius::from_duckdb_vec(schema.types),
                                                            0);
  sirius::op::scan::sirius_parquet_metadata_scan_operator op(&gpu_op,
                                                             sirius::from_duckdb_vec(schema.types),
                                                             sirius::from_duckdb_vec(schema.types),
                                                             0,
                                                             files,
                                                             schema.column_ids,
                                                             no_projection,
                                                             schema.names);

  auto input = op.get_next_task_input_data();
  REQUIRE(input);

  auto output = op.execute(*input, cudf::get_default_stream());
  REQUIRE(output);

  auto* meta = dynamic_cast<sirius::op::scan::partitioned_parquet_metadata*>(output.get());
  REQUIRE(meta);
  REQUIRE(meta->file_paths.size() == 1);
  REQUIRE_FALSE(meta->row_group_partitions.empty());
  REQUIRE(meta->reader_options != nullptr);
}

TEST_CASE("two-pipeline scan - basic scan with all columns", "[two_pipeline_scan][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 2000;
  create_synthetic_table(con, "basic_scan", NUM_ROWS);
  auto path = write_parquet(con, "basic_scan", 500);
  parquet_file_cleanup cleanup{{path}};

  auto schema                    = synthetic_table_schema();
  std::vector<std::string> files = {path.string()};
  duckdb::vector<duckdb::idx_t> no_projection;

  auto batches = run_two_pipeline_scan(files,
                                       schema.types,
                                       schema.types,
                                       schema.column_ids,
                                       no_projection,
                                       schema.names,
                                       1024 * 1024,
                                       *gpu_space);

  REQUIRE_FALSE(batches.empty());

  // Verify total row count
  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    REQUIRE(batch);
    auto table = sirius::get_cudf_table_view(*batch);
    REQUIRE(table.num_columns() == 4);
    total_rows += table.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);
}

TEST_CASE("two-pipeline scan - projection selects subset of columns",
          "[two_pipeline_scan][projection][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 1000;
  create_synthetic_table(con, "proj_scan", NUM_ROWS);
  auto path = write_parquet(con, "proj_scan", 500);
  parquet_file_cleanup cleanup{{path}};

  auto schema                    = synthetic_table_schema();
  std::vector<std::string> files = {path.string()};

  // Project: id (col 0) and price (col 2)
  duckdb::vector<duckdb::idx_t> projection_ids{0, 2};

  // Output types for projected columns
  duckdb::vector<duckdb::LogicalType> output_types;
  output_types.push_back(schema.types[0]);
  output_types.push_back(schema.types[2]);

  auto batches = run_two_pipeline_scan(files,
                                       output_types,
                                       schema.types,
                                       schema.column_ids,
                                       projection_ids,
                                       schema.names,
                                       1024 * 1024,
                                       *gpu_space);

  REQUIRE_FALSE(batches.empty());
  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto table = sirius::get_cudf_table_view(*batch);
    REQUIRE(table.num_columns() == 2);
    // id column should be INT32
    REQUIRE(table.column(0).type().id() == cudf::type_id::INT32);
    // price column should be FLOAT64
    REQUIRE(table.column(1).type().id() == cudf::type_id::FLOAT64);
    total_rows += table.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);
}

TEST_CASE("two-pipeline scan - diverse types (VARCHAR, DECIMAL, DATE)",
          "[two_pipeline_scan][types][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 500;
  create_diverse_table(con, "diverse_scan", NUM_ROWS);
  auto path = write_parquet(con, "diverse_scan", 200);
  parquet_file_cleanup cleanup{{path}};

  auto schema                    = diverse_table_schema();
  std::vector<std::string> files = {path.string()};
  duckdb::vector<duckdb::idx_t> no_projection;

  auto batches = run_two_pipeline_scan(files,
                                       schema.types,
                                       schema.types,
                                       schema.column_ids,
                                       no_projection,
                                       schema.names,
                                       1024 * 1024,
                                       *gpu_space);

  REQUIRE_FALSE(batches.empty());

  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto table = sirius::get_cudf_table_view(*batch);
    REQUIRE(table.num_columns() == 5);
    total_rows += table.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);

  // Verify first batch content
  auto first_table = sirius::get_cudf_table_view(*batches[0]);
  auto ids         = copy_int32_column(first_table.column(0));
  auto labels      = copy_string_column(first_table.column(3));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    REQUIRE(labels[i] == "label_" + std::to_string(ids[i]));
  }
}

TEST_CASE("two-pipeline scan - filter pushdown with integer filter",
          "[two_pipeline_scan][filter][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 2000;
  create_synthetic_table(con, "filter_scan", NUM_ROWS);
  auto path = write_parquet(con, "filter_scan", 500);
  parquet_file_cleanup cleanup{{path}};

  auto schema                    = synthetic_table_schema();
  std::vector<std::string> files = {path.string()};
  duckdb::vector<duckdb::idx_t> no_projection;

  // Filter: id >= 1000  (should yield ~1000 rows)
  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  table_filters->PushFilter(
    duckdb::ColumnIndex(0),
    duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO,
                                              duckdb::Value::INTEGER(1000)));

  auto batches = run_two_pipeline_scan(files,
                                       schema.types,
                                       schema.types,
                                       schema.column_ids,
                                       no_projection,
                                       schema.names,
                                       1024 * 1024,
                                       *gpu_space,
                                       std::move(table_filters));

  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto table = sirius::get_cudf_table_view(*batch);
    auto ids   = copy_int32_column(table.column(0));
    for (auto id : ids) {
      REQUIRE(id >= 1000);
    }
    total_rows += table.num_rows();
  }
  REQUIRE(total_rows == 1000);
}

TEST_CASE("two-pipeline scan - filter on BIGINT column",
          "[two_pipeline_scan][filter][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 1000;
  create_synthetic_table(con, "bigint_filter", NUM_ROWS);
  auto path = write_parquet(con, "bigint_filter", 200);
  parquet_file_cleanup cleanup{{path}};

  auto schema                    = synthetic_table_schema();
  std::vector<std::string> files = {path.string()};
  duckdb::vector<duckdb::idx_t> no_projection;

  // Filter: value < 50000  (value = id * 100, so id < 500 → 500 rows)
  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  table_filters->PushFilter(
    duckdb::ColumnIndex(1),
    duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                              duckdb::Value::BIGINT(50000)));

  auto batches = run_two_pipeline_scan(files,
                                       schema.types,
                                       schema.types,
                                       schema.column_ids,
                                       no_projection,
                                       schema.names,
                                       1024 * 1024,
                                       *gpu_space,
                                       std::move(table_filters));

  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto table  = sirius::get_cudf_table_view(*batch);
    auto values = copy_int64_column(table.column(1));
    for (auto v : values) {
      REQUIRE(v < 50000);
    }
    total_rows += table.num_rows();
  }
  REQUIRE(total_rows == 500);
}

TEST_CASE("two-pipeline scan - projection with filter",
          "[two_pipeline_scan][filter][projection][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 1000;
  create_synthetic_table(con, "proj_filter", NUM_ROWS);
  auto path = write_parquet(con, "proj_filter", 500);
  parquet_file_cleanup cleanup{{path}};

  auto schema                    = synthetic_table_schema();
  std::vector<std::string> files = {path.string()};

  // Project: id (0), price (2)
  duckdb::vector<duckdb::idx_t> projection_ids{0, 2};
  duckdb::vector<duckdb::LogicalType> output_types;
  output_types.push_back(schema.types[0]);
  output_types.push_back(schema.types[2]);

  // Filter: id < 500
  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  table_filters->PushFilter(
    duckdb::ColumnIndex(0),
    duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                              duckdb::Value::INTEGER(500)));

  auto batches = run_two_pipeline_scan(files,
                                       output_types,
                                       schema.types,
                                       schema.column_ids,
                                       projection_ids,
                                       schema.names,
                                       1024 * 1024,
                                       *gpu_space,
                                       std::move(table_filters));

  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto table = sirius::get_cudf_table_view(*batch);
    REQUIRE(table.num_columns() == 2);
    auto ids = copy_int32_column(table.column(0));
    for (auto id : ids) {
      REQUIRE(id < 500);
    }
    total_rows += table.num_rows();
  }
  REQUIRE(total_rows == 500);
}

TEST_CASE("two-pipeline scan - multiple files", "[two_pipeline_scan][multi_file][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con] = sirius::make_test_db_and_connection();

  // Create two separate tables and write each to its own parquet file.
  create_synthetic_table(con, "multi_a", 500);
  create_synthetic_table(con, "multi_b", 500);
  auto path_a = write_parquet(con, "multi_a", 200);
  auto path_b = write_parquet(con, "multi_b", 200);
  parquet_file_cleanup cleanup{{path_a, path_b}};

  // Both files have the same schema.
  auto schema                    = synthetic_table_schema();
  std::vector<std::string> files = {path_a.string(), path_b.string()};
  duckdb::vector<duckdb::idx_t> no_projection;

  auto batches = run_two_pipeline_scan(files,
                                       schema.types,
                                       schema.types,
                                       schema.column_ids,
                                       no_projection,
                                       schema.names,
                                       1024 * 1024,
                                       *gpu_space);

  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    total_rows += sirius::get_cudf_table_view(*batch).num_rows();
  }
  REQUIRE(total_rows == 1000);
}

TEST_CASE("two-pipeline scan - small batch size creates multiple partitions",
          "[two_pipeline_scan][partitioning][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 5000;
  create_synthetic_table(con, "small_batch", NUM_ROWS);
  // Small row groups → many row groups
  auto path = write_parquet(con, "small_batch", 500);
  parquet_file_cleanup cleanup{{path}};

  auto schema                    = synthetic_table_schema();
  std::vector<std::string> files = {path.string()};
  duckdb::vector<duckdb::idx_t> no_projection;

  // Very small batch size to force multiple partitions
  auto batches = run_two_pipeline_scan(files,
                                       schema.types,
                                       schema.types,
                                       schema.column_ids,
                                       no_projection,
                                       schema.names,
                                       1024,  // tiny batch target
                                       *gpu_space);

  // Should produce multiple batches
  REQUIRE(batches.size() > 1);

  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    total_rows += sirius::get_cudf_table_view(*batch).num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);
}

TEST_CASE("gpu_scan_operator - sink and finalize lifecycle", "[gpu_scan_operator][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType::INTEGER);
  sirius::op::scan::sirius_gpu_parquet_scan_operator op(sirius::from_duckdb_vec(types), 0);

  REQUIRE(op.is_source());

  // With no metadata accumulated and no "handoff" port wired to an upstream
  // pipeline, the gpu scan operator has nothing to claim and nothing to defer
  // to — all_ports_empty() is true and get_next_task_hint() returns nullopt.
  REQUIRE(op.all_ports_empty());
  REQUIRE(op.get_next_task_hint() == std::nullopt);
  REQUIRE(op.get_next_task_input_data() == nullptr);

  // Finalize with no metadata → no partitions, still idle.
  op.finalize_partitions();
  REQUIRE(op.all_ports_empty());
  REQUIRE(op.get_next_task_hint() == std::nullopt);
}

TEST_CASE("two-pipeline scan - diverse types with filter on INTEGER",
          "[two_pipeline_scan][types][filter][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 500;
  create_diverse_table(con, "diverse_filter", NUM_ROWS);
  auto path = write_parquet(con, "diverse_filter", 100);
  parquet_file_cleanup cleanup{{path}};

  auto schema                    = diverse_table_schema();
  std::vector<std::string> files = {path.string()};
  duckdb::vector<duckdb::idx_t> no_projection;

  // Filter: id < 100
  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  table_filters->PushFilter(
    duckdb::ColumnIndex(0),
    duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                              duckdb::Value::INTEGER(100)));

  auto batches = run_two_pipeline_scan(files,
                                       schema.types,
                                       schema.types,
                                       schema.column_ids,
                                       no_projection,
                                       schema.names,
                                       1024 * 1024,
                                       *gpu_space,
                                       std::move(table_filters));

  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto table = sirius::get_cudf_table_view(*batch);
    REQUIRE(table.num_columns() == 5);
    auto ids = copy_int32_column(table.column(0));
    for (auto id : ids) {
      REQUIRE(id < 100);
    }
    // Verify string column content is correct for filtered rows
    auto labels = copy_string_column(table.column(3));
    for (std::size_t i = 0; i < ids.size(); ++i) {
      REQUIRE(labels[i] == "label_" + std::to_string(ids[i]));
    }
    total_rows += table.num_rows();
  }
  REQUIRE(total_rows == 100);
}

TEST_CASE("two-pipeline scan - projection on diverse types",
          "[two_pipeline_scan][types][projection][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 300;
  create_diverse_table(con, "diverse_proj", NUM_ROWS);
  auto path = write_parquet(con, "diverse_proj", 100);
  parquet_file_cleanup cleanup{{path}};

  auto schema                    = diverse_table_schema();
  std::vector<std::string> files = {path.string()};

  // Project: label (3) and created (4) — VARCHAR and DATE columns
  duckdb::vector<duckdb::idx_t> projection_ids{3, 4};
  duckdb::vector<duckdb::LogicalType> output_types;
  output_types.push_back(schema.types[3]);
  output_types.push_back(schema.types[4]);

  auto batches = run_two_pipeline_scan(files,
                                       output_types,
                                       schema.types,
                                       schema.column_ids,
                                       projection_ids,
                                       schema.names,
                                       1024 * 1024,
                                       *gpu_space);

  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto table = sirius::get_cudf_table_view(*batch);
    REQUIRE(table.num_columns() == 2);
    // First column should be STRING (label)
    REQUIRE(table.column(0).type().id() == cudf::type_id::STRING);
    total_rows += table.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);
}

TEST_CASE("two-pipeline scan - empty result from filter",
          "[two_pipeline_scan][filter][edge_case][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con] = sirius::make_test_db_and_connection();
  create_synthetic_table(con, "empty_filter", 1000);
  auto path = write_parquet(con, "empty_filter", 500);
  parquet_file_cleanup cleanup{{path}};

  auto schema                    = synthetic_table_schema();
  std::vector<std::string> files = {path.string()};
  duckdb::vector<duckdb::idx_t> no_projection;

  // Filter: id > 99999  (no rows match)
  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  table_filters->PushFilter(
    duckdb::ColumnIndex(0),
    duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_GREATERTHAN,
                                              duckdb::Value::INTEGER(99999)));

  auto batches = run_two_pipeline_scan(files,
                                       schema.types,
                                       schema.types,
                                       schema.column_ids,
                                       no_projection,
                                       schema.names,
                                       1024 * 1024,
                                       *gpu_space,
                                       std::move(table_filters));

  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    total_rows += sirius::get_cudf_table_view(*batch).num_rows();
  }
  REQUIRE(total_rows == 0);
}

TEST_CASE("two-pipeline scan - pure filter column pruning",
          "[two_pipeline_scan][filter][projection][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 1000;
  create_synthetic_table(con, "pure_filter_col", NUM_ROWS);
  auto path = write_parquet(con, "pure_filter_col", 500);
  parquet_file_cleanup cleanup{{path}};

  // Schema: id(0) INTEGER, value(1) BIGINT, price(2) DOUBLE, name(3) VARCHAR
  auto schema                    = synthetic_table_schema();
  std::vector<std::string> files = {path.string()};

  // Output only id(0) and price(2).  value(1) is a pure filter column: it appears
  // in projection_ids but not in output_types, so it should be pruned after filtering.
  //
  // DuckDB convention: projection_ids are indices into column_ids.  The first
  // output_types.size() entries are the real output columns; the rest are pure
  // filter columns.  make_selected_column_indices collects the union of all
  // referenced column_ids indices into a projected_set, then iterates column_ids
  // in order, emitting only those in the set.  post_filter_projection_ids stores
  // the raw projection_id values and uses them to index into the reader output.
  //
  // This only works when the projected_set forms a contiguous range {0..N-1},
  // so that column_ids index == reader output position.  DuckDB's planner
  // guarantees this: column_ids only contains referenced columns, and
  // projection_ids (output + filter-only) covers all of them.
  //
  // With projection_ids = {0, 2, 1}, projected_set = {0, 1, 2} (contiguous),
  // the reader produces 3 columns in column_ids order [id, value, price], and
  // post_filter_projection_ids = {0, 2} selects columns[0]=id and columns[2]=price.
  duckdb::vector<duckdb::LogicalType> output_types;
  output_types.push_back(schema.types[0]);  // id    INTEGER
  output_types.push_back(schema.types[2]);  // price DOUBLE

  duckdb::vector<duckdb::idx_t> projection_ids{0, 2, 1};  // id, price, value(filter-only)

  // Filter: value < 50000  (value = id * 100, so id < 500 → 500 rows)
  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  table_filters->PushFilter(
    duckdb::ColumnIndex(1),
    duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                              duckdb::Value::BIGINT(50000)));

  auto batches = run_two_pipeline_scan(files,
                                       output_types,
                                       schema.types,
                                       schema.column_ids,
                                       projection_ids,
                                       schema.names,
                                       1024 * 1024,
                                       *gpu_space,
                                       std::move(table_filters));

  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto table = sirius::get_cudf_table_view(*batch);
    // Pure filter column (value) should have been pruned — only id and price remain.
    REQUIRE(table.num_columns() == 2);
    REQUIRE(table.column(0).type().id() == cudf::type_id::INT32);
    REQUIRE(table.column(1).type().id() == cudf::type_id::FLOAT64);
    auto ids = copy_int32_column(table.column(0));
    for (auto id : ids) {
      REQUIRE(id < 500);
    }
    total_rows += table.num_rows();
  }
  REQUIRE(total_rows == 500);
}

TEST_CASE("two-pipeline scan - DECIMAL filter falls back to DuckDB expression executor",
          "[two_pipeline_scan][filter][fallback][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 500;
  create_diverse_table(con, "decimal_fallback", NUM_ROWS);
  auto path = write_parquet(con, "decimal_fallback", 200);
  parquet_file_cleanup cleanup{{path}};

  // Schema: id(0) INTEGER, value(1) BIGINT, price(2) DECIMAL(12,2), label(3) VARCHAR, created(4)
  // DATE. A filter on the DECIMAL column will fail cudf AST translation (DECIMAL types are
  // disabled due to a cudf bug), so the DuckDB expression executor fallback path is exercised.
  auto schema                    = diverse_table_schema();
  std::vector<std::string> files = {path.string()};
  duckdb::vector<duckdb::idx_t> no_projection;

  // Filter: price < 100.00  (price_cents = i*100+1, so price = i + 0.01; id < 100 → 100 rows)
  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  table_filters->PushFilter(
    duckdb::ColumnIndex(2),
    duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                              duckdb::Value::DECIMAL(10000, 12, 2)));

  auto batches = run_two_pipeline_scan(files,
                                       schema.types,
                                       schema.types,
                                       schema.column_ids,
                                       no_projection,
                                       schema.names,
                                       1024 * 1024,
                                       *gpu_space,
                                       std::move(table_filters));

  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto table = sirius::get_cudf_table_view(*batch);
    REQUIRE(table.num_columns() == 5);
    auto ids = copy_int32_column(table.column(0));
    for (auto id : ids) {
      REQUIRE(id < 100);
    }
    total_rows += table.num_rows();
  }
  REQUIRE(total_rows == 100);
}
