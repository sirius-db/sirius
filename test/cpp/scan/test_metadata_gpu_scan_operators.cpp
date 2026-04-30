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
#include <algorithm>
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

/// Run the full two-pipeline scan: metadata scan → sink → GPU scan.
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
    {},
    approximate_batch_size);

  // Execute all metadata tasks and sink results into the GPU operator via metadata_op.sink().
  while (!metadata_op.all_ports_empty()) {
    auto input = metadata_op.get_next_task_input_data();
    if (!input) { break; }
    auto output = metadata_op.execute(*input, stream);
    REQUIRE(output);
    metadata_op.sink(*output, stream);
  }

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

TEST_CASE("metadata_scan_operator - projection restricts byte accounting to selected chunks",
          "[metadata_scan][projection][shared_context]")
{
  // Regression test for the name-based DuckDB→parquet column-chunk mapping in the
  // metadata-scan operator.  Constructs a table whose first column ("heavy") holds
  // large VARCHAR payloads and whose remaining columns are small INTEGERs.
  // When projecting only the small columns, the per-partition uncompressed byte
  // count must reflect just those chunks — the VARCHAR chunk must not be included.
  // The loop resolves each projected column name to its parquet chunk index via
  // detail::leaf_indices_for_column; this test verifies that resolution is actually
  // applied and used for byte accounting.
  auto [db_owner, con] = sirius::make_test_db_and_connection();

  std::string const table = "meta_proj_bytes";
  auto create = con.Query("CREATE TABLE " + table + " (heavy VARCHAR, a INTEGER, b INTEGER)");
  REQUIRE(create);
  REQUIRE(!create->HasError());

  // 500 rows of ~200-char VARCHAR; small ints. Single row group so there is one
  // row-group partition and the byte totals are easy to reason about.  Each row
  // gets a unique string so that dictionary encoding cannot collapse the VARCHAR
  // column to near-zero uncompressed bytes.
  constexpr std::size_t NUM_ROWS = 500;
  auto insert                    = con.Query("INSERT INTO " + table +
                          " SELECT lpad(cast(i AS VARCHAR), 200, 'x'), i, i * 2 "
                                             "FROM generate_series(0, " +
                          std::to_string(NUM_ROWS - 1) + ") t(i)");
  REQUIRE(insert);
  REQUIRE(!insert->HasError());

  auto path = write_parquet(con, table, NUM_ROWS);
  parquet_file_cleanup cleanup{{path}};

  duckdb::vector<duckdb::ColumnIndex> column_ids{
    duckdb::ColumnIndex(0), duckdb::ColumnIndex(1), duckdb::ColumnIndex(2)};
  duckdb::vector<std::string> names{"heavy", "a", "b"};
  duckdb::vector<duckdb::LogicalType> full_types{
    duckdb::LogicalType::VARCHAR, duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER};

  std::vector<std::string> files = {path.string()};
  // Use a very large batch target so the whole file fits in one partition — then
  // the reported byte count reflects the full scanned dataset.
  constexpr std::size_t BIG_BATCH = 1ull << 30;

  // --- Full scan (no projection): all chunks contribute. ---
  std::size_t full_uncompressed_bytes = 0;
  {
    sirius::op::scan::sirius_gpu_parquet_scan_operator gpu_op(sirius::from_duckdb_vec(full_types),
                                                              0);
    sirius::op::scan::sirius_parquet_metadata_scan_operator op(
      &gpu_op,
      sirius::from_duckdb_vec(full_types),
      sirius::from_duckdb_vec(full_types),
      0,
      files,
      column_ids,
      /*projection_ids=*/duckdb::vector<duckdb::idx_t>{},
      names,
      /*table_filter_set=*/nullptr,
      /*partition_indices=*/{},
      BIG_BATCH);
    auto input = op.get_next_task_input_data();
    REQUIRE(input);
    auto output = op.execute(*input, cudf::get_default_stream());
    auto* meta  = dynamic_cast<sirius::op::scan::partitioned_parquet_metadata*>(output.get());
    REQUIRE(meta);
    REQUIRE(meta->row_group_partitions.size() == 1);
    full_uncompressed_bytes = meta->row_group_partitions[0].reserved_uncompressed_bytes;
  }

  // --- Projected scan: only the two INTEGER columns ("a", "b"). ---
  std::size_t projected_uncompressed_bytes = 0;
  {
    duckdb::vector<duckdb::LogicalType> projected_types{duckdb::LogicalType::INTEGER,
                                                        duckdb::LogicalType::INTEGER};
    duckdb::vector<duckdb::idx_t> projection_ids{1, 2};
    sirius::op::scan::sirius_gpu_parquet_scan_operator gpu_op(
      sirius::from_duckdb_vec(projected_types), 0);
    sirius::op::scan::sirius_parquet_metadata_scan_operator op(
      &gpu_op,
      sirius::from_duckdb_vec(projected_types),
      sirius::from_duckdb_vec(full_types),
      0,
      files,
      column_ids,
      projection_ids,
      names,
      /*table_filter_set=*/nullptr,
      /*partition_indices=*/{},
      BIG_BATCH);
    auto input = op.get_next_task_input_data();
    REQUIRE(input);
    auto output = op.execute(*input, cudf::get_default_stream());
    auto* meta  = dynamic_cast<sirius::op::scan::partitioned_parquet_metadata*>(output.get());
    REQUIRE(meta);
    REQUIRE(meta->row_group_partitions.size() == 1);
    projected_uncompressed_bytes = meta->row_group_partitions[0].reserved_uncompressed_bytes;
  }

  // Two 4-byte ints × 500 rows = 4000 bytes of column payload, plus some per-page
  // overhead. The VARCHAR payload alone is ~100 KB, so the projected byte count
  // must be at least an order of magnitude smaller than the full-scan count.
  REQUIRE(projected_uncompressed_bytes > 0);
  REQUIRE(full_uncompressed_bytes > 10 * projected_uncompressed_bytes);
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

TEST_CASE("gpu_scan_operator - idle without handoff port", "[gpu_scan_operator][shared_context]")
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

//===----------------------------------------------------------------------===//
// Two-pipeline scan tests for parquet files containing STRUCT and LIST
// columns.  cuDF reassembles nested columns into a single top-level
// cudf::column when the parent name is passed via parquet_reader_options::set_column_names, so the
// metadata-scan operator need only (a) push every leaf chunk into selected_chunk_indices for byte
// accounting and (b) accept that a single DuckDB column may resolve to multiple parquet leaves.
//
// Out of scope (not tested here):
//   - Filter on a sub-field (e.g. WHERE s.a > 10): requires a
//     duckdb::Expression referencing a struct field rather than a column-level
//     TableFilter, and depends on GpuExpressionExecutor sub-field support.
//   - Pure-filter nested columns: TableFilter does not naturally express a
//     filter on a STRUCT/LIST column at the column level.
//===----------------------------------------------------------------------===//

//---------- Basic nested projection ----------//

TEST_CASE("two-pipeline scan - project STRUCT column",
          "[two_pipeline_scan][nested][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 300;

  std::string const table = "nested_struct_scan";
  auto create =
    con.Query("CREATE TABLE " + table + " (id INTEGER, s STRUCT(a INTEGER, b VARCHAR))");
  REQUIRE(create);
  REQUIRE(!create->HasError());

  auto insert = con.Query("INSERT INTO " + table +
                          " SELECT i, {'a': i, 'b': 'row_' || cast(i AS VARCHAR)} "
                          "FROM generate_series(0, " +
                          std::to_string(NUM_ROWS - 1) + ") t(i)");
  REQUIRE(insert);
  REQUIRE(!insert->HasError());

  auto path = write_parquet(con, table, 100);
  parquet_file_cleanup cleanup{{path}};

  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  duckdb::vector<std::string> names{"id", "s"};
  duckdb::vector<duckdb::idx_t> projection_ids{1};
  duckdb::vector<duckdb::LogicalType> output_types{duckdb::LogicalType::STRUCT({})};
  duckdb::vector<duckdb::LogicalType> returned_types{duckdb::LogicalType::INTEGER,
                                                     duckdb::LogicalType::STRUCT({})};

  std::vector<std::string> files = {path.string()};
  auto batches                   = run_two_pipeline_scan(files,
                                       output_types,
                                       returned_types,
                                       column_ids,
                                       projection_ids,
                                       names,
                                       1024 * 1024,
                                       *gpu_space);

  REQUIRE_FALSE(batches.empty());
  std::vector<int32_t> all_a;
  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto view = sirius::get_cudf_table_view(*batch);
    REQUIRE(view.num_columns() == 1);
    auto const struct_col = view.column(0);
    REQUIRE(struct_col.type().id() == cudf::type_id::STRUCT);
    REQUIRE(struct_col.num_children() == 2);
    REQUIRE(struct_col.child(0).type().id() == cudf::type_id::INT32);
    REQUIRE(struct_col.child(1).type().id() == cudf::type_id::STRING);

    // Child values must be paired: s.b == "row_" || cast(s.a AS VARCHAR).
    auto a = copy_int32_column(struct_col.child(0));
    auto b = copy_string_column(struct_col.child(1));
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      REQUIRE(b[i] == "row_" + std::to_string(a[i]));
    }
    all_a.insert(all_a.end(), a.begin(), a.end());
    total_rows += view.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);

  // Every expected id must appear exactly once across batches.
  std::sort(all_a.begin(), all_a.end());
  REQUIRE(all_a.size() == NUM_ROWS);
  for (std::size_t i = 0; i < NUM_ROWS; ++i) {
    REQUIRE(all_a[i] == static_cast<int32_t>(i));
  }
}

TEST_CASE("two-pipeline scan - project LIST column", "[two_pipeline_scan][nested][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 300;

  std::string const table = "nested_list_scan";
  auto create             = con.Query("CREATE TABLE " + table + " (id INTEGER, l INTEGER[])");
  REQUIRE(create);
  REQUIRE(!create->HasError());

  auto insert =
    con.Query("INSERT INTO " + table + " SELECT i, [i, i+1, i+2] FROM generate_series(0, " +
              std::to_string(NUM_ROWS - 1) + ") t(i)");
  REQUIRE(insert);
  REQUIRE(!insert->HasError());

  auto path = write_parquet(con, table, 100);
  parquet_file_cleanup cleanup{{path}};

  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  duckdb::vector<std::string> names{"id", "l"};
  duckdb::vector<duckdb::idx_t> projection_ids{1};
  duckdb::vector<duckdb::LogicalType> output_types{
    duckdb::LogicalType::LIST(duckdb::LogicalType::INTEGER)};
  duckdb::vector<duckdb::LogicalType> returned_types{
    duckdb::LogicalType::INTEGER, duckdb::LogicalType::LIST(duckdb::LogicalType::INTEGER)};

  std::vector<std::string> files = {path.string()};
  auto batches                   = run_two_pipeline_scan(files,
                                       output_types,
                                       returned_types,
                                       column_ids,
                                       projection_ids,
                                       names,
                                       1024 * 1024,
                                       *gpu_space);

  REQUIRE_FALSE(batches.empty());
  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto view = sirius::get_cudf_table_view(*batch);
    REQUIRE(view.num_columns() == 1);
    REQUIRE(view.column(0).type().id() == cudf::type_id::LIST);
    total_rows += view.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);
}

TEST_CASE("two-pipeline scan - table with nested column, project flat only",
          "[two_pipeline_scan][nested][shared_context]")
{
  // Regression: the mere presence of a STRUCT column in the table's schema
  // should not affect a scan that projects only flat columns.
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 200;

  std::string const table = "mixed_nested_scan";
  auto create             = con.Query("CREATE TABLE " + table +
                          " (id INTEGER, s STRUCT(a INTEGER, b VARCHAR), tail INTEGER)");
  REQUIRE(create);
  REQUIRE(!create->HasError());

  auto insert = con.Query("INSERT INTO " + table +
                          " SELECT i, {'a': i, 'b': 'x'}, i * 2 "
                          "FROM generate_series(0, " +
                          std::to_string(NUM_ROWS - 1) + ") t(i)");
  REQUIRE(insert);
  REQUIRE(!insert->HasError());

  auto path = write_parquet(con, table, 100);
  parquet_file_cleanup cleanup{{path}};

  duckdb::vector<duckdb::ColumnIndex> column_ids{
    duckdb::ColumnIndex(0), duckdb::ColumnIndex(1), duckdb::ColumnIndex(2)};
  duckdb::vector<std::string> names{"id", "s", "tail"};
  duckdb::vector<duckdb::idx_t> projection_ids{0, 2};
  duckdb::vector<duckdb::LogicalType> output_types{duckdb::LogicalType::INTEGER,
                                                   duckdb::LogicalType::INTEGER};
  duckdb::vector<duckdb::LogicalType> returned_types{
    duckdb::LogicalType::INTEGER, duckdb::LogicalType::STRUCT({}), duckdb::LogicalType::INTEGER};

  std::vector<std::string> files = {path.string()};
  auto batches                   = run_two_pipeline_scan(files,
                                       output_types,
                                       returned_types,
                                       column_ids,
                                       projection_ids,
                                       names,
                                       1024 * 1024,
                                       *gpu_space);

  REQUIRE_FALSE(batches.empty());
  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto view = sirius::get_cudf_table_view(*batch);
    REQUIRE(view.num_columns() == 2);
    REQUIRE(view.column(0).type().id() == cudf::type_id::INT32);
    REQUIRE(view.column(1).type().id() == cudf::type_id::INT32);
    total_rows += view.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);
}

//---------- Mixed and multi-file ----------//

TEST_CASE("two-pipeline scan - mixed flat and nested projection",
          "[two_pipeline_scan][nested][shared_context]")
{
  // Project one flat column alongside one nested column.  Verifies that
  // output-column ordering is preserved when the projection spans both.
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 200;

  std::string const table = "mixed_flat_nested";
  auto create =
    con.Query("CREATE TABLE " + table + " (id INTEGER, s STRUCT(a INTEGER, b VARCHAR))");
  REQUIRE(create);
  REQUIRE(!create->HasError());

  auto insert = con.Query("INSERT INTO " + table +
                          " SELECT i, {'a': i, 'b': 'v'} "
                          "FROM generate_series(0, " +
                          std::to_string(NUM_ROWS - 1) + ") t(i)");
  REQUIRE(insert);
  REQUIRE(!insert->HasError());

  auto path = write_parquet(con, table, 100);
  parquet_file_cleanup cleanup{{path}};

  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  duckdb::vector<std::string> names{"id", "s"};
  duckdb::vector<duckdb::idx_t> projection_ids{0, 1};
  duckdb::vector<duckdb::LogicalType> output_types{duckdb::LogicalType::INTEGER,
                                                   duckdb::LogicalType::STRUCT({})};
  duckdb::vector<duckdb::LogicalType> returned_types{duckdb::LogicalType::INTEGER,
                                                     duckdb::LogicalType::STRUCT({})};

  std::vector<std::string> files = {path.string()};
  auto batches                   = run_two_pipeline_scan(files,
                                       output_types,
                                       returned_types,
                                       column_ids,
                                       projection_ids,
                                       names,
                                       1024 * 1024,
                                       *gpu_space);

  REQUIRE_FALSE(batches.empty());
  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto view = sirius::get_cudf_table_view(*batch);
    REQUIRE(view.num_columns() == 2);
    REQUIRE(view.column(0).type().id() == cudf::type_id::INT32);
    REQUIRE(view.column(1).type().id() == cudf::type_id::STRUCT);
    total_rows += view.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);
}

TEST_CASE("two-pipeline scan - nested column across multiple files",
          "[two_pipeline_scan][nested][multi_file][shared_context]")
{
  // Two files with identical schemas; name-based resolution must work per file.
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]            = sirius::make_test_db_and_connection();
  constexpr std::size_t ROWS_EACH = 150;

  std::vector<std::filesystem::path> paths;
  for (auto const& name : {"multi_nested_a", "multi_nested_b"}) {
    auto create = con.Query(std::string("CREATE TABLE ") + name +
                            " (id INTEGER, s STRUCT(a INTEGER, b VARCHAR))");
    REQUIRE(create);
    REQUIRE(!create->HasError());

    auto insert = con.Query(std::string("INSERT INTO ") + name +
                            " SELECT i, {'a': i, 'b': 'y'} "
                            "FROM generate_series(0, " +
                            std::to_string(ROWS_EACH - 1) + ") t(i)");
    REQUIRE(insert);
    REQUIRE(!insert->HasError());

    paths.push_back(write_parquet(con, name, 50));
  }
  parquet_file_cleanup cleanup{paths};

  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  duckdb::vector<std::string> names{"id", "s"};
  duckdb::vector<duckdb::idx_t> projection_ids{1};
  duckdb::vector<duckdb::LogicalType> output_types{duckdb::LogicalType::STRUCT({})};
  duckdb::vector<duckdb::LogicalType> returned_types{duckdb::LogicalType::INTEGER,
                                                     duckdb::LogicalType::STRUCT({})};

  std::vector<std::string> files;
  for (auto const& p : paths) {
    files.push_back(p.string());
  }

  auto batches = run_two_pipeline_scan(files,
                                       output_types,
                                       returned_types,
                                       column_ids,
                                       projection_ids,
                                       names,
                                       1024 * 1024,
                                       *gpu_space);

  REQUIRE_FALSE(batches.empty());
  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto view = sirius::get_cudf_table_view(*batch);
    REQUIRE(view.num_columns() == 1);
    REQUIRE(view.column(0).type().id() == cudf::type_id::STRUCT);
    total_rows += view.num_rows();
  }
  REQUIRE(total_rows == 2 * ROWS_EACH);
}

//---------- Deeper nesting ----------//

TEST_CASE("two-pipeline scan - STRUCT inside STRUCT", "[two_pipeline_scan][nested][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 100;

  std::string const table = "struct_in_struct";
  auto create             = con.Query("CREATE TABLE " + table +
                          " (id INTEGER, s STRUCT(a INTEGER, b STRUCT(c INTEGER, d VARCHAR)))");
  REQUIRE(create);
  REQUIRE(!create->HasError());

  auto insert = con.Query("INSERT INTO " + table +
                          " SELECT i, {'a': i, 'b': {'c': i * 10, 'd': 'inner'}} "
                          "FROM generate_series(0, " +
                          std::to_string(NUM_ROWS - 1) + ") t(i)");
  REQUIRE(insert);
  REQUIRE(!insert->HasError());

  auto path = write_parquet(con, table);
  parquet_file_cleanup cleanup{{path}};

  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  duckdb::vector<std::string> names{"id", "s"};
  duckdb::vector<duckdb::idx_t> projection_ids{1};
  duckdb::vector<duckdb::LogicalType> output_types{duckdb::LogicalType::STRUCT({})};
  duckdb::vector<duckdb::LogicalType> returned_types{duckdb::LogicalType::INTEGER,
                                                     duckdb::LogicalType::STRUCT({})};

  std::vector<std::string> files = {path.string()};
  auto batches                   = run_two_pipeline_scan(files,
                                       output_types,
                                       returned_types,
                                       column_ids,
                                       projection_ids,
                                       names,
                                       1024 * 1024,
                                       *gpu_space);

  REQUIRE_FALSE(batches.empty());
  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto view = sirius::get_cudf_table_view(*batch);
    REQUIRE(view.num_columns() == 1);
    REQUIRE(view.column(0).type().id() == cudf::type_id::STRUCT);
    REQUIRE(view.column(0).num_children() == 2);
    REQUIRE(view.column(0).child(1).type().id() == cudf::type_id::STRUCT);
    total_rows += view.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);
}

TEST_CASE("two-pipeline scan - LIST of LIST", "[two_pipeline_scan][nested][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 100;

  std::string const table = "list_of_list";
  auto create             = con.Query("CREATE TABLE " + table + " (id INTEGER, ll INTEGER[][])");
  REQUIRE(create);
  REQUIRE(!create->HasError());

  auto insert = con.Query("INSERT INTO " + table +
                          " SELECT i, [[i, i+1], [i+2]] "
                          "FROM generate_series(0, " +
                          std::to_string(NUM_ROWS - 1) + ") t(i)");
  REQUIRE(insert);
  REQUIRE(!insert->HasError());

  auto path = write_parquet(con, table);
  parquet_file_cleanup cleanup{{path}};

  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  duckdb::vector<std::string> names{"id", "ll"};
  duckdb::vector<duckdb::idx_t> projection_ids{1};
  duckdb::vector<duckdb::LogicalType> output_types{
    duckdb::LogicalType::LIST(duckdb::LogicalType::LIST(duckdb::LogicalType::INTEGER))};
  duckdb::vector<duckdb::LogicalType> returned_types{
    duckdb::LogicalType::INTEGER,
    duckdb::LogicalType::LIST(duckdb::LogicalType::LIST(duckdb::LogicalType::INTEGER))};

  std::vector<std::string> files = {path.string()};
  auto batches                   = run_two_pipeline_scan(files,
                                       output_types,
                                       returned_types,
                                       column_ids,
                                       projection_ids,
                                       names,
                                       1024 * 1024,
                                       *gpu_space);

  REQUIRE_FALSE(batches.empty());
  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto view = sirius::get_cudf_table_view(*batch);
    REQUIRE(view.num_columns() == 1);
    REQUIRE(view.column(0).type().id() == cudf::type_id::LIST);
    total_rows += view.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);
}

TEST_CASE("two-pipeline scan - LIST inside STRUCT", "[two_pipeline_scan][nested][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 100;

  std::string const table = "list_in_struct";
  auto create =
    con.Query("CREATE TABLE " + table + " (id INTEGER, s STRUCT(a INTEGER, l INTEGER[]))");
  REQUIRE(create);
  REQUIRE(!create->HasError());

  auto insert = con.Query("INSERT INTO " + table +
                          " SELECT i, {'a': i, 'l': [i, i+1, i+2]} "
                          "FROM generate_series(0, " +
                          std::to_string(NUM_ROWS - 1) + ") t(i)");
  REQUIRE(insert);
  REQUIRE(!insert->HasError());

  auto path = write_parquet(con, table);
  parquet_file_cleanup cleanup{{path}};

  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  duckdb::vector<std::string> names{"id", "s"};
  duckdb::vector<duckdb::idx_t> projection_ids{1};
  duckdb::vector<duckdb::LogicalType> output_types{duckdb::LogicalType::STRUCT({})};
  duckdb::vector<duckdb::LogicalType> returned_types{duckdb::LogicalType::INTEGER,
                                                     duckdb::LogicalType::STRUCT({})};

  std::vector<std::string> files = {path.string()};
  auto batches                   = run_two_pipeline_scan(files,
                                       output_types,
                                       returned_types,
                                       column_ids,
                                       projection_ids,
                                       names,
                                       1024 * 1024,
                                       *gpu_space);

  REQUIRE_FALSE(batches.empty());
  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto view = sirius::get_cudf_table_view(*batch);
    REQUIRE(view.num_columns() == 1);
    REQUIRE(view.column(0).type().id() == cudf::type_id::STRUCT);
    REQUIRE(view.column(0).num_children() == 2);
    REQUIRE(view.column(0).child(1).type().id() == cudf::type_id::LIST);
    total_rows += view.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);
}

//---------- Filter interaction ----------//

TEST_CASE("two-pipeline scan - flat filter with nested projection",
          "[two_pipeline_scan][nested][filter][shared_context]")
{
  // Filter pushdown on a flat column while the projection includes a nested
  // column.  The filter runs at scan time; the nested column is projected as
  // a single top-level cudf::column.
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 500;
  constexpr int32_t FILTER_MIN   = 100;

  std::string const table = "filter_with_nested";
  auto create =
    con.Query("CREATE TABLE " + table + " (id INTEGER, s STRUCT(a INTEGER, b VARCHAR))");
  REQUIRE(create);
  REQUIRE(!create->HasError());

  auto insert = con.Query("INSERT INTO " + table +
                          " SELECT i, {'a': i, 'b': 'v'} "
                          "FROM generate_series(0, " +
                          std::to_string(NUM_ROWS - 1) + ") t(i)");
  REQUIRE(insert);
  REQUIRE(!insert->HasError());

  auto path = write_parquet(con, table, 100);
  parquet_file_cleanup cleanup{{path}};

  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  duckdb::vector<std::string> names{"id", "s"};
  duckdb::vector<duckdb::idx_t> projection_ids{0, 1};
  duckdb::vector<duckdb::LogicalType> output_types{duckdb::LogicalType::INTEGER,
                                                   duckdb::LogicalType::STRUCT({})};
  duckdb::vector<duckdb::LogicalType> returned_types{duckdb::LogicalType::INTEGER,
                                                     duckdb::LogicalType::STRUCT({})};

  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  table_filters->PushFilter(
    duckdb::ColumnIndex(0),
    duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO,
                                              duckdb::Value::INTEGER(FILTER_MIN)));

  std::vector<std::string> files = {path.string()};
  auto batches                   = run_two_pipeline_scan(files,
                                       output_types,
                                       returned_types,
                                       column_ids,
                                       projection_ids,
                                       names,
                                       1024 * 1024,
                                       *gpu_space,
                                       std::move(table_filters));

  REQUIRE_FALSE(batches.empty());
  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto view = sirius::get_cudf_table_view(*batch);
    REQUIRE(view.num_columns() == 2);
    REQUIRE(view.column(0).type().id() == cudf::type_id::INT32);
    REQUIRE(view.column(1).type().id() == cudf::type_id::STRUCT);
    auto ids = copy_int32_column(view.column(0));
    for (auto id : ids) {
      REQUIRE(id >= FILTER_MIN);
    }
    total_rows += view.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS - FILTER_MIN);
}

TEST_CASE("two-pipeline scan - filter on nested column throws cleanly",
          "[two_pipeline_scan][nested][filter][negative][shared_context]")
{
  // cuDF's AST has no operand support for STRUCT or LIST in comparison or
  // binary ops, so a filter pushed directly against a nested column cannot be
  // pushed down or evaluated post-scan.  The scan must surface this as an
  // exception rather than a crash.  This test pins down the "throws cleanly"
  // contract; the precise layer that throws (filter-to-expression conversion,
  // AST translation, or the post-scan expression executor fallback) is left
  // unspecified.
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 100;

  std::string const table = "filter_on_nested";
  auto create =
    con.Query("CREATE TABLE " + table + " (id INTEGER, s STRUCT(a INTEGER, b VARCHAR))");
  REQUIRE(create);
  REQUIRE(!create->HasError());

  auto insert = con.Query("INSERT INTO " + table +
                          " SELECT i, {'a': i, 'b': 'v'} "
                          "FROM generate_series(0, " +
                          std::to_string(NUM_ROWS - 1) + ") t(i)");
  REQUIRE(insert);
  REQUIRE(!insert->HasError());

  auto path = write_parquet(con, table, 100);
  parquet_file_cleanup cleanup{{path}};

  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  duckdb::vector<std::string> names{"id", "s"};
  duckdb::vector<duckdb::idx_t> projection_ids{0, 1};
  duckdb::vector<duckdb::LogicalType> output_types{duckdb::LogicalType::INTEGER,
                                                   duckdb::LogicalType::STRUCT({})};
  duckdb::vector<duckdb::LogicalType> returned_types{duckdb::LogicalType::INTEGER,
                                                     duckdb::LogicalType::STRUCT({})};

  // Push a constant filter against the STRUCT column.  The constant value type
  // is unimportant — the point is that the filtered column is nested.
  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  table_filters->PushFilter(duckdb::ColumnIndex(1),
                            duckdb::make_uniq<duckdb::ConstantFilter>(
                              duckdb::ExpressionType::COMPARE_EQUAL, duckdb::Value::INTEGER(0)));

  std::vector<std::string> files = {path.string()};
  REQUIRE_THROWS(run_two_pipeline_scan(files,
                                       output_types,
                                       returned_types,
                                       column_ids,
                                       projection_ids,
                                       names,
                                       1024 * 1024,
                                       *gpu_space,
                                       std::move(table_filters)));
}

//---------- Edge cases ----------//

TEST_CASE("two-pipeline scan - LIST with empty and populated rows",
          "[two_pipeline_scan][nested][edge][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 100;

  std::string const table = "empty_list_scan";
  auto create             = con.Query("CREATE TABLE " + table + " (id INTEGER, l INTEGER[])");
  REQUIRE(create);
  REQUIRE(!create->HasError());

  auto insert = con.Query("INSERT INTO " + table +
                          " SELECT i, CASE WHEN i % 2 = 0 THEN [] ELSE [i, i+1] END "
                          "FROM generate_series(0, " +
                          std::to_string(NUM_ROWS - 1) + ") t(i)");
  REQUIRE(insert);
  REQUIRE(!insert->HasError());

  auto path = write_parquet(con, table);
  parquet_file_cleanup cleanup{{path}};

  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  duckdb::vector<std::string> names{"id", "l"};
  duckdb::vector<duckdb::idx_t> projection_ids{1};
  duckdb::vector<duckdb::LogicalType> output_types{
    duckdb::LogicalType::LIST(duckdb::LogicalType::INTEGER)};
  duckdb::vector<duckdb::LogicalType> returned_types{
    duckdb::LogicalType::INTEGER, duckdb::LogicalType::LIST(duckdb::LogicalType::INTEGER)};

  std::vector<std::string> files = {path.string()};
  auto batches                   = run_two_pipeline_scan(files,
                                       output_types,
                                       returned_types,
                                       column_ids,
                                       projection_ids,
                                       names,
                                       1024 * 1024,
                                       *gpu_space);

  REQUIRE_FALSE(batches.empty());
  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto view = sirius::get_cudf_table_view(*batch);
    REQUIRE(view.num_columns() == 1);
    REQUIRE(view.column(0).type().id() == cudf::type_id::LIST);
    total_rows += view.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);
}

TEST_CASE("two-pipeline scan - STRUCT with NULL rows",
          "[two_pipeline_scan][nested][edge][shared_context]")
{
  auto memory_manager = initialize_memory_manager();
  auto* gpu_space     = get_space(*memory_manager, Tier::GPU);
  REQUIRE(gpu_space);

  auto [db_owner, con]           = sirius::make_test_db_and_connection();
  constexpr std::size_t NUM_ROWS = 100;

  std::string const table = "null_struct_scan";
  auto create =
    con.Query("CREATE TABLE " + table + " (id INTEGER, s STRUCT(a INTEGER, b VARCHAR))");
  REQUIRE(create);
  REQUIRE(!create->HasError());

  auto insert = con.Query("INSERT INTO " + table +
                          " SELECT i, CASE WHEN i % 3 = 0 THEN NULL ELSE {'a': i, 'b': 'v'} END "
                          "FROM generate_series(0, " +
                          std::to_string(NUM_ROWS - 1) + ") t(i)");
  REQUIRE(insert);
  REQUIRE(!insert->HasError());

  auto path = write_parquet(con, table);
  parquet_file_cleanup cleanup{{path}};

  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  duckdb::vector<std::string> names{"id", "s"};
  duckdb::vector<duckdb::idx_t> projection_ids{1};
  duckdb::vector<duckdb::LogicalType> output_types{duckdb::LogicalType::STRUCT({})};
  duckdb::vector<duckdb::LogicalType> returned_types{duckdb::LogicalType::INTEGER,
                                                     duckdb::LogicalType::STRUCT({})};

  std::vector<std::string> files = {path.string()};
  auto batches                   = run_two_pipeline_scan(files,
                                       output_types,
                                       returned_types,
                                       column_ids,
                                       projection_ids,
                                       names,
                                       1024 * 1024,
                                       *gpu_space);

  REQUIRE_FALSE(batches.empty());
  std::size_t total_rows = 0;
  for (auto const& batch : batches) {
    auto view = sirius::get_cudf_table_view(*batch);
    REQUIRE(view.num_columns() == 1);
    REQUIRE(view.column(0).type().id() == cudf::type_id::STRUCT);
    total_rows += view.num_rows();
  }
  REQUIRE(total_rows == NUM_ROWS);
}
