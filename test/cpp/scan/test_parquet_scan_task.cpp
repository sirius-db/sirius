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
#include <op/scan/duckdb_scan_task_queue.hpp>
#include <op/scan/parquet_scan_task.hpp>
#include <op/sirius_physical_parquet_scan.hpp>
#include <parallel/task_executor.hpp>
#include <pipeline/sirius_pipeline_task_states.hpp>

// cucascade
#include <cucascade/memory/memory_reservation_manager.hpp>

// rmm
#include <rmm/cuda_stream.hpp>

// cudf
#include <cudf/logger.hpp>

#include <rapids_logger/logger.hpp>

// duckdb
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/parser/expression/constant_expression.hpp>
#include <duckdb/parser/expression/function_expression.hpp>
#include <duckdb/parser/tableref/table_function_ref.hpp>

// standard library
#include <filesystem>
#include <string>
#include <thread>

using namespace sirius;
using namespace sirius::scan_test_utils;
using namespace cucascade::memory;

using table_creator_t = void (*)(duckdb::Connection&,
                                 std::string const& table_name,
                                 size_t num_rows);

using batch_validator_t = void (*)(const std::vector<std::shared_ptr<cucascade::data_batch>>&,
                                   size_t,
                                   cucascade::memory::memory_reservation_manager&,
                                   rmm::cuda_stream_view);

static std::unique_ptr<sirius::op::sirius_physical_parquet_scan> make_parquet_scan(
  duckdb::ClientContext& ctx,
  std::string const& parquet_path,
  duckdb::vector<duckdb::idx_t> projection_ids = {})
{
  auto& table_function_entry = duckdb::Catalog::GetEntry<duckdb::TableFunctionCatalogEntry>(
    ctx, INVALID_CATALOG, DEFAULT_SCHEMA, "parquet_scan");

  duckdb::vector<duckdb::LogicalType> arg_types;
  arg_types.emplace_back(duckdb::LogicalTypeId::VARCHAR);
  auto table_function = table_function_entry.functions.GetFunctionByArguments(ctx, arg_types);

  duckdb::vector<duckdb::Value> inputs;
  inputs.emplace_back(parquet_path);

  duckdb::named_parameter_map_t named_parameters;
  duckdb::vector<duckdb::LogicalType> input_table_types;
  duckdb::vector<std::string> input_table_names;

  duckdb::TableFunctionRef ref;
  duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children;
  children.push_back(duckdb::make_uniq<duckdb::ConstantExpression>(duckdb::Value(parquet_path)));
  ref.function = duckdb::make_uniq<duckdb::FunctionExpression>(
    "parquet_scan", std::move(children), nullptr, nullptr, false, false, false);

  duckdb::vector<duckdb::LogicalType> return_types;
  duckdb::vector<std::string> names;
  duckdb::TableFunctionBindInput bind_input(inputs,
                                            named_parameters,
                                            input_table_types,
                                            input_table_names,
                                            nullptr,
                                            nullptr,
                                            table_function,
                                            ref);
  auto bind_data = table_function.bind(ctx, bind_input, return_types, names);
  REQUIRE(bind_data);

  duckdb::vector<duckdb::ColumnIndex> column_ids;
  for (size_t i = 0; i < return_types.size(); ++i) {
    column_ids.emplace_back(duckdb::ColumnIndex(i));
  }

  duckdb::virtual_column_map_t virtual_columns;
  if (auto* multi_bind = dynamic_cast<duckdb::MultiFileBindData*>(bind_data.get())) {
    virtual_columns = multi_bind->virtual_columns;
  }

  duckdb::ExtraOperatorInfo extra_info;
  duckdb::vector<duckdb::LogicalType> output_types;
  if (projection_ids.empty() || projection_ids.size() == return_types.size()) {
    output_types = return_types;
  } else {
    output_types.reserve(projection_ids.size());
    for (auto const projection_id : projection_ids) {
      REQUIRE(projection_id < column_ids.size());
      output_types.push_back(return_types[column_ids[projection_id].GetPrimaryIndex()]);
    }
  }

  return std::make_unique<sirius::op::sirius_physical_parquet_scan>(output_types,
                                                                    table_function,
                                                                    std::move(bind_data),
                                                                    return_types,
                                                                    std::move(column_ids),
                                                                    std::move(projection_ids),
                                                                    std::move(names),
                                                                    nullptr,
                                                                    0,
                                                                    std::move(extra_info),
                                                                    duckdb::vector<duckdb::Value>(),
                                                                    std::move(virtual_columns));
}

static void write_parquet_from_table_to_path(duckdb::Connection& con,
                                             std::string const& table_name,
                                             std::filesystem::path const& parquet_path,
                                             size_t row_group_size = 0)
{
  std::string sql;
  if (row_group_size != 0) {
    sql = "COPY " + table_name + " TO '" + parquet_path.string() +
          "' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE " + std::to_string(row_group_size) +
          ")";
  } else {
    sql = "COPY " + table_name + " TO '" + parquet_path.string() +
          "' (FORMAT PARQUET, COMPRESSION zstd)";
  }
  auto result = con.Query(sql);
  REQUIRE(result);
  REQUIRE(!result->HasError());
}

static std::filesystem::path write_parquet_from_table(duckdb::Connection& con,
                                                      std::string const& table_name,
                                                      size_t row_group_size = 0)
{
  auto parquet_path = std::filesystem::temp_directory_path() /
                      (table_name + "_" + std::to_string(row_group_size) + ".parquet");
  write_parquet_from_table_to_path(con, table_name, parquet_path, row_group_size);
  return parquet_path;
}

static void validate_scanned_batches_suppress_cudf(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& batches,
  size_t expected_rows,
  cucascade::memory::memory_reservation_manager& mem_mgr,
  rmm::cuda_stream_view stream)
{
  rapids_logger::log_level_setter guard(cudf::default_logger(), rapids_logger::level_enum::error);
  validate_scanned_batches(batches, expected_rows, mem_mgr, stream);
}

static void create_synthetic_table_with_nested_list(duckdb::Connection& con,
                                                    std::string const& table_name,
                                                    size_t num_rows)
{
  // clang-format off
  std::string create_sql = "CREATE TABLE " + table_name + " ("
                           "id INTEGER, "
                           "value BIGINT, "
                           "price DOUBLE, "
                           "name VARCHAR, "
                           "nested INTEGER[]"
                           ");";
  // clang-format on
  auto result = con.Query(create_sql);
  REQUIRE(result);
  REQUIRE(!result->HasError());

  constexpr size_t BATCH_SIZE = 1000;
  for (size_t start = 0; start < num_rows; start += BATCH_SIZE) {
    size_t end             = std::min(start + BATCH_SIZE, num_rows);
    std::string insert_sql = "INSERT INTO " + table_name + " VALUES ";

    for (size_t i = start; i < end; ++i) {
      if (i > start) { insert_sql += ", "; }
      auto id          = static_cast<int32_t>(i);
      auto value       = static_cast<int64_t>(i * 100);
      auto price       = static_cast<double>(i) * 1.5;
      std::string name = "item_" + std::to_string(i);
      insert_sql += "(" + std::to_string(id) + ", " + std::to_string(value) + ", " +
                    std::to_string(price) + ", " + "'" + name + "', " + "[" + std::to_string(id) +
                    ", " + std::to_string(id + 1) + "])";
    }

    result = con.Query(insert_sql);
    REQUIRE(result);
    REQUIRE(!result->HasError());
  }
}

static void create_synthetic_table_with_offset(duckdb::Connection& con,
                                               std::string const& table_name,
                                               size_t num_rows,
                                               size_t start_id)
{
  // clang-format off
  std::string create_sql = "CREATE TABLE " + table_name + " ("
                           "id INTEGER, "
                           "value BIGINT, "
                           "price DOUBLE, "
                           "name VARCHAR"
                           ");";
  // clang-format on
  auto result = con.Query(create_sql);
  REQUIRE(result);
  REQUIRE(!result->HasError());

  constexpr size_t BATCH_SIZE = 1000;
  for (size_t start = 0; start < num_rows; start += BATCH_SIZE) {
    size_t end             = std::min(start + BATCH_SIZE, num_rows);
    std::string insert_sql = "INSERT INTO " + table_name + " VALUES ";

    for (size_t i = start; i < end; ++i) {
      if (i > start) { insert_sql += ", "; }
      auto id          = static_cast<int32_t>(start_id + i);
      auto value       = static_cast<int64_t>(id * 100);
      auto price       = static_cast<double>(id) * 1.5;
      std::string name = "item_" + std::to_string(id);
      insert_sql += "(" + std::to_string(id) + ", " + std::to_string(value) + ", " +
                    std::to_string(price) + ", " + "'" + name + "')";
    }

    result = con.Query(insert_sql);
    REQUIRE(result);
    REQUIRE(!result->HasError());
  }
}

static void run_parquet_scan_test(std::string const& table_name,
                                  size_t num_rows,
                                  int num_threads,
                                  size_t batch_size,
                                  size_t row_group_size                        = 0,
                                  duckdb::vector<duckdb::idx_t> projection_ids = {},
                                  batch_validator_t validator   = validate_scanned_batches,
                                  table_creator_t table_creator = create_synthetic_table)
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();

  table_creator(con, table_name, num_rows);
  auto parquet_path = write_parquet_from_table(con, table_name, row_group_size);

  auto& client_ctx = *con.context;
  auto sirius_ctx  = sirius::get_sirius_context(con, get_test_config_path());
  auto& mem_mgr    = sirius_ctx->get_memory_manager();
  auto* mem_space  = get_space(mem_mgr, Tier::HOST);
  REQUIRE(mem_space != nullptr);

  // Begin transaction for catalog access.
  auto begin_result = con.Query("BEGIN TRANSACTION");
  REQUIRE(begin_result);
  REQUIRE(!begin_result->HasError());

  auto physical_scan =
    make_parquet_scan(client_ctx, parquet_path.string(), std::move(projection_ids));
  REQUIRE(physical_scan);

  auto global_state = std::make_shared<op::scan::parquet_scan_task_global_state>(
    nullptr, physical_scan.get(), batch_size);

  cucascade::shared_data_repository data_repo;

  sirius::parallel::task_executor_config executor_config{num_threads, false};
  auto task_queue =
    std::make_unique<sirius::op::scan::duckdb_scan_task_queue>(executor_config.num_threads);
  sirius::parallel::itask_executor executor(std::move(task_queue), std::move(executor_config));

  auto run_scan = [&]() -> std::vector<std::shared_ptr<cucascade::data_batch>> {
    executor.start();
    uint64_t task_id = 1;
    size_t scheduled = 0;
    while (true) {
      auto const partition_idx = global_state->get_next_rg_partition_idx();
      if (!partition_idx.has_value()) { break; }
      auto local_state =
        std::make_unique<op::scan::parquet_scan_task_local_state>(*global_state, *partition_idx);
      auto reservation = mem_mgr.request_reservation(
        cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::HOST},
        local_state->get_reserved_compressed_bytes());
      local_state->set_reservation(std::move(reservation));
      auto task = std::make_unique<op::scan::parquet_scan_task>(
        task_id++, &data_repo, std::move(local_state), global_state);
      executor.schedule(std::move(task));
      ++scheduled;
    }
    while (data_repo.total_size() < scheduled) {
      std::this_thread::yield();
    }

    executor.stop();
    auto batches = drain_data_repo(data_repo);
    REQUIRE(batches.size() == scheduled);
    return batches;
  };

  // The stream must be declared before batches so it outlives GPU data allocated on it.
  rmm::cuda_stream stream;
  auto batches = run_scan();
  validator(batches, num_rows, mem_mgr, stream);

  // End the transaction.
  auto commit_result = con.Query("COMMIT");
  REQUIRE(commit_result);
  REQUIRE(!commit_result->HasError());

  auto drop_result = con.Query("DROP TABLE " + table_name);
  REQUIRE(drop_result);
  REQUIRE(!drop_result->HasError());
  std::filesystem::remove(parquet_path);
}

static void run_multi_file_parquet_scan_test(std::string const& table_prefix,
                                             std::vector<size_t> const& file_row_counts,
                                             int num_threads,
                                             size_t batch_size,
                                             size_t row_group_size                        = 0,
                                             duckdb::vector<duckdb::idx_t> projection_ids = {},
                                             batch_validator_t validator = validate_scanned_batches)
{
  REQUIRE(!file_row_counts.empty());

  auto [db_owner, con] = sirius::make_test_db_and_connection();

  auto parquet_dir = std::filesystem::temp_directory_path() / (table_prefix + "_multi_file");
  std::filesystem::remove_all(parquet_dir);
  std::filesystem::create_directories(parquet_dir);

  std::vector<std::string> table_names;
  table_names.reserve(file_row_counts.size());
  size_t next_id    = 0;
  size_t total_rows = 0;
  for (size_t file_idx = 0; file_idx < file_row_counts.size(); ++file_idx) {
    auto const table_name = table_prefix + "_part_" + std::to_string(file_idx);
    auto const row_count  = file_row_counts[file_idx];
    create_synthetic_table_with_offset(con, table_name, row_count, next_id);
    write_parquet_from_table_to_path(
      con, table_name, parquet_dir / (table_name + ".parquet"), row_group_size);
    table_names.push_back(table_name);
    next_id += row_count;
    total_rows += row_count;
  }

  auto& client_ctx = *con.context;
  auto sirius_ctx  = sirius::get_sirius_context(con, get_test_config_path());
  auto& mem_mgr    = sirius_ctx->get_memory_manager();
  auto* mem_space  = get_space(mem_mgr, Tier::HOST);
  REQUIRE(mem_space != nullptr);

  auto begin_result = con.Query("BEGIN TRANSACTION");
  REQUIRE(begin_result);
  REQUIRE(!begin_result->HasError());

  auto physical_scan =
    make_parquet_scan(client_ctx, (parquet_dir / "*.parquet").string(), std::move(projection_ids));
  REQUIRE(physical_scan);

  auto global_state = std::make_shared<op::scan::parquet_scan_task_global_state>(
    nullptr, physical_scan.get(), batch_size);

  cucascade::shared_data_repository data_repo;

  sirius::parallel::task_executor_config executor_config{num_threads, false};
  auto task_queue =
    std::make_unique<sirius::op::scan::duckdb_scan_task_queue>(executor_config.num_threads);
  sirius::parallel::itask_executor executor(std::move(task_queue), std::move(executor_config));

  auto run_scan = [&]() -> std::vector<std::shared_ptr<cucascade::data_batch>> {
    executor.start();
    uint64_t task_id = 1;
    size_t scheduled = 0;
    while (true) {
      auto const partition_idx = global_state->get_next_rg_partition_idx();
      if (!partition_idx.has_value()) { break; }
      auto local_state =
        std::make_unique<op::scan::parquet_scan_task_local_state>(*global_state, *partition_idx);
      auto reservation = mem_mgr.request_reservation(
        cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::HOST},
        local_state->get_reserved_compressed_bytes());
      local_state->set_reservation(std::move(reservation));
      auto task = std::make_unique<op::scan::parquet_scan_task>(
        task_id++, &data_repo, std::move(local_state), global_state);
      executor.schedule(std::move(task));
      ++scheduled;
    }
    while (data_repo.total_size() < scheduled) {
      std::this_thread::yield();
    }

    executor.stop();
    auto batches = drain_data_repo(data_repo);
    REQUIRE(batches.size() == scheduled);
    return batches;
  };

  // The stream must be declared before batches so it outlives GPU data allocated on it.
  rmm::cuda_stream stream;
  auto batches = run_scan();
  validator(batches, total_rows, mem_mgr, stream);

  auto commit_result = con.Query("COMMIT");
  REQUIRE(commit_result);
  REQUIRE(!commit_result->HasError());

  for (auto const& table_name : table_names) {
    auto drop_result = con.Query("DROP TABLE " + table_name);
    REQUIRE(drop_result);
    REQUIRE(!drop_result->HasError());
  }
  std::filesystem::remove_all(parquet_dir);
}

TEST_CASE("parquet_scan_task - single threaded small table",
          "[parquet_scan_task][single_thread][shared_context]")
{
  run_parquet_scan_test("parquet_small", 2000, 1, 200000, 500);
}

TEST_CASE("parquet_scan_task - single threaded small batches",
          "[parquet_scan_task][single_thread][shared_context]")
{
  run_parquet_scan_test("parquet_medium", 10000, 1, 150000, 500);
}

TEST_CASE("parquet_scan_task - multi threaded medium table",
          "[parquet_scan_task][multi_thread][shared_context]")
{
  run_parquet_scan_test("parquet_mt", 100000, 4, 1000000, 0);
}

TEST_CASE("parquet_scan_task - multi threaded large table",
          "[parquet_scan_task][multi_thread][shared_context]")
{
  run_parquet_scan_test("parquet_mt_large", 500000, 8, 10000000, 0);
}

TEST_CASE("parquet_scan_task - single partition row group",
          "[parquet_scan_task][edge_case][shared_context]")
{
  run_parquet_scan_test("parquet_single_partition", 5000, 2, 5000000, 100000);
}

TEST_CASE("parquet_scan_task - projected subset", "[parquet_scan_task][projection][shared_context]")
{
  duckdb::vector<duckdb::idx_t> projection_ids{0, 2};  // id, price
  run_parquet_scan_test("parquet_projected",
                        8000,
                        2,
                        200000,
                        500,
                        std::move(projection_ids),
                        validate_projected_id_price_batches);
}

TEST_CASE("parquet_scan_task - projected flat columns with nested schema",
          "[parquet_scan_task][projection][nested_schema][shared_context]")
{
  duckdb::vector<duckdb::idx_t> projection_ids{0, 2};  // id, price
  run_parquet_scan_test("parquet_projected_nested_schema",
                        8000,
                        2,
                        200000,
                        500,
                        std::move(projection_ids),
                        validate_projected_id_price_batches,
                        create_synthetic_table_with_nested_list);
}

TEST_CASE("parquet_scan_task - empty table", "[parquet_scan_task][edge_case][shared_context]")
{
  run_parquet_scan_test("parquet_empty", 0, 1, 200000, 500);
}

TEST_CASE("parquet_scan_task - single row table", "[parquet_scan_task][edge_case][shared_context]")
{
  // Suppress cudf warnings about single-row parquet files
  run_parquet_scan_test("parquet_single_row",
                        1,
                        1,
                        200000,
                        500,
                        duckdb::vector<duckdb::idx_t>{},
                        validate_scanned_batches_suppress_cudf);
}

TEST_CASE("parquet_scan_task - multi file full scan",
          "[parquet_scan_task][multi_file][shared_context]")
{
  run_multi_file_parquet_scan_test("parquet_multi_file", {3000, 4200}, 4, 200000, 500);
}

TEST_CASE("parquet_scan_task - multi file projected subset",
          "[parquet_scan_task][multi_file][projection][shared_context]")
{
  duckdb::vector<duckdb::idx_t> projection_ids{0, 2};  // id, price
  run_multi_file_parquet_scan_test("parquet_multi_file_projected",
                                   {2500, 3500, 1800},
                                   4,
                                   200000,
                                   500,
                                   std::move(projection_ids),
                                   validate_projected_id_price_batches);
}

TEST_CASE("parquet_scan_task - multi file full scan five files mixed sizes",
          "[parquet_scan_task][multi_file][shared_context]")
{
  run_multi_file_parquet_scan_test(
    "parquet_multi_file_five", {1400, 2600, 0, 3100, 900}, 6, 150000, 300);
}
