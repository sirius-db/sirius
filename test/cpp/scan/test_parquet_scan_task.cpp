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

// cucascade
#include <cucascade/memory/memory_reservation_manager.hpp>

// rmm
#include <rmm/cuda_stream.hpp>

// cudf
#include <cudf/logger.hpp>

#include <rapids_logger/logger.hpp>

// duckdb
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/parser/expression/constant_expression.hpp>
#include <duckdb/parser/expression/function_expression.hpp>
#include <duckdb/parser/tableref/table_function_ref.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>

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
  duckdb::vector<duckdb::idx_t> projection_ids             = {},
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters = nullptr)
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
                                                                    std::move(table_filters),
                                                                    0,
                                                                    std::move(extra_info),
                                                                    duckdb::vector<duckdb::Value>(),
                                                                    std::move(virtual_columns));
}

static duckdb::unique_ptr<duckdb::TableFilterSet> make_id_constant_filter(
  duckdb::ExpressionType comparison, int32_t constant)
{
  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  auto filter =
    duckdb::make_uniq<duckdb::ConstantFilter>(comparison, duckdb::Value::INTEGER(constant));
  table_filters->PushFilter(duckdb::ColumnIndex(0), std::move(filter));  // id column
  return table_filters;
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
      auto partition = global_state->claim_next_rg_partition();
      if (!partition.has_value()) { break; }
      auto local_state =
        std::make_unique<op::scan::parquet_scan_task_local_state>(std::move(*partition));
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
      auto const partition = global_state->claim_next_rg_partition();
      if (!partition.has_value()) { break; }
      auto local_state =
        std::make_unique<op::scan::parquet_scan_task_local_state>(std::move(*partition));
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

static void run_parquet_scan_test_with_filter(
  std::string const& table_name,
  size_t num_rows,
  size_t expected_rows,
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters,
  int num_threads,
  size_t batch_size,
  size_t row_group_size                        = 0,
  duckdb::vector<duckdb::idx_t> projection_ids = {},
  batch_validator_t validator                  = validate_scanned_batches)
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  create_synthetic_table(con, table_name, num_rows);
  auto parquet_path = write_parquet_from_table(con, table_name, row_group_size);

  auto& client_ctx = *con.context;
  auto sirius_ctx  = sirius::get_sirius_context(con, get_test_config_path());
  auto& mem_mgr    = sirius_ctx->get_memory_manager();
  auto* mem_space  = get_space(mem_mgr, Tier::HOST);
  REQUIRE(mem_space != nullptr);

  auto begin_result = con.Query("BEGIN TRANSACTION");
  REQUIRE(begin_result);
  REQUIRE(!begin_result->HasError());

  auto physical_scan = make_parquet_scan(
    client_ctx, parquet_path.string(), std::move(projection_ids), std::move(table_filters));
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
      auto partition = global_state->claim_next_rg_partition();
      if (!partition.has_value()) { break; }
      auto local_state =
        std::make_unique<op::scan::parquet_scan_task_local_state>(std::move(*partition));
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

  auto batches = run_scan();
  validator(batches, expected_rows, mem_mgr, rmm::cuda_stream_default);

  auto commit_result = con.Query("COMMIT");
  REQUIRE(commit_result);
  REQUIRE(!commit_result->HasError());

  auto drop_result = con.Query("DROP TABLE " + table_name);
  REQUIRE(drop_result);
  REQUIRE(!drop_result->HasError());
  std::filesystem::remove(parquet_path);
}

//------------------------------------------------------------------------------//
// Test cases
//------------------------------------------------------------------------------//

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

TEST_CASE("parquet_scan_task - filter prunes all rows", "[parquet_scan_task][filter]")
{
  auto table_filters = make_id_constant_filter(duckdb::ExpressionType::COMPARE_LESSTHAN, 0);
  run_parquet_scan_test_with_filter(
    "parquet_filter_none", 10000, 0, std::move(table_filters), 2, 200000, 500);
}

TEST_CASE("parquet_scan_task - filter keeps prefix rows", "[parquet_scan_task][filter]")
{
  constexpr int32_t threshold = 1234;
  auto table_filters = make_id_constant_filter(duckdb::ExpressionType::COMPARE_LESSTHAN, threshold);
  run_parquet_scan_test_with_filter("parquet_filter_prefix",
                                    8000,
                                    static_cast<size_t>(threshold),
                                    std::move(table_filters),
                                    2,
                                    200000,
                                    500);
}

TEST_CASE("parquet_scan_task - filter actually prunes row groups",
          "[parquet_scan_task][filter][row_group_pruning]")
{
  // With 100000 rows and ROW_GROUP_SIZE=5000, we get exactly 20 row groups.
  // ids are sequential [0..N), so a filter id < 5000 matches only the first row group.

  constexpr size_t num_rows                   = 100000;
  constexpr size_t row_group_size             = 5000;
  constexpr int32_t threshold                 = 5000;
  constexpr size_t expected_pruned_partitions = 1;  // only the first row group survives

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  create_synthetic_table(con, "parquet_rg_prune", num_rows);
  auto parquet_path = write_parquet_from_table(con, "parquet_rg_prune", row_group_size);

  auto& client_ctx = *con.context;
  auto sirius_ctx  = sirius::get_sirius_context(con, get_test_config_path());
  auto& mem_mgr    = sirius_ctx->get_memory_manager();

  auto begin_result = con.Query("BEGIN TRANSACTION");
  REQUIRE(begin_result);
  REQUIRE(!begin_result->HasError());

  // Build scan WITHOUT filter to get the unfiltered partition count.
  // Use batch_size=1 so each row group becomes its own partition.
  auto physical_scan_no_filter = make_parquet_scan(client_ctx, parquet_path.string());
  REQUIRE(physical_scan_no_filter);
  auto global_state_no_filter = std::make_shared<op::scan::parquet_scan_task_global_state>(
    nullptr, physical_scan_no_filter.get(), 1);
  auto const total_partitions = global_state_no_filter->get_num_row_group_partitions();
  REQUIRE(total_partitions > 1);  // Must have multiple row groups for pruning to be meaningful

  // Build scan WITH filter id < threshold (should match only the first row group)
  auto table_filters = make_id_constant_filter(duckdb::ExpressionType::COMPARE_LESSTHAN, threshold);
  auto physical_scan_filtered =
    make_parquet_scan(client_ctx, parquet_path.string(), {}, std::move(table_filters));
  REQUIRE(physical_scan_filtered);
  auto global_state_filtered = std::make_shared<op::scan::parquet_scan_task_global_state>(
    nullptr, physical_scan_filtered.get(), 1);
  auto const pruned_partitions = global_state_filtered->get_num_row_group_partitions();

  REQUIRE(pruned_partitions == expected_pruned_partitions);

  auto commit_result = con.Query("COMMIT");
  REQUIRE(commit_result);
  REQUIRE(!commit_result->HasError());

  auto drop_result = con.Query("DROP TABLE parquet_rg_prune");
  REQUIRE(drop_result);
  REQUIRE(!drop_result->HasError());
  std::filesystem::remove(parquet_path);
}

TEST_CASE("parquet_scan_task - filter prunes row groups with decimal comparison",
          "[parquet_scan_task][filter][row_group_pruning]")
{
  // Table has a DECIMAL(10,2) column whose values grow with id, so row-group
  // statistics let us prune higher-valued row groups with a less-than filter.
  // 100000 rows / 5000 per row group = 20 row groups.
  // amount = id * 1.25; filter amount < 6250.00 ⇔ id < 5000 → 1 row group.
  constexpr size_t num_rows                   = 100000;
  constexpr size_t row_group_size             = 5000;
  constexpr size_t expected_pruned_partitions = 1;

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  // Create a table with a decimal column: amount = id * 1.25, stored as DECIMAL(10,2)
  std::string const table_name = "parquet_rg_prune_decimal";
  {
    auto result = con.Query("CREATE TABLE " + table_name + " (id INTEGER, amount DECIMAL(10,2))");
    REQUIRE(result);
    REQUIRE(!result->HasError());

    constexpr size_t BATCH_SIZE = 1000;
    for (size_t start = 0; start < num_rows; start += BATCH_SIZE) {
      size_t end             = std::min(start + BATCH_SIZE, num_rows);
      std::string insert_sql = "INSERT INTO " + table_name +
                               " SELECT i, "
                               "CAST(i * 1.25 AS DECIMAL(10,2)) "
                               "FROM generate_series(" +
                               std::to_string(start) + ", " + std::to_string(end - 1) + ") t(i)";
      auto result2 = con.Query(insert_sql);
      REQUIRE(result2);
      REQUIRE(!result2->HasError());
    }
  }

  auto parquet_path = write_parquet_from_table(con, table_name, row_group_size);

  auto& client_ctx = *con.context;
  auto sirius_ctx  = sirius::get_sirius_context(con, get_test_config_path());

  auto begin_result = con.Query("BEGIN TRANSACTION");
  REQUIRE(begin_result);
  REQUIRE(!begin_result->HasError());

  // Unfiltered partition count (batch_size=1 so each row group is its own partition)
  auto physical_scan_no_filter = make_parquet_scan(client_ctx, parquet_path.string());
  REQUIRE(physical_scan_no_filter);
  auto global_state_no_filter = std::make_shared<op::scan::parquet_scan_task_global_state>(
    nullptr, physical_scan_no_filter.get(), 1);
  auto const total_partitions = global_state_no_filter->get_num_row_group_partitions();
  REQUIRE(total_partitions > 1);

  // Filter: amount < 6250.00  (corresponds to id < 5000, i.e. first row group)
  {
    auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    auto filter        = duckdb::make_uniq<duckdb::ConstantFilter>(
      duckdb::ExpressionType::COMPARE_LESSTHAN,
      duckdb::Value::DECIMAL(static_cast<int64_t>(625000), 10, 2));  // 6250.00
    table_filters->PushFilter(duckdb::ColumnIndex(1), std::move(filter));   // amount column

    auto physical_scan_filtered =
      make_parquet_scan(client_ctx, parquet_path.string(), {}, std::move(table_filters));
    REQUIRE(physical_scan_filtered);
    auto global_state_filtered = std::make_shared<op::scan::parquet_scan_task_global_state>(
      nullptr, physical_scan_filtered.get(), 1);
    auto const pruned_partitions = global_state_filtered->get_num_row_group_partitions();

    REQUIRE(pruned_partitions == expected_pruned_partitions);
  }

  auto commit_result = con.Query("COMMIT");
  REQUIRE(commit_result);
  REQUIRE(!commit_result->HasError());

  auto drop_result2 = con.Query("DROP TABLE " + table_name);
  REQUIRE(drop_result2);
  REQUIRE(!drop_result2->HasError());
  std::filesystem::remove(parquet_path);
}

TEST_CASE("parquet_scan_task - filter prunes row groups with date comparison",
          "[parquet_scan_task][filter][row_group_pruning]")
{
  // Table has a DATE column whose values grow with id, so row-group statistics
  // allow pruning with a less-than filter on the date column.
  // 100000 rows / 5000 per row group = 20 row groups.
  // dt = '2020-01-01' + id days; filter dt < '2020-01-15' → only the first row group.
  constexpr size_t num_rows                   = 100000;
  constexpr size_t row_group_size             = 5000;
  constexpr size_t expected_pruned_partitions = 1;

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  // Create a table with a date column: dt = DATE '2020-01-01' + INTERVAL (id) DAY
  std::string const table_name = "parquet_rg_prune_date";
  {
    auto result = con.Query("CREATE TABLE " + table_name + " (id INTEGER, dt DATE)");
    REQUIRE(result);
    REQUIRE(!result->HasError());

    constexpr size_t BATCH_SIZE = 1000;
    for (size_t start = 0; start < num_rows; start += BATCH_SIZE) {
      size_t end             = std::min(start + BATCH_SIZE, num_rows);
      std::string insert_sql = "INSERT INTO " + table_name +
                               " SELECT i, "
                               "DATE '2020-01-01' + INTERVAL (i) DAY "
                               "FROM generate_series(" +
                               std::to_string(start) + ", " + std::to_string(end - 1) + ") t(i)";
      auto result2 = con.Query(insert_sql);
      REQUIRE(result2);
      REQUIRE(!result2->HasError());
    }
  }

  auto parquet_path = write_parquet_from_table(con, table_name, row_group_size);

  auto& client_ctx = *con.context;
  auto sirius_ctx  = sirius::get_sirius_context(con, get_test_config_path());

  auto begin_result = con.Query("BEGIN TRANSACTION");
  REQUIRE(begin_result);
  REQUIRE(!begin_result->HasError());

  // Unfiltered partition count
  auto physical_scan_no_filter = make_parquet_scan(client_ctx, parquet_path.string());
  REQUIRE(physical_scan_no_filter);
  auto global_state_no_filter = std::make_shared<op::scan::parquet_scan_task_global_state>(
    nullptr, physical_scan_no_filter.get(), 1);
  auto const total_partitions = global_state_no_filter->get_num_row_group_partitions();
  REQUIRE(total_partitions > 1);

  // Filter: dt < DATE '2020-01-15'  (only the first ~14 days, within the first row group)
  {
    auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    auto filter        = duckdb::make_uniq<duckdb::ConstantFilter>(
      duckdb::ExpressionType::COMPARE_LESSTHAN, duckdb::Value::DATE(2020, 1, 15));
    table_filters->PushFilter(duckdb::ColumnIndex(1), std::move(filter));  // dt column

    auto physical_scan_filtered =
      make_parquet_scan(client_ctx, parquet_path.string(), {}, std::move(table_filters));
    REQUIRE(physical_scan_filtered);
    auto global_state_filtered = std::make_shared<op::scan::parquet_scan_task_global_state>(
      nullptr, physical_scan_filtered.get(), 1);
    auto const pruned_partitions = global_state_filtered->get_num_row_group_partitions();

    REQUIRE(pruned_partitions == expected_pruned_partitions);
  }

  auto commit_result = con.Query("COMMIT");
  REQUIRE(commit_result);
  REQUIRE(!commit_result->HasError());

  auto drop_result2 = con.Query("DROP TABLE " + table_name);
  REQUIRE(drop_result2);
  REQUIRE(!drop_result2->HasError());
  std::filesystem::remove(parquet_path);
}

TEST_CASE("parquet_scan_task - filter on non-projected column",
          "[parquet_scan_task][filter][projection]")
{
  // Project only columns {0, 2} (id, price) but filter on column 1 (value).
  // value = id * 100, so value < 50000 ⇔ id < 500 → 500 rows expected.
  constexpr int64_t val_threshold = 50000;
  auto table_filters              = duckdb::make_uniq<duckdb::TableFilterSet>();
  auto filter = duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                                          duckdb::Value::BIGINT(val_threshold));
  table_filters->PushFilter(duckdb::ColumnIndex(1), std::move(filter));  // value column

  duckdb::vector<duckdb::idx_t> projection_ids{0, 2};  // id, price
  run_parquet_scan_test_with_filter("parquet_filter_non_proj",
                                    8000,
                                    500,
                                    std::move(table_filters),
                                    2,
                                    200000,
                                    500,
                                    std::move(projection_ids),
                                    validate_projected_id_price_batches);
}

TEST_CASE("parquet_scan_task - filter prunes row groups with multi-column comparison",
          "[parquet_scan_task][filter][row_group_pruning]")
{
  // Uses the standard synthetic table (id INTEGER, value BIGINT, price DOUBLE, name VARCHAR).
  // Applies two column filters simultaneously:
  //   id    < 5000   (column 0)
  //   value < 500000 (column 1, value = id * 100, so value < 500000 ⇔ id < 5000)
  // Both filters target the first row group only.
  // 100000 rows / 5000 per row group = 20 row groups; exactly 1 survives.

  constexpr size_t num_rows                   = 100000;
  constexpr size_t row_group_size             = 5000;
  constexpr int32_t id_threshold              = 5000;
  constexpr int64_t val_threshold             = 500000;  // = id_threshold * 100
  constexpr size_t expected_pruned_partitions = 1;

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  std::string const table_name = "parquet_rg_prune_multi";
  create_synthetic_table(con, table_name, num_rows);
  auto parquet_path = write_parquet_from_table(con, table_name, row_group_size);

  auto& client_ctx = *con.context;
  auto sirius_ctx  = sirius::get_sirius_context(con, get_test_config_path());

  auto begin_result = con.Query("BEGIN TRANSACTION");
  REQUIRE(begin_result);
  REQUIRE(!begin_result->HasError());

  // Unfiltered partition count
  auto physical_scan_no_filter = make_parquet_scan(client_ctx, parquet_path.string());
  REQUIRE(physical_scan_no_filter);
  auto global_state_no_filter = std::make_shared<op::scan::parquet_scan_task_global_state>(
    nullptr, physical_scan_no_filter.get(), 1);
  auto const total_partitions = global_state_no_filter->get_num_row_group_partitions();
  REQUIRE(total_partitions > 1);

  // Multi-column filter: id < 5000 AND value < 500000
  {
    auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();

    auto id_filter = duckdb::make_uniq<duckdb::ConstantFilter>(
      duckdb::ExpressionType::COMPARE_LESSTHAN, duckdb::Value::INTEGER(id_threshold));
    table_filters->PushFilter(duckdb::ColumnIndex(0), std::move(id_filter));  // id column

    auto val_filter = duckdb::make_uniq<duckdb::ConstantFilter>(
      duckdb::ExpressionType::COMPARE_LESSTHAN, duckdb::Value::BIGINT(val_threshold));
    table_filters->PushFilter(duckdb::ColumnIndex(1), std::move(val_filter));  // value column

    auto physical_scan_filtered =
      make_parquet_scan(client_ctx, parquet_path.string(), {}, std::move(table_filters));
    REQUIRE(physical_scan_filtered);
    auto global_state_filtered = std::make_shared<op::scan::parquet_scan_task_global_state>(
      nullptr, physical_scan_filtered.get(), 1);
    auto const pruned_partitions = global_state_filtered->get_num_row_group_partitions();

    REQUIRE(pruned_partitions == expected_pruned_partitions);
  }

  auto commit_result = con.Query("COMMIT");
  REQUIRE(commit_result);
  REQUIRE(!commit_result->HasError());

  auto drop_result = con.Query("DROP TABLE " + table_name);
  REQUIRE(drop_result);
  REQUIRE(!drop_result->HasError());
  std::filesystem::remove(parquet_path);
}