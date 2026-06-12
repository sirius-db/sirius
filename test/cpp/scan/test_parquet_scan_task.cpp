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
#include <utils/telemetry_utils.hpp>
#include <utils/utils.hpp>

// sirius
#include <exec/config.hpp>
#include <exec/thread_pool.hpp>
#include <helper/logical_type.hpp>
#include <helper/type_conversions.hpp>
#include <op/scan/parquet_scan_task.hpp>
#include <op/sirius_physical_parquet_scan.hpp>
#include <parallel/task_executor.hpp>
#include <scan_manager/split_connector.hpp>

// Phase 19 IO-15: include test_helpers_ioctx.hpp LAST among sirius/test
// headers — it transitively pulls liburing.h via uring_ioctx.hpp ->
// uring_reactor.hpp, and liburing.h defines a BLOCK_SIZE macro that collides
// with the BLOCK_SIZE static member in <blockingconcurrentqueue.h> (transitively
// pulled via utils/utils.hpp -> sirius_context.hpp -> task_creator.hpp ->
// bounded_thread_pool.hpp). All consumers of blockingconcurrentqueue.h must
// precede this include. Mirrors src/op/scan/parquet_scan_task.cpp:30-35.
#include <scan/test_helpers_ioctx.hpp>

// cucascade
#include <cucascade/memory/memory_reservation_manager.hpp>

// sirius IO framework (Phase 19 IO-15 helper preparation)
#include <io/types.hpp>
#include <io/uring/uring_ioctx.hpp>

// rmm
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

// cudf
#include <cudf/logger.hpp>

#include <rapids_logger/logger.hpp>

// duckdb
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/parser/expression/constant_expression.hpp>
#include <duckdb/parser/expression/function_expression.hpp>
#include <duckdb/parser/tableref/table_function_ref.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>

// standard library
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

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

/**
 * Minimal concrete executor for scan task tests.
 */
class scan_test_executor : public sirius::parallel::itask_executor {
 public:
  explicit scan_test_executor(sirius::exec::thread_pool_config config)
    : itask_executor(std::move(config), sirius::test::make_test_telemetry_context())
  {
  }

 protected:
  void manager_loop() override
  {
    while (_running.load()) {
      auto slot = _bounded_pool->reserve();
      if (!slot) { break; }
      auto task = _task_queue.pop();
      if (!task) { break; }
      _bounded_pool->dispatch(std::move(slot), [t = std::move(task)]() mutable {
        t->execute(cudf::get_default_stream());
      });
    }
  }
};

// Phase 22.1 D-08: make_test_gpu_ioctxs() lifted into shared header
// scan/test_helpers_ioctx.hpp so that test_parquet_split_provider.cpp can
// reuse the same helper. Uses sirius::scan_test_utils::make_test_gpu_ioctxs.
using sirius::scan_test_utils::make_test_gpu_ioctxs;

static std::unique_ptr<sirius::op::sirius_physical_parquet_scan> make_parquet_scan(
  duckdb::ClientContext& ctx,
  std::string const& parquet_path,
  duckdb::vector<duckdb::idx_t> const& projection_indices  = {},
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

  // Unused stuff
  duckdb::virtual_column_map_t virtual_columns;
  if (auto* multi_bind = dynamic_cast<duckdb::MultiFileBindData*>(bind_data.get())) {
    virtual_columns = multi_bind->virtual_columns;
  }
  duckdb::ExtraOperatorInfo extra_info;

  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<duckdb::idx_t> projection_ids;
  duckdb::vector<duckdb::LogicalType> output_types;
  if (projection_indices.empty()) {
    for (size_t i = 0; i < return_types.size(); ++i) {
      column_ids.emplace_back(duckdb::ColumnIndex(i));
    }
    output_types = return_types;
  } else {
    // First, collect the set of ALL projected columns (including those needed for filter)
    std::unordered_set<size_t> projected_column_index_set(projection_indices.begin(),
                                                          projection_indices.end());
    std::vector<size_t> pure_filter_indices;
    if (table_filters) {
      for (auto const& entry : table_filters->filters) {
        if (!projected_column_index_set.contains(entry.first)) {
          pure_filter_indices.push_back(entry.first);
          projected_column_index_set.insert(entry.first);
        }
      }
    }
    std::vector<size_t> projected_column_indices(projected_column_index_set.begin(),
                                                 projected_column_index_set.end());

    // Sort the projected column indices before inserting into column_ids
    // The goal is to mirror the behavior of the duckdb planner
    std::sort(projected_column_indices.begin(), projected_column_indices.end());

    column_ids.reserve(return_types.size());
    projection_ids.reserve(projection_indices.size());
    for (auto const idx : projected_column_indices) {
      column_ids.push_back(duckdb::ColumnIndex(idx));
    }
    // Insert the leftover column indices
    for (size_t i = 0; i < return_types.size(); ++i) {
      if (!projected_column_index_set.contains(i)) { column_ids.push_back(duckdb::ColumnIndex(i)); }
    }

    // Projection ids are the positions of the projected columns within column_ids
    // Pure filter columns are placed AFTER output projection columns
    projection_ids.reserve(projection_indices.size() + pure_filter_indices.size());
    for (auto const idx : projection_indices) {
      auto it =
        std::find_if(column_ids.begin(), column_ids.end(), [idx](const duckdb::ColumnIndex& id) {
          return id.GetPrimaryIndex() == idx;
        });
      projection_ids.push_back(std::distance(column_ids.begin(), it));
    }
    for (auto const idx : pure_filter_indices) {
      auto it =
        std::find_if(column_ids.begin(), column_ids.end(), [idx](const duckdb::ColumnIndex& id) {
          return id.GetPrimaryIndex() == idx;
        });
      projection_ids.push_back(std::distance(column_ids.begin(), it));
    }

    output_types.reserve(projection_indices.size());
    for (auto const idx : projection_indices) {
      output_types.push_back(return_types[idx]);
    }
  }

  return std::make_unique<sirius::op::sirius_physical_parquet_scan>(
    sirius::from_duckdb_vec(output_types),
    table_function,
    std::move(bind_data),
    sirius::from_duckdb_vec(return_types),
    std::move(column_ids),
    std::move(projection_ids),
    std::move(names),
    std::move(table_filters),
    0,
    std::move(extra_info),
    duckdb::vector<duckdb::Value>(),
    std::move(virtual_columns),
    nullptr);
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
                                  size_t row_group_size                                   = 0,
                                  duckdb::vector<duckdb::idx_t> const& projection_indices = {},
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
    make_parquet_scan(client_ctx, parquet_path.string(), std::move(projection_indices));
  REQUIRE(physical_scan);

  auto telemetry_context = sirius::test::make_test_telemetry_context();
  auto global_state      = std::make_shared<op::scan::parquet_scan_task_global_state>(
    nullptr,
    physical_scan.get(),
    telemetry_context,
    batch_size,
    sirius::scan_test_utils::make_test_ioctx());

  cucascade::shared_data_repository data_repo;

  sirius::exec::thread_pool_config executor_config{num_threads, "scan_test"};
  scan_test_executor executor(executor_config);

  auto run_scan = [&]() -> std::vector<std::shared_ptr<cucascade::data_batch>> {
    executor.start();
    uint64_t task_id = 1;
    size_t scheduled = 0;
    while (true) {
      auto partition = global_state->claim_next_rg_partition();
      if (!partition.has_value()) { break; }
      auto local_state = std::make_unique<op::scan::parquet_scan_task_local_state>(
        *global_state, std::move(*partition));
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

static void run_multi_file_parquet_scan_test(
  std::string const& table_prefix,
  std::vector<size_t> const& file_row_counts,
  int num_threads,
  size_t batch_size,
  size_t row_group_size                                   = 0,
  duckdb::vector<duckdb::idx_t> const& projection_indices = {},
  batch_validator_t validator                             = validate_scanned_batches)
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

  auto physical_scan = make_parquet_scan(
    client_ctx, (parquet_dir / "*.parquet").string(), std::move(projection_indices));
  REQUIRE(physical_scan);

  auto telemetry_context = sirius::test::make_test_telemetry_context();
  auto global_state      = std::make_shared<op::scan::parquet_scan_task_global_state>(
    nullptr,
    physical_scan.get(),
    telemetry_context,
    batch_size,
    sirius::scan_test_utils::make_test_ioctx());

  cucascade::shared_data_repository data_repo;

  sirius::exec::thread_pool_config executor_config{num_threads, "scan_test"};
  scan_test_executor executor(executor_config);

  auto run_scan = [&]() -> std::vector<std::shared_ptr<cucascade::data_batch>> {
    executor.start();
    uint64_t task_id = 1;
    size_t scheduled = 0;
    while (true) {
      auto const partition = global_state->claim_next_rg_partition();
      if (!partition.has_value()) { break; }
      auto local_state = std::make_unique<op::scan::parquet_scan_task_local_state>(
        *global_state, std::move(*partition));
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
  size_t row_group_size                                   = 0,
  duckdb::vector<duckdb::idx_t> const& projection_indices = {},
  batch_validator_t validator                             = validate_scanned_batches)
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
    client_ctx, parquet_path.string(), std::move(projection_indices), std::move(table_filters));
  REQUIRE(physical_scan);

  auto telemetry_context = sirius::test::make_test_telemetry_context();
  auto global_state      = std::make_shared<op::scan::parquet_scan_task_global_state>(
    nullptr,
    physical_scan.get(),
    telemetry_context,
    batch_size,
    sirius::scan_test_utils::make_test_ioctx());

  cucascade::shared_data_repository data_repo;

  sirius::exec::thread_pool_config executor_config{num_threads, "scan_test"};
  scan_test_executor executor(executor_config);

  auto run_scan = [&]() -> std::vector<std::shared_ptr<cucascade::data_batch>> {
    executor.start();
    uint64_t task_id = 1;
    size_t scheduled = 0;
    while (true) {
      auto partition = global_state->claim_next_rg_partition();
      if (!partition.has_value()) { break; }
      auto local_state = std::make_unique<op::scan::parquet_scan_task_local_state>(
        *global_state, std::move(*partition));
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

  // HYG-02: use an explicit local stream instead of the default-stream sentinel.
  rmm::cuda_stream validator_stream;
  auto batches = run_scan();
  validator(batches, expected_rows, mem_mgr, validator_stream.view());

  auto commit_result = con.Query("COMMIT");
  REQUIRE(commit_result);
  REQUIRE(!commit_result->HasError());

  auto drop_result = con.Query("DROP TABLE " + table_name);
  REQUIRE(drop_result);
  REQUIRE(!drop_result->HasError());
  std::filesystem::remove(parquet_path);
}

static size_t count_row_group_partitions(
  duckdb::ClientContext& client_ctx,
  std::string const& parquet_path,
  size_t batch_size                                        = 1,
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters = nullptr,
  duckdb::vector<duckdb::idx_t> const& projection_indices  = {})
{
  auto physical_scan = make_parquet_scan(
    client_ctx, parquet_path, std::move(projection_indices), std::move(table_filters));
  REQUIRE(physical_scan);

  auto telemetry_context = sirius::test::make_test_telemetry_context();
  auto global_state      = std::make_shared<op::scan::parquet_scan_task_global_state>(
    nullptr,
    physical_scan.get(),
    telemetry_context,
    batch_size,
    sirius::scan_test_utils::make_test_ioctx());
  return global_state->get_num_row_group_partitions();
}

template <typename TableSetupFn, typename FilterFactoryFn>
static void run_row_group_pruning_test(std::string const& table_name,
                                       size_t row_group_size,
                                       size_t expected_pruned_partitions,
                                       TableSetupFn&& setup_table,
                                       FilterFactoryFn&& make_filters)
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  setup_table(con, table_name);
  auto parquet_path = write_parquet_from_table(con, table_name, row_group_size);

  auto& client_ctx = *con.context;
  auto sirius_ctx  = sirius::get_sirius_context(con, get_test_config_path());
  REQUIRE(sirius_ctx != nullptr);

  auto begin_result = con.Query("BEGIN TRANSACTION");
  REQUIRE(begin_result);
  REQUIRE(!begin_result->HasError());

  auto const total_partitions = count_row_group_partitions(client_ctx, parquet_path.string());
  REQUIRE(total_partitions > 1);

  auto const pruned_partitions =
    count_row_group_partitions(client_ctx, parquet_path.string(), 1, make_filters());
  REQUIRE(pruned_partitions == expected_pruned_partitions);

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
  duckdb::vector<duckdb::idx_t> projection_indices{0, 2};  // id, price
  run_parquet_scan_test("parquet_projected",
                        8000,
                        2,
                        200000,
                        500,
                        std::move(projection_indices),
                        validate_projected_id_price_batches);
}

TEST_CASE("parquet_scan_task - projected flat columns with nested schema",
          "[parquet_scan_task][projection][nested_schema][shared_context]")
{
  duckdb::vector<duckdb::idx_t> projection_indices{0, 2};  // id, price
  run_parquet_scan_test("parquet_projected_nested_schema",
                        8000,
                        2,
                        200000,
                        500,
                        std::move(projection_indices),
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
  duckdb::vector<duckdb::idx_t> projection_indices{0, 2};  // id, price
  run_multi_file_parquet_scan_test("parquet_multi_file_projected",
                                   {2500, 3500, 1800},
                                   4,
                                   200000,
                                   500,
                                   std::move(projection_indices),
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

  run_row_group_pruning_test(
    "parquet_rg_prune",
    row_group_size,
    expected_pruned_partitions,
    [num_rows](duckdb::Connection& con, std::string const& table_name) {
      create_synthetic_table(con, table_name, num_rows);
    },
    [threshold]() {
      return make_id_constant_filter(duckdb::ExpressionType::COMPARE_LESSTHAN, threshold);
    });
}

TEST_CASE("parquet_scan_task - decimal filter prunes row groups",
          "[parquet_scan_task][filter][row_group_pruning]")
{
  // Decimal column-vs-literal comparisons now translate to the cuDF AST and
  // push down to the parquet reader (rapidsai/cudf#21447 + #21681). The table
  // has amount = id * 1.25, so filter `amount < 6250.00` matches id < 5000 and
  // should leave only the first row group.
  constexpr size_t num_rows                   = 100000;
  constexpr size_t row_group_size             = 5000;
  constexpr size_t expected_pruned_partitions = 1;
  std::string const table_name                = "parquet_rg_prune_decimal";

  run_row_group_pruning_test(
    table_name,
    row_group_size,
    expected_pruned_partitions,
    [num_rows](duckdb::Connection& con, std::string const& name) {
      auto result = con.Query("CREATE TABLE " + name + " (id INTEGER, amount DECIMAL(10,2))");
      REQUIRE(result);
      REQUIRE(!result->HasError());

      constexpr size_t BATCH_SIZE = 1000;
      for (size_t start = 0; start < num_rows; start += BATCH_SIZE) {
        size_t end             = std::min(start + BATCH_SIZE, num_rows);
        std::string insert_sql = "INSERT INTO " + name +
                                 " SELECT i, "
                                 "CAST(i * 1.25 AS DECIMAL(10,2)) "
                                 "FROM generate_series(" +
                                 std::to_string(start) + ", " + std::to_string(end - 1) + ") t(i)";
        auto insert_result = con.Query(insert_sql);
        REQUIRE(insert_result);
        REQUIRE(!insert_result->HasError());
      }
    },
    []() {
      auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
      auto filter        = duckdb::make_uniq<duckdb::ConstantFilter>(
        duckdb::ExpressionType::COMPARE_LESSTHAN,
        duckdb::Value::DECIMAL(static_cast<int64_t>(625000), 10, 2));  // 6250.00
      table_filters->PushFilter(duckdb::ColumnIndex(1), std::move(filter));   // amount column
      return table_filters;
    });
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

  run_row_group_pruning_test(
    "parquet_rg_prune_date",
    row_group_size,
    expected_pruned_partitions,
    [num_rows](duckdb::Connection& con, std::string const& table_name) {
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
        auto insert_result = con.Query(insert_sql);
        REQUIRE(insert_result);
        REQUIRE(!insert_result->HasError());
      }
    },
    []() {
      auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
      auto filter        = duckdb::make_uniq<duckdb::ConstantFilter>(
        duckdb::ExpressionType::COMPARE_LESSTHAN, duckdb::Value::DATE(2020, 1, 15));
      table_filters->PushFilter(duckdb::ColumnIndex(1), std::move(filter));  // dt column
      return table_filters;
    });
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

  duckdb::vector<duckdb::idx_t> projection_indices{0, 2};  // id, price
  run_parquet_scan_test_with_filter("parquet_filter_non_proj",
                                    8000,
                                    500,
                                    std::move(table_filters),
                                    2,
                                    200000,
                                    500,
                                    std::move(projection_indices),
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

  run_row_group_pruning_test(
    "parquet_rg_prune_multi",
    row_group_size,
    expected_pruned_partitions,
    [num_rows](duckdb::Connection& con, std::string const& table_name) {
      create_synthetic_table(con, table_name, num_rows);
    },
    [id_threshold, val_threshold]() {
      auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();

      auto id_filter = duckdb::make_uniq<duckdb::ConstantFilter>(
        duckdb::ExpressionType::COMPARE_LESSTHAN, duckdb::Value::INTEGER(id_threshold));
      table_filters->PushFilter(duckdb::ColumnIndex(0), std::move(id_filter));  // id column

      auto val_filter = duckdb::make_uniq<duckdb::ConstantFilter>(
        duckdb::ExpressionType::COMPARE_LESSTHAN, duckdb::Value::BIGINT(val_threshold));
      table_filters->PushFilter(duckdb::ColumnIndex(1), std::move(val_filter));  // value column
      return table_filters;
    });
}
