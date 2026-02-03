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
#include <utils/utils.hpp>

// sirius
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <exec/config.hpp>
#include <op/scan/duckdb_scan_executor.hpp>
#include <op/scan/duckdb_scan_task.hpp>
#include <op/sirius_physical_table_scan.hpp>
#include <pipeline/pipeline_executor.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

// cudf
#include <cudf/strings/strings_column_view.hpp>

// duckdb
#include <duckdb.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/catalog/catalog_transaction.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/parallel/thread_context.hpp>

// standard library
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

using namespace sirius;
using namespace cucascade::memory;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static std::filesystem::path get_test_config_path()
{
  return std::filesystem::path(__FILE__).parent_path() / "memory.cfg";
}

static memory_space* get_space(cucascade::memory::memory_reservation_manager& mem_mgr, Tier tier)
{
  auto* space = mem_mgr.get_memory_space(tier, 0);
  if (space) { return space; }
  auto spaces = mem_mgr.get_memory_spaces_for_tier(tier);
  if (!spaces.empty()) { return const_cast<memory_space*>(spaces.front()); }
  return nullptr;
}

/**
 * @brief Create a simple synthetic table with multiple columns and rows
 */
static void create_synthetic_table(duckdb::Connection& con,
                                   std::string const& table_name,
                                   size_t num_rows)
{
  // Create table with INTEGER, BIGINT, DOUBLE, and VARCHAR columns
  // clang-format off
  std::string create_sql = "CREATE TABLE " + table_name + " \
                            ( \
                              id INTEGER, \
                              value BIGINT, \
                              price DOUBLE, \
                              name VARCHAR \
                            );";
  // clang-format on
  auto result = con.Query(create_sql);
  REQUIRE(result);
  REQUIRE(!result->HasError());

  // Insert data in batches
  constexpr size_t BATCH_SIZE = 1000;
  for (size_t start = 0; start < num_rows; start += BATCH_SIZE) {
    size_t end             = std::min(start + BATCH_SIZE, num_rows);
    std::string insert_sql = "INSERT INTO " + table_name + " VALUES ";

    for (size_t i = start; i < end; ++i) {
      if (i > start) { insert_sql += ", "; }
      // Generate predictable test data
      auto id          = static_cast<int32_t>(i);
      auto value       = static_cast<int64_t>(i * 100);
      auto price       = static_cast<double>(i) * 1.5;
      std::string name = "item_" + std::to_string(i);

      insert_sql += "(" + std::to_string(id) + ", " + std::to_string(value) + ", " +
                    std::to_string(price) + ", " + "'" + name + "')";
    }

    result = con.Query(insert_sql);
    REQUIRE(result);
    REQUIRE(!result->HasError());
  }
}

/**
 * @brief Drain all batches from a shared data repository.
 */
static std::vector<std::shared_ptr<cucascade::data_batch>> drain_data_repo(
  cucascade::shared_data_repository& data_repo)
{
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  while (true) {
    auto batch = data_repo.pop_data_batch(cucascade::batch_state::task_created);
    if (!batch) { break; }
    batches.push_back(std::move(batch));
  }
  return batches;
}

static std::vector<int64_t> copy_string_offsets(const cudf::column_view& offsets_col)
{
  auto num_offsets = offsets_col.size();
  std::vector<int64_t> offsets(num_offsets, 0);
  if (num_offsets == 0) { return offsets; }
  if (offsets_col.type().id() == cudf::type_id::INT64) {
    cudaMemcpy(offsets.data(),
               offsets_col.data<int64_t>(),
               sizeof(int64_t) * offsets.size(),
               cudaMemcpyDeviceToHost);
  } else if (offsets_col.type().id() == cudf::type_id::INT32) {
    std::vector<cudf::size_type> offsets32(num_offsets, 0);
    cudaMemcpy(offsets32.data(),
               offsets_col.data<cudf::size_type>(),
               sizeof(cudf::size_type) * offsets32.size(),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < offsets.size(); ++i) {
      offsets[i] = offsets32[i];
    }
  } else {
    FAIL("Unsupported offsets type in string column");
  }
  return offsets;
}

static void validate_scanned_batches(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& batches,
  size_t expected_rows,
  cucascade::memory::memory_reservation_manager& mem_mgr)
{
  auto* gpu_space = get_space(mem_mgr, Tier::GPU);
  REQUIRE(gpu_space != nullptr);
  auto& registry = sirius::converter_registry::get();

  if (expected_rows > 0) { REQUIRE_FALSE(batches.empty()); }

  std::vector<bool> seen(expected_rows, false);
  size_t total_rows = 0;

  for (auto const& batch : batches) {
    REQUIRE(batch != nullptr);
    batch->convert_to<cucascade::gpu_table_representation>(
      registry, gpu_space, cudf::get_default_stream());
    auto table_view = sirius::get_cudf_table_view(*batch);

    REQUIRE(table_view.num_columns() == 4);
    REQUIRE(table_view.column(0).type().id() == cudf::type_id::INT32);
    REQUIRE(table_view.column(1).type().id() == cudf::type_id::INT64);
    REQUIRE(table_view.column(2).type().id() == cudf::type_id::FLOAT64);
    REQUIRE(table_view.column(3).type().id() == cudf::type_id::STRING);

    auto const num_rows = table_view.num_rows();
    total_rows += num_rows;
    if (num_rows == 0) { continue; }

    std::vector<int32_t> ids(num_rows);
    std::vector<int64_t> values(num_rows);
    std::vector<double> prices(num_rows);
    cudaMemcpy(ids.data(),
               table_view.column(0).data<int32_t>(),
               sizeof(int32_t) * num_rows,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(values.data(),
               table_view.column(1).data<int64_t>(),
               sizeof(int64_t) * num_rows,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(prices.data(),
               table_view.column(2).data<double>(),
               sizeof(double) * num_rows,
               cudaMemcpyDeviceToHost);

    cudf::strings_column_view name_col(table_view.column(3));
    auto offsets = copy_string_offsets(name_col.offsets());
    REQUIRE(offsets.size() == static_cast<size_t>(num_rows + 1));
    std::vector<char> chars;
    if (!offsets.empty() && offsets.back() > 0) {
      chars.resize(static_cast<size_t>(offsets.back()));
      cudaMemcpy(chars.data(),
                 name_col.chars_begin(cudf::get_default_stream()),
                 chars.size(),
                 cudaMemcpyDeviceToHost);
    }

    for (cudf::size_type i = 0; i < num_rows; ++i) {
      auto id = ids[i];
      REQUIRE(id >= 0);
      REQUIRE(static_cast<size_t>(id) < expected_rows);
      REQUIRE_FALSE(seen[id]);
      seen[id] = true;

      auto const expected_value = static_cast<int64_t>(id) * 100;
      auto const expected_price = static_cast<double>(id) * 1.5;
      auto const expected_name  = "item_" + std::to_string(id);

      REQUIRE(values[i] == expected_value);
      REQUIRE(prices[i] == expected_price);

      auto const start = static_cast<size_t>(offsets[i]);
      auto const end   = static_cast<size_t>(offsets[i + 1]);
      std::string actual_name;
      if (end > start) { actual_name.assign(chars.data() + start, chars.data() + end); }
      REQUIRE(actual_name == expected_name);
    }
  }

  REQUIRE(total_rows == expected_rows);
  for (auto const& was_seen : seen) {
    REQUIRE(was_seen);
  }
}

/**
 * @brief Create a PhysicalTableScan for the given table
 */
static std::unique_ptr<sirius::op::sirius_physical_duckdb_scan> make_physical_table_scan(
  duckdb::ClientContext& ctx, std::string const& table_name)
{
  auto& catalog = duckdb::Catalog::GetCatalog(ctx, "");
  duckdb::CatalogTransaction txn(catalog, ctx);
  auto& schema = catalog.GetSchema(txn, "main");

  auto table_entry = schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, table_name);
  REQUIRE(table_entry);

  auto& table_catalog_entry = table_entry->Cast<duckdb::TableCatalogEntry>();

  // Get all column IDs as ColumnIndex
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<duckdb::idx_t> projection_ids;
  for (size_t i = 0; i < table_catalog_entry.GetColumns().LogicalColumnCount(); ++i) {
    column_ids.push_back(duckdb::ColumnIndex(i));
    projection_ids.push_back(i);  // Map output column i to internal column i
  }

  // Create bind data
  auto bind_data = std::make_unique<duckdb::TableScanBindData>(table_catalog_entry);

  // Get the table scan function
  auto table_scan_function = duckdb::TableScanFunction::GetFunction();

  // Get column names
  duckdb::vector<std::string> column_names;
  for (size_t i = 0; i < table_catalog_entry.GetColumns().LogicalColumnCount(); ++i) {
    column_names.push_back(table_catalog_entry.GetColumn(duckdb::LogicalIndex(i)).GetName());
  }

  // Create extra operator info (must be a variable, not a temporary)
  duckdb::ExtraOperatorInfo extra_info;

  auto virtual_columns = table_catalog_entry.GetVirtualColumns();

  // Create sirius_physical_duckdb_scan with all required parameters
  auto physical_scan = std::make_unique<sirius::op::sirius_physical_duckdb_scan>(
    table_catalog_entry.GetTypes(),   // types
    table_scan_function,              // function
    std::move(bind_data),             // bind_data
    table_catalog_entry.GetTypes(),   // returned_types
    std::move(column_ids),            // column_ids
    std::move(projection_ids),        // projection_ids (maps output to internal columns)
    std::move(column_names),          // names
    nullptr,                          // table_filters
    0,                                // estimated_cardinality
    std::move(extra_info),            // extra_info
    duckdb::vector<duckdb::Value>(),  // parameters
    std::move(virtual_columns)        // virtual_columns
  );

  return physical_scan;
}

/**
 * @brief Helper to run a scan test with the given parameters
 *
 * This encapsulates the common test setup and execution logic.
 */
static void run_scan_test(std::string const& table_name,
                          size_t num_rows,
                          int num_threads,
                          size_t batch_size)
{
  // Setup DuckDB database
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  // Create and populate table
  create_synthetic_table(con, table_name, num_rows);

  // Get client context
  auto& client_ctx = *con.context;
  auto sirius_ctx  = sirius::get_sirius_context(con, get_test_config_path());

  // Verify memory manager is initialized
  auto& mem_mgr   = sirius_ctx->get_memory_manager();
  auto* mem_space = get_space(mem_mgr, Tier::HOST);
  REQUIRE(mem_space != nullptr);

  // Begin transaction for catalog access
  auto begin_result = con.Query("BEGIN TRANSACTION");
  REQUIRE(begin_result);
  REQUIRE(!begin_result->HasError());

  // Create physical table scan
  auto physical_scan = make_physical_table_scan(client_ctx, table_name);
  REQUIRE(physical_scan);

  // Get pipeline executor from sirius context (owns the scan executor)
  auto& pipeline_executor = sirius_ctx->get_pipeline_executor();
  auto& scan_executor     = pipeline_executor.get_scan_executor();

  // Create execution context using dummy query
  auto dummy_query = "SELECT * FROM " + table_name + " LIMIT 0";
  auto prepared    = con.Prepare(dummy_query);
  REQUIRE(prepared);
  REQUIRE(!prepared->HasError());
  auto dummy_result = prepared->Execute();
  REQUIRE(dummy_result);
  REQUIRE(!dummy_result->HasError());

  // Create global state
  auto global_state = std::make_shared<op::scan::duckdb_scan_task_global_state>(
    nullptr, pipeline_executor, client_ctx, physical_scan.get());

  // Create data repository manager (empty, unused for this test)
  cucascade::shared_data_repository data_repo;

  // Create a single execution context for all scan tasks.
  auto thread_context = std::make_unique<duckdb::ThreadContext>(client_ctx);
  auto execution_context =
    std::make_unique<duckdb::ExecutionContext>(client_ctx, *thread_context, nullptr);

  // Run tasks
  const auto scan_start = std::chrono::steady_clock::now();
  pipeline_executor.start();
  for (int i = 0; i < scan_executor.get_num_threads(); ++i) {
    auto local_state = std::make_unique<op::scan::duckdb_scan_task_local_state>(
      *global_state, *execution_context, batch_size);
    auto task = std::make_unique<op::scan::duckdb_scan_task>(
      static_cast<uint64_t>(i + 1), &data_repo, std::move(local_state), global_state);
    pipeline_executor.schedule(std::move(task));
  }
  while (!global_state->is_source_drained()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  pipeline_executor.stop();
  const auto scan_end = std::chrono::steady_clock::now();
  const auto elapsed_ms =
    std::chrono::duration_cast<std::chrono::milliseconds>(scan_end - scan_start).count();
  std::cout << "scan_executor elapsed: " << elapsed_ms << " ms\n";

  // Validate repository data batches
  auto batches = drain_data_repo(data_repo);
  validate_scanned_batches(batches, num_rows, mem_mgr);

  // End the transaction
  auto commit_result = con.Query("COMMIT");
  REQUIRE(commit_result);
  REQUIRE(!commit_result->HasError());

  // Cleanup
  con.Query("DROP TABLE " + table_name);
}

//===----------------------------------------------------------------------===//
// Test: Single-threaded scan executor
//===----------------------------------------------------------------------===//

TEST_CASE("scan_executor - single threaded small table", "[scan_executor][single_thread]")
{
  // Use 10MB batch size to ensure multiple 1MB blocks are allocated
  run_scan_test("test_small", 100, 1, 10000000);
}

TEST_CASE("scan_executor - single threaded with small batches", "[scan_executor][single_thread]")
{
  // Use a small batch size to force multiple batches
  // With 4 columns (INT + BIGINT + DOUBLE + VARCHAR(256)) = 4 + 8 + 8 + 256 = 276 bytes per row
  // So ~600000 bytes should fit about 2175 rows (1 vector)
  run_scan_test("test_medium", 10000, 1, 600000);
}

//===----------------------------------------------------------------------===//
// Test: Multi-threaded scan executor
//===----------------------------------------------------------------------===//

TEST_CASE("scan_executor - multi threaded small table", "[scan_executor][multi_thread]")
{
  run_scan_test("test_mt_small", 1000, 4, 1000000);
}

TEST_CASE("scan_executor - multi threaded medium table", "[scan_executor][multi_thread]")
{
  run_scan_test("test_mt_medium", 100000, 4, 1000000);
}

TEST_CASE("scan_executor - multi threaded large table", "[scan_executor][multi_thread]")
{
  run_scan_test("test_mt_large", 500000, 8, 10000000);
}

//===----------------------------------------------------------------------===//
// Test: Edge cases
//===----------------------------------------------------------------------===//

TEST_CASE("scan_executor - empty table", "[scan_executor][edge_case]")
{
  run_scan_test("test_empty", 0, 1, 1000000);
}

TEST_CASE("scan_executor - single row table", "[scan_executor][edge_case]")
{
  run_scan_test("test_single_row", 1, 1, 1000000);
}
