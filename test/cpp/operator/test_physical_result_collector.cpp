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
#include <op/sirius_physical_dummy_scan.hpp>
#include <op/sirius_physical_result_collector.hpp>
#include <sirius_interface.hpp>

// cucascades
#include <cucascade/data/cpu_data_representation.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

// rmm
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

// duckdb
#include <duckdb/common/vector_size.hpp>
#include <duckdb/main/prepared_statement_data.hpp>

// standard library
#include <algorithm>
#include <atomic>
#include <exception>
#include <filesystem>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace sirius;
using namespace cucascade;
using namespace cucascade::memory;
using sirius::op::operator_data;

namespace {

struct expected_table_data {
  std::vector<int32_t> int32_values;
  std::vector<int64_t> int64_values;
  std::vector<int64_t> string_offsets;
  std::vector<char> string_chars;
};

std::filesystem::path get_test_config_path()
{
  return std::filesystem::path(__FILE__).parent_path() / "result.cfg";
}

memory_space* get_default_gpu_space(duckdb::shared_ptr<duckdb::SiriusContext>& sirius_ctx)
{
  auto& manager = sirius_ctx->get_memory_manager();
  auto* space   = manager.get_memory_space(Tier::GPU, 0);
  if (space) { return space; }
  auto spaces = manager.get_memory_spaces_for_tier(Tier::GPU);
  if (!spaces.empty()) { return const_cast<memory_space*>(spaces.front()); }
  return nullptr;
}

expected_table_data extract_expected_data(cudf::table_view const& view)
{
  expected_table_data data;
  auto const num_rows = static_cast<size_t>(view.num_rows());

  data.int32_values.resize(num_rows);
  data.int64_values.resize(num_rows);

  if (num_rows > 0) {
    cudaMemcpy(data.int32_values.data(),
               view.column(0).data<int32_t>(),
               sizeof(int32_t) * num_rows,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(data.int64_values.data(),
               view.column(1).data<int64_t>(),
               sizeof(int64_t) * num_rows,
               cudaMemcpyDeviceToHost);
  }

  if (view.num_columns() < 3) { return data; }

  cudf::strings_column_view str_col(view.column(2));
  auto offsets_view = str_col.offsets();
  data.string_offsets.resize(num_rows + 1, 0);
  if (num_rows > 0) {
    if (offsets_view.type().id() == cudf::type_id::INT64) {
      cudaMemcpy(data.string_offsets.data(),
                 offsets_view.data<int64_t>(),
                 sizeof(int64_t) * (num_rows + 1),
                 cudaMemcpyDeviceToHost);
    } else {
      std::vector<cudf::size_type> temp_offsets(num_rows + 1, 0);
      cudaMemcpy(temp_offsets.data(),
                 offsets_view.data<cudf::size_type>(),
                 sizeof(cudf::size_type) * (num_rows + 1),
                 cudaMemcpyDeviceToHost);
      std::transform(temp_offsets.begin(),
                     temp_offsets.end(),
                     data.string_offsets.begin(),
                     [](cudf::size_type value) { return static_cast<int64_t>(value); });
    }
  }

  auto const chars_size = static_cast<size_t>(str_col.chars_size(cudf::get_default_stream()));
  data.string_chars.resize(chars_size);
  if (chars_size > 0) {
    cudaMemcpy(data.string_chars.data(),
               str_col.chars_begin(cudf::get_default_stream()),
               chars_size,
               cudaMemcpyDeviceToHost);
  }

  return data;
}

std::vector<std::string> build_expected_strings(expected_table_data const& data)
{
  if (data.string_offsets.empty()) { return {}; }
  std::vector<std::string> strings;
  strings.reserve(data.string_offsets.size() - 1);
  for (size_t i = 0; i + 1 < data.string_offsets.size(); ++i) {
    auto const start = static_cast<size_t>(data.string_offsets[i]);
    auto const end   = static_cast<size_t>(data.string_offsets[i + 1]);
    if (end == start) {
      strings.emplace_back();
      continue;
    }
    strings.emplace_back(data.string_chars.data() + start, end - start);
  }
  return strings;
}

size_t estimate_packed_data_bytes(cudf::table_view const& view)
{
  size_t total_bytes = 0;
  for (auto const& col : view) {
    if (col.type().id() == cudf::type_id::STRING) {
      cudf::strings_column_view strings(col);
      auto const offsets_view  = strings.offsets();
      auto const offsets_bytes = static_cast<size_t>(offsets_view.size()) *
                                 static_cast<size_t>(cudf::size_of(offsets_view.type()));
      auto const chars_bytes = static_cast<size_t>(strings.chars_size(cudf::get_default_stream()));
      total_bytes += offsets_bytes + chars_bytes;
    } else {
      total_bytes +=
        static_cast<size_t>(col.size()) * static_cast<size_t>(cudf::size_of(col.type()));
    }
    if (col.nullable()) {
      total_bytes += static_cast<size_t>(cudf::bitmask_allocation_size_bytes(col.size()));
    }
  }
  return total_bytes;
}

void convert_batch_to_host(duckdb::shared_ptr<duckdb::SiriusContext> sirius_ctx,
                           std::shared_ptr<data_batch> const& batch,
                           rmm::cuda_stream_view stream)
{
  auto* data = batch->get_data();
  if (!data) { throw std::runtime_error("data_batch has no data representation"); }

  auto const view       = sirius::get_cudf_table_view(*batch);
  auto const data_bytes = estimate_packed_data_bytes(view);

  auto& manager = sirius_ctx->get_memory_manager();

  auto reservation = manager.request_reservation(any_memory_space_in_tier{Tier::HOST}, data_bytes);
  if (!reservation) { throw std::runtime_error("Failed to reserve host memory for test"); }

  auto* host_space = manager.get_memory_space(reservation->tier(), reservation->device_id());
  if (!host_space) { throw std::runtime_error("Invalid host memory space for test"); }

  auto& registry = sirius::converter_registry::get();
  batch->convert_to<cucascade::host_data_representation>(registry, host_space, stream);
}

}  // namespace

TEST_CASE("sirius_physical_materialized_collector sink with host input",
          "[operator][physical_result_collector][shared_context]")
{
  constexpr size_t num_rows = STANDARD_VECTOR_SIZE + 7;
  auto [db_owner, con]      = sirius::make_test_db_and_connection();
  auto sirius_ctx           = sirius::get_sirius_context(con, get_test_config_path());
  auto* gpu_space           = get_default_gpu_space(sirius_ctx);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream stream;  // Must outlive data_batch for cudaMemcpyBatchAsync

  std::vector<cudf::data_type> column_types{cudf::data_type{cudf::type_id::INT32},
                                            cudf::data_type{cudf::type_id::INT64},
                                            cudf::data_type{cudf::type_id::STRING}};
  std::vector<std::optional<std::pair<int, int>>> ranges{
    std::make_pair(0, 100), std::make_pair(1000, 2000), std::make_pair(0, 100)};

  auto table = sirius::create_cudf_table_with_random_data(
    num_rows, column_types, ranges, stream, gpu_space->get_default_allocator(), true);
  auto batch = sirius::make_data_batch(std::move(table), *gpu_space);

  expected_table_data expected;
  std::vector<std::string> expected_strings;
  {
    REQUIRE(batch->try_to_create_task());
    auto lock_result = batch->try_to_lock_for_processing(gpu_space->get_id());
    REQUIRE(lock_result.success);
    auto handle = std::move(lock_result.handle);

    auto const gpu_view = sirius::get_cudf_table_view(*batch);
    expected            = extract_expected_data(gpu_view);
    expected_strings    = build_expected_strings(expected);
  }

  convert_batch_to_host(sirius_ctx, batch, stream);

  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER),
                                            duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT),
                                            duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR)};
  auto prepared =
    duckdb::make_shared_ptr<duckdb::PreparedStatementData>(duckdb::StatementType::SELECT_STATEMENT);
  prepared->types = types;
  prepared->names = {"c0", "c1", "c2"};
  auto plan       = duckdb::make_uniq<sirius::op::sirius_physical_dummy_scan>(types, 0);
  auto sirius_prepared =
    duckdb::make_shared_ptr<sirius_prepared_statement_data>(prepared, std::move(plan));
  sirius::op::sirius_physical_materialized_collector collector(*sirius_prepared, *con.context);

  collector.sink(operator_data({batch}), cudf::get_default_stream());
  duckdb::GlobalSinkState sink_state;
  auto result = collector.get_result(sink_state);
  REQUIRE(result != nullptr);

  size_t row_base = 0;
  for (;;) {
    auto chunk = result->FetchRaw();
    if (!chunk) { break; }
    auto const count = static_cast<size_t>(chunk->size());
    auto* int32_data = duckdb::FlatVector::GetData<int32_t>(chunk->data[0]);
    auto* int64_data = duckdb::FlatVector::GetData<int64_t>(chunk->data[1]);
    auto* str_data   = duckdb::FlatVector::GetData<duckdb::string_t>(chunk->data[2]);

    for (size_t i = 0; i < count; ++i) {
      REQUIRE(int32_data[i] == expected.int32_values[row_base + i]);
      REQUIRE(int64_data[i] == expected.int64_values[row_base + i]);
      auto const actual = std::string(str_data[i].GetData(), str_data[i].GetSize());
      REQUIRE(actual == expected_strings[row_base + i]);
    }
    row_base += count;
  }
  REQUIRE(row_base == num_rows);
}

TEST_CASE("sirius_physical_materialized_collector sink converts GPU input",
          "[operator][physical_result_collector][shared_context]")
{
  constexpr size_t num_rows = STANDARD_VECTOR_SIZE * 2 + 3;
  auto [db_owner, con]      = sirius::make_test_db_and_connection();
  auto sirius_ctx           = sirius::get_sirius_context(con, get_test_config_path());
  auto* gpu_space           = get_default_gpu_space(sirius_ctx);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream stream;  // Must outlive data_batch for cudaMemcpyBatchAsync

  std::vector<cudf::data_type> column_types{cudf::data_type{cudf::type_id::INT32},
                                            cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges{std::make_pair(0, 100),
                                                         std::make_pair(1000, 2000)};

  auto table = sirius::create_cudf_table_with_random_data(
    num_rows, column_types, ranges, stream, gpu_space->get_default_allocator(), false);
  auto batch = sirius::make_data_batch(std::move(table), *gpu_space);

  expected_table_data expected;
  {
    REQUIRE(batch->try_to_create_task());
    auto lock_result = batch->try_to_lock_for_processing(gpu_space->get_id());
    REQUIRE(lock_result.success);
    auto handle = std::move(lock_result.handle);

    auto const gpu_view = sirius::get_cudf_table_view(*batch);
    expected            = extract_expected_data(gpu_view);
  }

  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER),
                                            duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT)};
  auto prepared =
    duckdb::make_shared_ptr<duckdb::PreparedStatementData>(duckdb::StatementType::SELECT_STATEMENT);
  prepared->types = types;
  prepared->names = {"c0", "c1"};
  auto plan       = duckdb::make_uniq<sirius::op::sirius_physical_dummy_scan>(types, 0);
  auto sirius_prepared =
    duckdb::make_shared_ptr<sirius_prepared_statement_data>(prepared, std::move(plan));
  sirius::op::sirius_physical_materialized_collector collector(*sirius_prepared, *con.context);

  collector.sink(operator_data({batch}), stream);
  duckdb::GlobalSinkState sink_state;
  auto result = collector.get_result(sink_state);
  REQUIRE(result != nullptr);

  size_t row_base = 0;
  for (;;) {
    auto chunk = result->FetchRaw();
    if (!chunk) { break; }
    auto const count = static_cast<size_t>(chunk->size());
    auto* int32_data = duckdb::FlatVector::GetData<int32_t>(chunk->data[0]);
    auto* int64_data = duckdb::FlatVector::GetData<int64_t>(chunk->data[1]);

    for (size_t i = 0; i < count; ++i) {
      REQUIRE(int32_data[i] == expected.int32_values[row_base + i]);
      REQUIRE(int64_data[i] == expected.int64_values[row_base + i]);
    }
    row_base += count;
  }
  REQUIRE(row_base == num_rows);
}

TEST_CASE("sirius_physical_materialized_collector sink supports concurrent append",
          "[operator][physical_result_collector][concurrent][shared_context]")
{
  constexpr int num_threads       = 4;
  constexpr size_t rows_per_batch = STANDARD_VECTOR_SIZE * 3 + 13;
  auto [db_owner, con]            = sirius::make_test_db_and_connection();
  auto sirius_ctx                 = sirius::get_sirius_context(con, get_test_config_path());
  auto* gpu_space                 = get_default_gpu_space(sirius_ctx);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream stream;  // Must outlive data_batch for cudaMemcpyBatchAsync

  using row_t = std::pair<int32_t, int64_t>;
  std::vector<row_t> expected_rows;
  expected_rows.reserve(static_cast<size_t>(num_threads) * rows_per_batch);

  std::vector<std::shared_ptr<data_batch>> batches;
  batches.reserve(num_threads);
  auto mr = gpu_space->get_default_allocator();

  for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    std::vector<int32_t> col0_values(rows_per_batch, thread_idx);
    std::vector<int64_t> col1_values(rows_per_batch);
    for (size_t row_idx = 0; row_idx < rows_per_batch; ++row_idx) {
      col1_values[row_idx] =
        static_cast<int64_t>(thread_idx) * static_cast<int64_t>(rows_per_batch) +
        static_cast<int64_t>(row_idx);
      expected_rows.emplace_back(col0_values[row_idx], col1_values[row_idx]);
    }

    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(2);

    auto col0 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(rows_per_batch),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
    cudaMemcpy(col0->mutable_view().data<int32_t>(),
               col0_values.data(),
               sizeof(int32_t) * rows_per_batch,
               cudaMemcpyHostToDevice);
    cols.push_back(std::move(col0));

    auto col1 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                          static_cast<cudf::size_type>(rows_per_batch),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
    cudaMemcpy(col1->mutable_view().data<int64_t>(),
               col1_values.data(),
               sizeof(int64_t) * rows_per_batch,
               cudaMemcpyHostToDevice);
    cols.push_back(std::move(col1));

    auto table = std::make_unique<cudf::table>(std::move(cols));
    auto batch = sirius::make_data_batch(std::move(table), *gpu_space);
    convert_batch_to_host(sirius_ctx, batch, stream);
    batches.emplace_back(std::move(batch));
  }

  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType::INTEGER,
                                            duckdb::LogicalType::BIGINT};
  auto prepared =
    duckdb::make_shared_ptr<duckdb::PreparedStatementData>(duckdb::StatementType::SELECT_STATEMENT);
  prepared->types = types;
  prepared->names = {"c0", "c1"};
  auto plan       = duckdb::make_uniq<sirius::op::sirius_physical_dummy_scan>(types, 0);
  auto sirius_prepared =
    duckdb::make_shared_ptr<sirius_prepared_statement_data>(prepared, std::move(plan));
  sirius::op::sirius_physical_materialized_collector collector(*sirius_prepared, *con.context);

  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::exception_ptr> exceptions(static_cast<size_t>(num_threads));
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    threads.emplace_back([&, thread_idx]() {
      ready.fetch_add(1, std::memory_order_relaxed);
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      try {
        collector.sink(operator_data({batches[static_cast<size_t>(thread_idx)]}),
                       cudf::get_default_stream());
      } catch (...) {
        exceptions[static_cast<size_t>(thread_idx)] = std::current_exception();
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != num_threads) {
    std::this_thread::yield();
  }
  // All spawned threads are ready, let them go
  go.store(true, std::memory_order_release);

  // Wait for all threads to finish
  for (auto& t : threads) {
    t.join();
  }

  // Check for exceptions
  for (auto const& ex : exceptions) {
    if (!ex) { continue; }
    try {
      std::rethrow_exception(ex);
    } catch (std::exception const& e) {
      FAIL(e.what());
    } catch (...) {
      FAIL("Unknown exception in concurrent sink.");
    }
  }

  duckdb::GlobalSinkState sink_state;
  auto result = collector.get_result(sink_state);
  REQUIRE(result != nullptr);

  std::vector<row_t> actual_rows;
  actual_rows.reserve(expected_rows.size());
  for (;;) {
    auto chunk = result->FetchRaw();
    if (!chunk) { break; }
    auto const count = static_cast<size_t>(chunk->size());
    auto* int32_data = duckdb::FlatVector::GetData<int32_t>(chunk->data[0]);
    auto* int64_data = duckdb::FlatVector::GetData<int64_t>(chunk->data[1]);
    for (size_t i = 0; i < count; ++i) {
      actual_rows.emplace_back(int32_data[i], int64_data[i]);
    }
  }

  std::sort(actual_rows.begin(), actual_rows.end());
  std::sort(expected_rows.begin(), expected_rows.end());
  REQUIRE(actual_rows == expected_rows);
}
