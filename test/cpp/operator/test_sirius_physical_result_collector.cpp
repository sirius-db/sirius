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
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// duckdb
#include <duckdb/common/vector_size.hpp>
#include <duckdb/main/prepared_statement_data.hpp>

// standard library
#include <algorithm>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using namespace sirius;
using namespace cucascade;
using namespace cucascade::memory;

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

duckdb::Connection& get_test_connection()
{
  static duckdb::DuckDB db(nullptr);
  static duckdb::Connection con(db);
  return con;
}

duckdb::ClientContext& get_test_client_context() { return *get_test_connection().context; }

duckdb::shared_ptr<duckdb::SiriusContext> get_test_sirius_context()
{
  return sirius::get_sirius_context(get_test_connection(), get_test_config_path());
}

memory_space* get_default_gpu_space()
{
  auto sirius_ctx = get_test_sirius_context();
  auto& manager   = sirius_ctx->get_memory_manager();
  auto* space     = manager.get_memory_space(Tier::GPU, 0);
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

void convert_batch_to_host(std::shared_ptr<data_batch> const& batch)
{
  auto* data = batch->get_data();
  if (!data) { throw std::runtime_error("data_batch has no data representation"); }

  auto const view       = sirius::get_cudf_table_view(*batch);
  auto const data_bytes = estimate_packed_data_bytes(view);

  auto sirius_ctx = get_test_sirius_context();
  auto& manager   = sirius_ctx->get_memory_manager();

  auto reservation = manager.request_reservation(any_memory_space_in_tier{Tier::HOST}, data_bytes);
  if (!reservation) { throw std::runtime_error("Failed to reserve host memory for test"); }

  auto* host_space = manager.get_memory_space(reservation->tier(), reservation->device_id());
  if (!host_space) { throw std::runtime_error("Invalid host memory space for test"); }

  auto& registry = sirius::converter_registry::get();
  batch->convert_to<host_table_representation>(registry, host_space, rmm::cuda_stream_default);
}

}  // namespace

TEST_CASE("sirius_physical_materialized_collector sink with host input",
          "[operator][result_collector][sirius_result_collector]")
{
  constexpr size_t num_rows = STANDARD_VECTOR_SIZE + 7;
  auto* gpu_space           = get_default_gpu_space();
  REQUIRE(gpu_space != nullptr);

  std::vector<cudf::data_type> column_types{cudf::data_type{cudf::type_id::INT32},
                                            cudf::data_type{cudf::type_id::INT64},
                                            cudf::data_type{cudf::type_id::STRING}};
  std::vector<std::optional<std::pair<int, int>>> ranges{
    std::make_pair(0, 100), std::make_pair(1000, 2000), std::make_pair(0, 100)};

  auto table = sirius::create_cudf_table_with_random_data(num_rows,
                                                          column_types,
                                                          ranges,
                                                          cudf::get_default_stream(),
                                                          gpu_space->get_default_allocator(),
                                                          true);
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

  convert_batch_to_host(batch);

  duckdb::vector<duckdb::LogicalType> types{
    duckdb::LogicalType::INTEGER, duckdb::LogicalType::BIGINT, duckdb::LogicalType::VARCHAR};
  auto prepared =
    duckdb::make_shared_ptr<duckdb::PreparedStatementData>(duckdb::StatementType::SELECT_STATEMENT);
  prepared->types = types;
  prepared->names = {"c0", "c1", "c2"};
  auto plan       = duckdb::make_uniq<sirius::op::sirius_physical_dummy_scan>(types, 0);
  auto sirius_prepared =
    duckdb::make_shared_ptr<sirius_prepared_statement_data>(prepared, std::move(plan));
  sirius::op::sirius_physical_materialized_collector collector(*sirius_prepared,
                                                               get_test_client_context());

  collector.sink({batch});
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
          "[operator][result_collector][sirius_result_collector]")
{
  constexpr size_t num_rows = STANDARD_VECTOR_SIZE * 2 + 3;
  auto* gpu_space           = get_default_gpu_space();
  REQUIRE(gpu_space != nullptr);

  std::vector<cudf::data_type> column_types{cudf::data_type{cudf::type_id::INT32},
                                            cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges{std::make_pair(0, 100),
                                                         std::make_pair(1000, 2000)};

  auto table = sirius::create_cudf_table_with_random_data(num_rows,
                                                          column_types,
                                                          ranges,
                                                          cudf::get_default_stream(),
                                                          gpu_space->get_default_allocator(),
                                                          false);
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

  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType::INTEGER,
                                            duckdb::LogicalType::BIGINT};
  auto prepared =
    duckdb::make_shared_ptr<duckdb::PreparedStatementData>(duckdb::StatementType::SELECT_STATEMENT);
  prepared->types = types;
  prepared->names = {"c0", "c1"};
  auto plan       = duckdb::make_uniq<sirius::op::sirius_physical_dummy_scan>(types, 0);
  auto sirius_prepared =
    duckdb::make_shared_ptr<sirius_prepared_statement_data>(prepared, std::move(plan));
  sirius::op::sirius_physical_materialized_collector collector(*sirius_prepared,
                                                               get_test_client_context());

  collector.sink({batch});
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
