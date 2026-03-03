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
#include <memory/multiple_blocks_allocation_accessor.hpp>
#include <op/result/host_table_chunk_reader.hpp>

// cudf
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

// rmm
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <algorithm>
#include <filesystem>
#include <initializer_list>
#include <string>
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

std::vector<cudf::size_type> build_null_indices(size_t num_rows,
                                                std::initializer_list<size_t> indices)
{
  std::vector<cudf::size_type> null_rows;
  null_rows.reserve(indices.size());
  for (auto idx : indices) {
    if (idx < num_rows) { null_rows.push_back(static_cast<cudf::size_type>(idx)); }
  }
  std::sort(null_rows.begin(), null_rows.end());
  null_rows.erase(std::unique(null_rows.begin(), null_rows.end()), null_rows.end());
  return null_rows;
}

void apply_null_mask(cudf::column& column,
                     std::vector<cudf::size_type> const& null_rows,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  if (null_rows.empty() || column.size() == 0) { return; }

  auto const bytes = cudf::bitmask_allocation_size_bytes(column.size());
  auto const words = bytes / sizeof(cudf::bitmask_type);
  std::vector<cudf::bitmask_type> host_mask(words, ~cudf::bitmask_type{0});
  for (auto idx : null_rows) {
    cudf::clear_bit_unsafe(host_mask.data(), idx);
  }

  rmm::device_buffer mask_buffer(host_mask.data(), bytes, stream, mr);
  column.set_null_mask(std::move(mask_buffer), static_cast<cudf::size_type>(null_rows.size()));
}

std::vector<bool> extract_validity(cudf::column_view const& col)
{
  auto const num_rows = static_cast<size_t>(col.size());
  std::vector<bool> valid(num_rows, true);
  if (num_rows == 0 || !col.nullable() || col.null_count() == 0) { return valid; }

  auto const bytes = cudf::bitmask_allocation_size_bytes(col.size());
  auto const words = bytes / sizeof(cudf::bitmask_type);
  std::vector<cudf::bitmask_type> host_mask(words);
  cudaMemcpy(host_mask.data(), col.null_mask(), bytes, cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < num_rows; ++i) {
    valid[i] = cudf::bit_is_set(host_mask.data(), static_cast<cudf::size_type>(i));
  }
  return valid;
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

host_data_representation const& convert_to_host_table(
  duckdb::shared_ptr<duckdb::SiriusContext> sirius_ctx,
  std::shared_ptr<data_batch> const& batch,
  rmm::cuda_stream_view stream)
{
  auto* data = batch->get_data();
  if (!data) { throw std::runtime_error("data_batch has no data representation"); }

  auto& manager = sirius_ctx->get_memory_manager();

  auto reservation =
    manager.request_reservation(any_memory_space_in_tier{Tier::HOST},
                                estimate_packed_data_bytes(sirius::get_cudf_table_view(*batch)));

  if (!reservation) { throw std::runtime_error("Failed to reserve host memory for test"); }

  auto* host_space = manager.get_memory_space(reservation->tier(), reservation->device_id());

  if (!host_space) { throw std::runtime_error("Invalid host memory space in test"); }

  auto& registry = sirius::converter_registry::get();
  batch->convert_to<host_data_representation>(registry, host_space, stream);

  data = batch->get_data();
  if (!data) { throw std::runtime_error("data_batch has no data after conversion"); }
  return data->cast<host_data_representation>();
}

}  // namespace

TEST_CASE("host_table_chunk_reader produces correct DataChunks",
          "[operator][result_collector][host_table_chunk_reader][shared_context]")
{
  constexpr size_t num_rows = STANDARD_VECTOR_SIZE + 5;
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

  auto const& host_table = convert_to_host_table(sirius_ctx, batch, stream);

  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER),
                                            duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT),
                                            duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR)};
  sirius::op::result::host_table_chunk_reader reader(*con.context, host_table, types);

  size_t row_base       = 0;
  auto const num_chunks = reader.calculate_num_chunks();
  for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    duckdb::DataChunk chunk;
    REQUIRE(reader.get_next_chunk(chunk));

    auto const count = static_cast<size_t>(chunk.size());
    REQUIRE(chunk.GetCapacity() == static_cast<duckdb::idx_t>(count));
    auto* int32_data = duckdb::FlatVector::GetData<int32_t>(chunk.data[0]);
    auto* int64_data = duckdb::FlatVector::GetData<int64_t>(chunk.data[1]);
    auto* str_data   = duckdb::FlatVector::GetData<duckdb::string_t>(chunk.data[2]);

    auto& int32_validity = duckdb::FlatVector::Validity(chunk.data[0]);
    auto& int64_validity = duckdb::FlatVector::Validity(chunk.data[1]);
    auto& str_validity   = duckdb::FlatVector::Validity(chunk.data[2]);
    REQUIRE(int32_validity.AllValid());
    REQUIRE(int64_validity.AllValid());
    REQUIRE(str_validity.AllValid());

    for (size_t i = 0; i < count; ++i) {
      REQUIRE(int32_data[i] == expected.int32_values[row_base + i]);
      REQUIRE(int64_data[i] == expected.int64_values[row_base + i]);
      auto const actual = std::string(str_data[i].GetData(), str_data[i].GetSize());
      REQUIRE(actual == expected_strings[row_base + i]);
    }

    row_base += count;
  }

  duckdb::DataChunk empty_chunk;
  REQUIRE(!reader.get_next_chunk(empty_chunk));
  REQUIRE(row_base == num_rows);
}

TEST_CASE("host_table_chunk_reader handles null masks",
          "[operator][result_collector][host_table_chunk_reader][shared_context]")
{
  constexpr size_t num_rows = STANDARD_VECTOR_SIZE * 2 + 3;
  auto [db_owner, con]      = sirius::make_test_db_and_connection();
  auto sirius_ctx           = sirius::get_sirius_context(con, get_test_config_path());
  auto* gpu_space           = get_default_gpu_space(sirius_ctx);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream stream;  // Must outlive data_batch for cudaMemcpyBatchAsync
  auto mr = gpu_space->get_default_allocator();

  std::vector<cudf::data_type> column_types{cudf::data_type{cudf::type_id::INT32},
                                            cudf::data_type{cudf::type_id::INT64},
                                            cudf::data_type{cudf::type_id::STRING}};
  std::vector<std::optional<std::pair<int, int>>> ranges{
    std::make_pair(0, 100), std::make_pair(1000, 2000), std::make_pair(0, 100)};

  auto table =
    sirius::create_cudf_table_with_random_data(num_rows, column_types, ranges, stream, mr, true);
  auto int64_nulls = build_null_indices(
    num_rows,
    {0, 5, STANDARD_VECTOR_SIZE - 1, STANDARD_VECTOR_SIZE, STANDARD_VECTOR_SIZE + 1, num_rows - 1});
  auto string_nulls = build_null_indices(
    num_rows, {1, 7, STANDARD_VECTOR_SIZE - 1, STANDARD_VECTOR_SIZE + 2, num_rows - 2});
  apply_null_mask(table->get_column(1), int64_nulls, stream, mr);
  apply_null_mask(table->get_column(2), string_nulls, stream, mr);

  auto batch = sirius::make_data_batch(std::move(table), *gpu_space);

  expected_table_data expected;
  std::vector<std::string> expected_strings;
  std::vector<bool> expected_int64_valid;
  std::vector<bool> expected_string_valid;
  {
    REQUIRE(batch->try_to_create_task());
    auto lock_result = batch->try_to_lock_for_processing(gpu_space->get_id());
    REQUIRE(lock_result.success);
    auto handle = std::move(lock_result.handle);

    auto const gpu_view   = sirius::get_cudf_table_view(*batch);
    expected              = extract_expected_data(gpu_view);
    expected_strings      = build_expected_strings(expected);
    expected_int64_valid  = extract_validity(gpu_view.column(1));
    expected_string_valid = extract_validity(gpu_view.column(2));
  }

  auto const& host_table = convert_to_host_table(sirius_ctx, batch, stream);

  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER),
                                            duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT),
                                            duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR)};
  sirius::op::result::host_table_chunk_reader reader(*con.context, host_table, types);

  size_t row_base       = 0;
  auto const num_chunks = reader.calculate_num_chunks();
  for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    duckdb::DataChunk chunk;
    REQUIRE(reader.get_next_chunk(chunk));

    auto const count = static_cast<size_t>(chunk.size());
    auto* int32_data = duckdb::FlatVector::GetData<int32_t>(chunk.data[0]);
    auto* int64_data = duckdb::FlatVector::GetData<int64_t>(chunk.data[1]);
    auto* str_data   = duckdb::FlatVector::GetData<duckdb::string_t>(chunk.data[2]);

    auto& int32_validity = duckdb::FlatVector::Validity(chunk.data[0]);
    auto& int64_validity = duckdb::FlatVector::Validity(chunk.data[1]);
    auto& str_validity   = duckdb::FlatVector::Validity(chunk.data[2]);

    REQUIRE(int32_validity.AllValid());

    for (size_t i = 0; i < count; ++i) {
      auto const row_idx = row_base + i;
      REQUIRE(int32_data[i] == expected.int32_values[row_idx]);

      REQUIRE(int64_validity.RowIsValid(static_cast<duckdb::idx_t>(i)) ==
              expected_int64_valid[row_idx]);
      if (expected_int64_valid[row_idx]) {
        REQUIRE(int64_data[i] == expected.int64_values[row_idx]);
      }

      REQUIRE(str_validity.RowIsValid(static_cast<duckdb::idx_t>(i)) ==
              expected_string_valid[row_idx]);
      if (expected_string_valid[row_idx]) {
        auto const actual = std::string(str_data[i].GetData(), str_data[i].GetSize());
        REQUIRE(actual == expected_strings[row_idx]);
      }
    }

    row_base += count;
  }

  duckdb::DataChunk empty_chunk;
  REQUIRE(!reader.get_next_chunk(empty_chunk));
  REQUIRE(row_base == num_rows);
}
