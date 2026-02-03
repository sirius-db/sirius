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
#include <helper/utils.hpp>
#include <memory/host_table_utils.hpp>
#include <memory/multiple_blocks_allocation_accessor.hpp>
#include <op/scan/duckdb_scan_task.hpp>

// cucascade
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/host_table.hpp>

// cudf
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

// duckdb
#include <duckdb.hpp>
#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/types/vector.hpp>

// cuda
#include <cuda_runtime.h>

// rmm
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

// standard library
#include <algorithm>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <vector>

using namespace sirius::op::scan;
using namespace cucascade::memory;

namespace {

constexpr size_t kDefaultVarcharSize = 256;

std::filesystem::path get_test_config_path()
{
  return std::filesystem::path(__FILE__).parent_path() / "memory.cfg";
}

memory_space* get_memory_space(cucascade::memory::Tier tier, int device_id)
{
  auto sirius_ctx = sirius::get_sirius_context(get_test_config_path());
  auto& manager   = sirius_ctx->get_memory_manager();
  auto* space     = manager.get_memory_space(tier, device_id);
  if (space) { return space; }
  auto spaces = manager.get_memory_spaces_for_tier(tier);
  if (!spaces.empty()) { return const_cast<memory_space*>(spaces.front()); }
  return nullptr;
}

std::vector<cudf::bitmask_type> copy_null_mask(const cudf::column_view& col)
{
  if (col.null_mask() == nullptr) { return {}; }
  auto const bytes = cudf::bitmask_allocation_size_bytes(col.size());
  std::vector<cudf::bitmask_type> mask(bytes / sizeof(cudf::bitmask_type));
  cudaMemcpy(mask.data(), col.null_mask(), bytes, cudaMemcpyDeviceToHost);
  return mask;
}

void verify_validity_mask(const cudf::column_view& col, const std::vector<bool>& expected_valid)
{
  size_t expected_nulls = 0;
  for (auto valid : expected_valid) {
    if (!valid) { ++expected_nulls; }
  }
  REQUIRE(col.null_count() == static_cast<cudf::size_type>(expected_nulls));

  auto mask = copy_null_mask(col);
  if (mask.empty()) {
    REQUIRE(expected_nulls == 0);
    return;
  }

  for (size_t i = 0; i < expected_valid.size(); ++i) {
    bool actual_valid = cudf::bit_is_set(mask.data(), i);
    REQUIRE(actual_valid == expected_valid[i]);
  }
}

template <typename T>
void verify_numeric_column(const cudf::column_view& col,
                           const std::vector<T>& expected,
                           const std::vector<bool>& expected_valid)
{
  REQUIRE(static_cast<size_t>(col.size()) == expected.size());
  REQUIRE(expected.size() == expected_valid.size());

  std::vector<T> actual(expected.size());
  cudaMemcpy(actual.data(), col.data<T>(), sizeof(T) * expected.size(), cudaMemcpyDeviceToHost);

  verify_validity_mask(col, expected_valid);

  auto mask = copy_null_mask(col);
  for (size_t i = 0; i < expected.size(); ++i) {
    bool is_valid = mask.empty() ? true : cudf::bit_is_set(mask.data(), i);
    if (is_valid) { REQUIRE(actual[i] == expected[i]); }
  }
}

void verify_string_column(const cudf::column_view& col,
                          const std::vector<std::string>& expected,
                          const std::vector<bool>& expected_valid)
{
  REQUIRE(static_cast<size_t>(col.size()) == expected.size());
  REQUIRE(expected.size() == expected_valid.size());

  cudf::strings_column_view str_col(col);
  auto offsets_col = str_col.offsets();
  std::vector<int64_t> offsets(expected.size() + 1);
  if (offsets_col.type().id() == cudf::type_id::INT64) {
    cudaMemcpy(offsets.data(),
               offsets_col.data<int64_t>(),
               offsets.size() * sizeof(int64_t),
               cudaMemcpyDeviceToHost);
  } else {
    std::vector<cudf::size_type> offsets32(expected.size() + 1);
    cudaMemcpy(offsets32.data(),
               offsets_col.data<cudf::size_type>(),
               offsets32.size() * sizeof(cudf::size_type),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < offsets.size(); ++i) {
      offsets[i] = static_cast<int64_t>(offsets32[i]);
    }
  }

  std::vector<char> chars(static_cast<size_t>(offsets.back()));
  if (!chars.empty()) {
    cudaMemcpy(chars.data(),
               str_col.chars_begin(cudf::get_default_stream()),
               chars.size(),
               cudaMemcpyDeviceToHost);
  }

  std::vector<int64_t> expected_offsets;
  expected_offsets.reserve(expected.size() + 1);
  expected_offsets.push_back(0);
  std::vector<char> expected_chars;

  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected_valid[i]) {
      expected_chars.insert(expected_chars.end(), expected[i].begin(), expected[i].end());
      expected_offsets.push_back(expected_offsets.back() +
                                 static_cast<int64_t>(expected[i].size()));
    } else {
      expected_offsets.push_back(expected_offsets.back());
    }
  }

  REQUIRE(expected_offsets.size() == offsets.size());
  for (size_t i = 0; i < offsets.size(); ++i) {
    REQUIRE(expected_offsets[i] == offsets[i]);
  }
  REQUIRE(expected_chars.size() == chars.size());
  for (size_t i = 0; i < chars.size(); ++i) {
    REQUIRE(expected_chars[i] == chars[i]);
  }

  verify_validity_mask(col, expected_valid);
}

struct expected_table_data {
  std::vector<int32_t> int32_values;
  std::vector<int64_t> int64_values;
  std::vector<int64_t> string_offsets;
  std::vector<char> string_chars;
};

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

size_t mask_bytes_for_rows(size_t num_rows) { return (num_rows + 7) / 8; }

void mask_unused_bits(std::vector<uint8_t>& mask, size_t num_rows)
{
  if (mask.empty()) { return; }
  auto const tail_bits = num_rows % 8;
  if (tail_bits == 0) { return; }
  auto const keep_mask = static_cast<uint8_t>((1u << tail_bits) - 1u);
  mask.back() &= keep_mask;
}

std::vector<uint8_t> extract_mask_bytes(cudf::column_view const& col)
{
  auto const num_rows = static_cast<size_t>(col.size());
  auto const bytes    = mask_bytes_for_rows(num_rows);
  std::vector<uint8_t> mask(bytes, 0);
  if (bytes == 0 || !col.nullable() || col.null_count() == 0) { return mask; }
  cudaMemcpy(mask.data(), col.null_mask(), bytes, cudaMemcpyDeviceToHost);
  mask_unused_bits(mask, num_rows);
  return mask;
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

cucascade::host_table_representation const& convert_to_host_table(
  std::shared_ptr<cucascade::data_batch> const& batch)
{
  auto* data = batch->get_data();
  if (!data) { throw std::runtime_error("data_batch has no data representation"); }

  auto sirius_ctx = sirius::get_sirius_context(get_test_config_path());
  auto& manager   = sirius_ctx->get_memory_manager();

  auto reservation =
    manager.request_reservation(any_memory_space_in_tier{Tier::HOST},
                                estimate_packed_data_bytes(sirius::get_cudf_table_view(*batch)));

  if (!reservation) { throw std::runtime_error("Failed to reserve host memory for test"); }

  auto* host_space = manager.get_memory_space(reservation->tier(), reservation->device_id());

  if (!host_space) { throw std::runtime_error("Invalid host memory space in test"); }

  auto& registry = sirius::converter_registry::get();
  batch->convert_to<cucascade::host_table_representation>(
    registry, host_space, rmm::cuda_stream_default);

  data = batch->get_data();
  if (!data) { throw std::runtime_error("data_batch has no data after conversion"); }
  return data->cast<cucascade::host_table_representation>();
}

}  // namespace

TEST_CASE("host_table_utils - pack metadata with gaps across multiple blocks",
          "[memory][host_table_utils]")
{
  using metadata_node = sirius::metadata_node;

  auto* host_space = get_memory_space(Tier::HOST, 0);
  REQUIRE(host_space != nullptr);
  auto* gpu_space = get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto* allocator = host_space->get_memory_resource_as<fixed_size_host_memory_resource>();
  REQUIRE(allocator != nullptr);

  size_t num_rows       = 1024;
  size_t mask_bytes     = 0;
  size_t total_size     = 0;
  auto const block_size = allocator->get_block_size();
  while (true) {
    mask_bytes = sirius::utils::ceil_div_8(num_rows);
    total_size = num_rows * sizeof(int32_t) + mask_bytes + (num_rows + 1) * sizeof(int64_t) +
                 num_rows * kDefaultVarcharSize + mask_bytes + num_rows * sizeof(int64_t) +
                 mask_bytes;
    if (total_size > block_size) { break; }
    num_rows *= 2;
  }

  auto allocation = allocator->allocate_multiple_blocks(total_size);
  REQUIRE(allocation != nullptr);
  REQUIRE(allocation->size() > 1);

  duckdb::LogicalType int_type(duckdb::LogicalTypeId::INTEGER);
  duckdb::LogicalType str_type(duckdb::LogicalTypeId::VARCHAR);
  duckdb::LogicalType big_type(duckdb::LogicalTypeId::BIGINT);

  duckdb_scan_task_local_state::column_builder int_builder(int_type, kDefaultVarcharSize);
  duckdb_scan_task_local_state::column_builder str_builder(str_type, kDefaultVarcharSize);
  duckdb_scan_task_local_state::column_builder big_builder(big_type, kDefaultVarcharSize);

  size_t byte_offset = 0;
  int_builder.initialize_accessors(num_rows, byte_offset, allocation);
  byte_offset += num_rows * sizeof(int32_t) + mask_bytes;
  str_builder.initialize_accessors(num_rows, byte_offset, allocation);
  byte_offset += (num_rows + 1) * sizeof(int64_t) + num_rows * kDefaultVarcharSize + mask_bytes;
  big_builder.initialize_accessors(num_rows, byte_offset, allocation);

  duckdb::Vector int_vec(int_type, num_rows);
  auto* int_data = reinterpret_cast<int32_t*>(int_vec.GetData());
  std::vector<int32_t> expected_int(num_rows);
  std::vector<bool> int_valid(num_rows, true);
  duckdb::ValidityMask int_validity(num_rows);
  int_validity.Initialize(num_rows);
  int_validity.SetAllValid(num_rows);
  for (size_t i = 0; i < num_rows; ++i) {
    expected_int[i] = static_cast<int32_t>(i * 2);
    int_data[i]     = expected_int[i];
    if (i % 10 == 0) {
      int_validity.SetInvalid(i);
      int_valid[i] = false;
    }
  }

  duckdb::Vector big_vec(big_type, num_rows);
  auto* big_data = reinterpret_cast<int64_t*>(big_vec.GetData());
  std::vector<int64_t> expected_big(num_rows);
  std::vector<bool> big_valid(num_rows, true);
  duckdb::ValidityMask big_validity(num_rows);
  big_validity.Initialize(num_rows);
  big_validity.SetAllValid(num_rows);
  for (size_t i = 0; i < num_rows; ++i) {
    expected_big[i] = static_cast<int64_t>(i) * 1000;
    big_data[i]     = expected_big[i];
    if (i % 9 == 0) {
      big_validity.SetInvalid(i);
      big_valid[i] = false;
    }
  }

  duckdb::Vector str_vec(str_type, num_rows);
  auto* str_data = reinterpret_cast<duckdb::string_t*>(str_vec.GetData());
  std::vector<std::string> expected_str(num_rows);
  std::vector<bool> str_valid(num_rows, true);
  duckdb::ValidityMask str_validity(num_rows);
  str_validity.Initialize(num_rows);
  str_validity.SetAllValid(num_rows);
  const char* patterns[] = {"a", "bb", "ccc", "dddd"};
  for (size_t i = 0; i < num_rows; ++i) {
    expected_str[i] = patterns[i % 4];
    str_data[i]     = duckdb::string_t(patterns[i % 4]);
    if (i % 7 == 0) {
      str_validity.SetInvalid(i);
      str_valid[i] = false;
    }
  }

  int_builder.process_column(int_vec, int_validity, num_rows, 0, allocation);
  str_builder.process_column(str_vec, str_validity, num_rows, 0, allocation);
  big_builder.process_column(big_vec, big_validity, num_rows, 0, allocation);

  REQUIRE(str_builder.total_data_bytes < str_builder.total_data_bytes_allocated);
  size_t str_end =
    str_builder.data_blocks_accessor.initial_byte_offset + str_builder.total_data_bytes;
  REQUIRE(str_end < big_builder.data_blocks_accessor.initial_byte_offset);

  std::vector<metadata_node> column_metadata;
  column_metadata.reserve(3);
  column_metadata.push_back(int_builder.make_metadata_node(num_rows));
  column_metadata.push_back(str_builder.make_metadata_node(num_rows));
  column_metadata.push_back(big_builder.make_metadata_node(num_rows));
  auto metadata = std::make_unique<std::vector<uint8_t>>(pack_metadata_from_nodes(column_metadata));

  auto const sz         = allocation->size_bytes();
  auto table_allocation = std::make_unique<cucascade::memory::host_table_allocation>(
    std::move(allocation), std::move(metadata), sz);
  auto host_table =
    std::make_unique<cucascade::host_table_representation>(std::move(table_allocation), host_space);
  auto batch =
    std::make_shared<cucascade::data_batch>(sirius::get_next_batch_id(), std::move(host_table));

  auto& registry = sirius::converter_registry::get();
  batch->convert_to<cucascade::gpu_table_representation>(
    registry, gpu_space, cudf::get_default_stream());

  cudf::table_view table_view = sirius::get_cudf_table_view(*batch);
  REQUIRE(table_view.num_rows() == static_cast<cudf::size_type>(num_rows));
  REQUIRE(table_view.num_columns() == 3);
  REQUIRE(table_view.column(0).type().id() == cudf::type_id::INT32);
  REQUIRE(table_view.column(1).type().id() == cudf::type_id::STRING);
  REQUIRE(table_view.column(2).type().id() == cudf::type_id::INT64);

  verify_numeric_column<int32_t>(table_view.column(0), expected_int, int_valid);
  verify_string_column(table_view.column(1), expected_str, str_valid);
  verify_numeric_column<int64_t>(table_view.column(2), expected_big, big_valid);
}

TEST_CASE("host_table_utils - underfilled varchar column truncates rows",
          "[memory][host_table_utils]")
{
  using metadata_node = sirius::metadata_node;

  auto* host_space = get_memory_space(Tier::HOST, 0);
  REQUIRE(host_space != nullptr);
  auto* gpu_space = get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto* allocator = host_space->get_memory_resource_as<fixed_size_host_memory_resource>();
  REQUIRE(allocator != nullptr);

  constexpr size_t kSmallVarcharSize = 8;
  size_t num_rows_expected           = 256;
  size_t mask_bytes                  = 0;
  size_t total_size                  = 0;
  auto const block_size              = allocator->get_block_size();
  while (true) {
    mask_bytes = sirius::utils::ceil_div_8(num_rows_expected);
    total_size = num_rows_expected * sizeof(int32_t) + mask_bytes +
                 (num_rows_expected + 1) * sizeof(int64_t) + num_rows_expected * kSmallVarcharSize +
                 mask_bytes;
    if (total_size > block_size) { break; }
    num_rows_expected *= 2;
  }

  auto allocation = allocator->allocate_multiple_blocks(total_size);
  REQUIRE(allocation != nullptr);
  REQUIRE(allocation->size() > 1);

  duckdb::LogicalType int_type(duckdb::LogicalTypeId::INTEGER);
  duckdb::LogicalType str_type(duckdb::LogicalTypeId::VARCHAR);

  duckdb_scan_task_local_state::column_builder int_builder(int_type, kSmallVarcharSize);
  duckdb_scan_task_local_state::column_builder str_builder(str_type, kSmallVarcharSize);

  size_t byte_offset = 0;
  int_builder.initialize_accessors(num_rows_expected, byte_offset, allocation);
  byte_offset += num_rows_expected * sizeof(int32_t) + mask_bytes;
  str_builder.initialize_accessors(num_rows_expected, byte_offset, allocation);

  duckdb::Vector int_vec(int_type, num_rows_expected);
  auto* int_data = reinterpret_cast<int32_t*>(int_vec.GetData());
  duckdb::ValidityMask int_validity(num_rows_expected);
  int_validity.Initialize(num_rows_expected);
  int_validity.SetAllValid(num_rows_expected);

  duckdb::Vector str_vec(str_type, num_rows_expected);
  auto* str_data = reinterpret_cast<duckdb::string_t*>(str_vec.GetData());
  duckdb::ValidityMask str_validity(num_rows_expected);
  str_validity.Initialize(num_rows_expected);
  str_validity.SetAllValid(num_rows_expected);

  std::vector<std::string> all_strings(num_rows_expected);
  std::vector<bool> all_str_valid(num_rows_expected, true);
  for (size_t i = 0; i < num_rows_expected; ++i) {
    size_t len     = 12 + (i % 5);
    all_strings[i] = std::string(len, static_cast<char>('a' + (i % 26)));
    str_data[i]    = duckdb::string_t(all_strings[i]);
    int_data[i]    = static_cast<int32_t>(i);
    if (i % 11 == 0) {
      str_validity.SetInvalid(i);
      all_str_valid[i] = false;
    }
    if (i % 13 == 0) { int_validity.SetInvalid(i); }
  }

  size_t allocated_bytes = str_builder.total_data_bytes_allocated;
  size_t used_bytes      = 0;
  size_t rows_fit        = 0;
  for (size_t i = 0; i < num_rows_expected; ++i) {
    size_t row_bytes = all_str_valid[i] ? all_strings[i].size() : 0;
    if (used_bytes + row_bytes > allocated_bytes) { break; }
    used_bytes += row_bytes;
    ++rows_fit;
  }

  REQUIRE(rows_fit > 0);
  REQUIRE(rows_fit < num_rows_expected);

  int_builder.process_column(int_vec, int_validity, rows_fit, 0, allocation);
  str_builder.process_column(str_vec, str_validity, rows_fit, 0, allocation);

  std::vector<int32_t> expected_int(rows_fit);
  std::vector<bool> expected_int_valid(rows_fit, true);
  std::vector<std::string> expected_str(rows_fit);
  std::vector<bool> expected_str_valid(rows_fit, true);
  for (size_t i = 0; i < rows_fit; ++i) {
    expected_int[i]       = static_cast<int32_t>(i);
    expected_int_valid[i] = (i % 13 != 0);
    expected_str[i]       = all_strings[i];
    expected_str_valid[i] = all_str_valid[i];
  }

  std::vector<metadata_node> column_metadata;
  column_metadata.reserve(2);
  column_metadata.push_back(int_builder.make_metadata_node(rows_fit));
  column_metadata.push_back(str_builder.make_metadata_node(rows_fit));
  auto metadata = std::make_unique<std::vector<uint8_t>>(pack_metadata_from_nodes(column_metadata));

  auto const sz         = allocation->size_bytes();
  auto table_allocation = std::make_unique<cucascade::memory::host_table_allocation>(
    std::move(allocation), std::move(metadata), sz);
  auto host_table =
    std::make_unique<cucascade::host_table_representation>(std::move(table_allocation), host_space);
  auto batch =
    std::make_shared<cucascade::data_batch>(sirius::get_next_batch_id(), std::move(host_table));

  auto& registry = sirius::converter_registry::get();
  batch->convert_to<cucascade::gpu_table_representation>(
    registry, gpu_space, cudf::get_default_stream());

  cudf::table_view table_view = sirius::get_cudf_table_view(*batch);
  REQUIRE(table_view.num_rows() == static_cast<cudf::size_type>(rows_fit));
  REQUIRE(table_view.num_columns() == 2);
  REQUIRE(table_view.column(0).type().id() == cudf::type_id::INT32);
  REQUIRE(table_view.column(1).type().id() == cudf::type_id::STRING);

  verify_numeric_column<int32_t>(table_view.column(0), expected_int, expected_int_valid);
  verify_string_column(table_view.column(1), expected_str, expected_str_valid);
}

TEST_CASE("host_table_utils - metadata offsets match packed data", "[memory][host_table_utils]")
{
  constexpr size_t num_rows = 257;
  auto* gpu_space           = get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = gpu_space->get_default_allocator();

  std::vector<cudf::data_type> column_types{cudf::data_type{cudf::type_id::INT32},
                                            cudf::data_type{cudf::type_id::INT64},
                                            cudf::data_type{cudf::type_id::STRING}};
  std::vector<std::optional<std::pair<int, int>>> ranges{
    std::make_pair(0, 100), std::make_pair(1000, 2000), std::make_pair(0, 100)};

  auto table =
    sirius::create_cudf_table_with_random_data(num_rows, column_types, ranges, stream, mr, true);
  auto int64_nulls  = build_null_indices(num_rows, {0, 7, 8, 63, 255});
  auto string_nulls = build_null_indices(num_rows, {1, 9, 64, 128, 256});
  apply_null_mask(table->get_column(1), int64_nulls, stream, mr);
  apply_null_mask(table->get_column(2), string_nulls, stream, mr);
  auto batch = sirius::make_data_batch(std::move(table), *gpu_space);

  expected_table_data expected;
  std::vector<uint8_t> expected_int64_mask;
  std::vector<uint8_t> expected_string_mask;
  {
    REQUIRE(batch->try_to_create_task());
    auto lock_result = batch->try_to_lock_for_processing(gpu_space->get_id());
    REQUIRE(lock_result.success);
    auto handle = std::move(lock_result.handle);

    auto const gpu_view  = sirius::get_cudf_table_view(*batch);
    expected             = extract_expected_data(gpu_view);
    expected_int64_mask  = extract_mask_bytes(gpu_view.column(1));
    expected_string_mask = extract_mask_bytes(gpu_view.column(2));
  }

  auto const& host_table = convert_to_host_table(batch);
  auto const& host_alloc = host_table.get_host_table();
  auto const& allocation = host_alloc->allocation;

  auto metadata_nodes = sirius::unpack_metadata_to_nodes(host_alloc->metadata);
  REQUIRE(metadata_nodes.size() == column_types.size());

  REQUIRE(metadata_nodes[0].null_count == 0);
  REQUIRE(metadata_nodes[0].null_mask_offset < 0);
  REQUIRE(metadata_nodes[1].null_count == static_cast<cudf::size_type>(int64_nulls.size()));
  REQUIRE(metadata_nodes[1].null_mask_offset >= 0);
  REQUIRE(metadata_nodes[2].null_count == static_cast<cudf::size_type>(string_nulls.size()));
  REQUIRE(metadata_nodes[2].null_mask_offset >= 0);

  sirius::memory::multiple_blocks_allocation_accessor<int32_t> int32_accessor;
  int32_accessor.initialize(static_cast<size_t>(metadata_nodes[0].data_offset), allocation);
  std::vector<int32_t> actual_int32(num_rows);
  if (num_rows > 0) {
    int32_accessor.memcpy_to(allocation, actual_int32.data(), sizeof(int32_t) * num_rows);
  }
  REQUIRE(actual_int32 == expected.int32_values);

  sirius::memory::multiple_blocks_allocation_accessor<int64_t> int64_accessor;
  int64_accessor.initialize(static_cast<size_t>(metadata_nodes[1].data_offset), allocation);
  std::vector<int64_t> actual_int64(num_rows);
  if (num_rows > 0) {
    int64_accessor.memcpy_to(allocation, actual_int64.data(), sizeof(int64_t) * num_rows);
  }
  REQUIRE(actual_int64 == expected.int64_values);

  sirius::memory::multiple_blocks_allocation_accessor<uint8_t> mask_accessor;
  if (!expected_int64_mask.empty()) {
    std::vector<uint8_t> actual_int64_mask(expected_int64_mask.size(), 0);
    mask_accessor.initialize(static_cast<size_t>(metadata_nodes[1].null_mask_offset), allocation);
    mask_accessor.memcpy_to(allocation, actual_int64_mask.data(), actual_int64_mask.size());
    mask_unused_bits(actual_int64_mask, num_rows);
    REQUIRE(actual_int64_mask == expected_int64_mask);
  }

  auto const& string_node = metadata_nodes[2];
  REQUIRE(string_node.children.size() == 1);
  REQUIRE(string_node.children[0].type.id() == cudf::type_id::INT64);

  sirius::memory::multiple_blocks_allocation_accessor<int64_t> offset_accessor;
  offset_accessor.initialize(static_cast<size_t>(string_node.children[0].data_offset), allocation);
  std::vector<int64_t> actual_offsets(num_rows + 1);
  if (num_rows > 0) {
    offset_accessor.memcpy_to(allocation, actual_offsets.data(), sizeof(int64_t) * (num_rows + 1));
  }
  REQUIRE(actual_offsets == expected.string_offsets);

  sirius::memory::multiple_blocks_allocation_accessor<uint8_t> chars_accessor;
  chars_accessor.initialize(static_cast<size_t>(string_node.data_offset), allocation);
  std::vector<char> actual_chars(expected.string_chars.size());
  if (!actual_chars.empty()) {
    chars_accessor.memcpy_to(allocation, actual_chars.data(), actual_chars.size());
  }
  REQUIRE(actual_chars == expected.string_chars);

  if (!expected_string_mask.empty()) {
    std::vector<uint8_t> actual_string_mask(expected_string_mask.size(), 0);
    mask_accessor.initialize(static_cast<size_t>(string_node.null_mask_offset), allocation);
    mask_accessor.memcpy_to(allocation, actual_string_mask.data(), actual_string_mask.size());
    mask_unused_bits(actual_string_mask, num_rows);
    REQUIRE(actual_string_mask == expected_string_mask);
  }
}
