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
#include <op/scan/duckdb_scan_task.hpp>

// cucascade
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/host_table.hpp>

// cudf
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

// duckdb
#include <duckdb.hpp>
#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/types/vector.hpp>

// cuda
#include <cuda_runtime.h>

// standard library
#include <filesystem>
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
