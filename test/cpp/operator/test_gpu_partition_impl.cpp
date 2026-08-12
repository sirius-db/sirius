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

#include "catch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "operator/operator_test_utils.hpp"
#include "scan/test_utils.hpp"
#include "utils/utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace sirius;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::op;

namespace {

memory_space* get_shared_mem_space()
{
  static auto manager = initialize_memory_manager();
  return manager->get_memory_space(Tier::GPU, 0);
}

/**
 * @brief Create a batch with random data in idle state.
 *
 * @return A shared_ptr to the newly created data_batch (idle state, ready for reading).
 */
std::shared_ptr<data_batch> create_batch_with_random_data(
  const int num_rows,
  const std::vector<cudf::data_type>& column_types,
  std::vector<std::optional<std::pair<int, int>>>& ranges,
  memory_space& mem_space)
{
  // Base input batches, make value ranges small so that we have duplicated partition keys
  for (size_t i = 0; i < ranges.size(); ++i) {
    if (!ranges[i].has_value()) { ranges[i] = {0, 4}; }
  }
  auto table = create_cudf_table_with_random_data(
    num_rows, column_types, ranges, cudf::get_default_stream(), mem_space.get_default_allocator());
  return sirius::make_data_batch(std::move(table),
                                 mem_space,
                                 cudf::get_default_stream(),
                                 sirius::telemetry::batch_telemetry_info{});
}

void copy_data_to_host_by_rows(cudf::table_view table, std::vector<std::vector<int64_t>>& h_rows)
{
  std::vector<std::vector<int64_t>> h_cols(table.num_columns());
  for (int c = 0; c < table.num_columns(); ++c) {
    const auto& col = table.column(c);
    switch (col.type().id()) {
      case cudf::type_id::INT32: {
        std::vector<int32_t> h_buf(table.num_rows());
        cudaMemcpy(h_buf.data(),
                   col.data<int32_t>(),
                   sizeof(int32_t) * table.num_rows(),
                   cudaMemcpyDeviceToHost);
        for (auto val : h_buf) {
          h_cols[c].push_back(val);
        }
        break;
      }
      case cudf::type_id::INT64: {
        std::vector<int64_t> h_buf(table.num_rows());
        cudaMemcpy(h_buf.data(),
                   col.data<int64_t>(),
                   sizeof(int64_t) * table.num_rows(),
                   cudaMemcpyDeviceToHost);
        for (auto val : h_buf) {
          h_cols[c].push_back(val);
        }
        break;
      }
      default:
        throw std::runtime_error("Unsupported cudf::data_type in `pull_data_to_host()`: " +
                                 std::to_string(static_cast<int>(col.type().id())));
    }
  }
  for (int r = 0; r < table.num_rows(); ++r) {
    h_rows.emplace_back(table.num_columns());
    auto& row = h_rows.back();
    for (int c = 0; c < table.num_columns(); ++c) {
      row[c] = h_cols[c][r];
    }
  }
}

void validate_hash_partition(data_batch& input_batch,
                             const std::vector<std::shared_ptr<data_batch>>& output_batches,
                             int num_partitions)
{
  cudf::table_view input_table_view = sirius::get_cudf_table_view(input_batch);
  std::vector<cudf::table_view> output_table_views;
  for (const auto& output_batch : output_batches) {
    output_table_views.push_back(sirius::get_cudf_table_view(*output_batch));
  }

  // Check metadata
  REQUIRE(output_batches.size() == static_cast<size_t>(num_partitions));
  int actual_num_rows = 0;
  for (const auto& output_table : output_table_views) {
    actual_num_rows += output_table.num_rows();
    REQUIRE(output_table.num_columns() == input_table_view.num_columns());
    for (int c = 0; c < output_table.num_columns(); ++c) {
      REQUIRE(output_table.column(c).type().id() == input_table_view.column(c).type().id());
    }
  }
  REQUIRE(actual_num_rows == input_table_view.num_rows());

  // Check data
  std::vector<std::vector<int64_t>> h_input_rows;
  copy_data_to_host_by_rows(input_table_view, h_input_rows);
  std::vector<std::vector<std::vector<int64_t>>> h_output_rows(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    copy_data_to_host_by_rows(output_table_views[i], h_output_rows[i]);
  }

  std::multiset<std::vector<int64_t>> output_set;
  for (const auto& partition : h_output_rows) {
    for (const auto& row : partition) {
      REQUIRE(!output_set.contains(row));
    }
    output_set.insert(partition.begin(), partition.end());
  }
  std::multiset<std::vector<int64_t>> input_set(h_input_rows.begin(), h_input_rows.end());
  REQUIRE(input_set == output_set);
}
using mixed_row     = std::tuple<int32_t, std::string, int64_t>;
using nullable_row  = std::tuple<bool, int32_t, int64_t>;
using mixed_rows    = std::multiset<mixed_row>;
using nullable_rows = std::multiset<nullable_row>;
using numeric_rows  = std::multiset<std::vector<int64_t>>;
using string_rows   = std::multiset<std::string>;

std::shared_ptr<data_batch> make_mixed_batch(memory_space& space,
                                             const std::vector<int32_t>& keys,
                                             const std::vector<std::string>& strings,
                                             const std::vector<int64_t>& payload)
{
  using namespace sirius::test::operator_utils;
  auto key_batch = make_numeric_batch(space, keys, cudf::type_id::INT32);
  auto str_batch = make_string_batch(space, strings);
  auto val_batch = make_numeric_batch(space, payload, cudf::type_id::INT64);
  return concatenate_batches_horizontal({key_batch, str_batch, val_batch}, space);
}

// STRING offsets may be INT32 or INT64 after converter round-trips. Normalize either width to
// `std::int64_t`; this reader intentionally requires a canonical view whose first offset is zero.
std::vector<std::string> copy_strings_to_host(cudf::column_view const& column)
{
  std::vector<std::string> strings;
  if (column.size() == 0) { return strings; }

  cudf::strings_column_view string_column{column};
  auto const offsets_view = string_column.offsets();
  auto const offset_count = static_cast<std::size_t>(column.size()) + 1;
  std::vector<std::int64_t> offsets(offset_count);
  if (offsets_view.type().id() == cudf::type_id::INT32) {
    std::vector<cudf::size_type> narrow_offsets(offset_count);
    REQUIRE(cudaMemcpy(narrow_offsets.data(),
                       offsets_view.data<cudf::size_type>(),
                       offset_count * sizeof(cudf::size_type),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    std::copy(narrow_offsets.begin(), narrow_offsets.end(), offsets.begin());
  } else {
    REQUIRE(offsets_view.type().id() == cudf::type_id::INT64);
    REQUIRE(cudaMemcpy(offsets.data(),
                       offsets_view.data<std::int64_t>(),
                       offset_count * sizeof(std::int64_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
  }

  REQUIRE(offsets.front() == 0);
  REQUIRE(offsets.back() >= 0);
  std::vector<char> chars(static_cast<std::size_t>(offsets.back()));
  if (!chars.empty()) {
    REQUIRE(cudaMemcpy(chars.data(),
                       string_column.chars_begin(cudf::get_default_stream()),
                       chars.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
  }

  strings.reserve(static_cast<std::size_t>(column.size()));
  for (cudf::size_type row = 0; row < column.size(); ++row) {
    auto const begin = offsets[static_cast<std::size_t>(row)];
    auto const end   = offsets[static_cast<std::size_t>(row) + 1];
    REQUIRE(begin >= 0);
    REQUIRE(end >= begin);
    REQUIRE(static_cast<std::size_t>(end) <= chars.size());
    if (begin == end) {
      strings.emplace_back();
    } else {
      strings.emplace_back(chars.data() + begin, static_cast<std::size_t>(end - begin));
    }
  }
  return strings;
}

mixed_rows copy_mixed_rows(cudf::table_view table)
{
  using sirius::test::operator_utils::copy_column_to_host;
  auto const keys    = copy_column_to_host<int32_t>(table.column(0));
  auto const strings = copy_strings_to_host(table.column(1));
  auto const payload = copy_column_to_host<int64_t>(table.column(2));

  mixed_rows rows;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    rows.emplace(keys[i], strings[i], payload[i]);
  }
  return rows;
}

numeric_rows copy_numeric_rows(cudf::table_view table)
{
  std::vector<std::vector<int64_t>> rows;
  copy_data_to_host_by_rows(table, rows);
  return numeric_rows(rows.begin(), rows.end());
}

string_rows copy_string_rows(cudf::table_view table)
{
  auto values = copy_strings_to_host(table.column(0));
  return string_rows(values.begin(), values.end());
}

// Null payload bytes are unspecified, so preserve validity separately and normalize invalid values
// before multiset comparison.
nullable_rows copy_nullable_rows(cudf::table_view table)
{
  using sirius::test::operator_utils::copy_column_to_host;
  using sirius::test::operator_utils::copy_validity_to_host;
  auto const nullable_values = copy_column_to_host<int32_t>(table.column(0));
  auto const validity        = copy_validity_to_host(table.column(0));
  auto const keys            = copy_column_to_host<int64_t>(table.column(1));

  nullable_rows rows;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    rows.emplace(validity[i], validity[i] ? nullable_values[i] : 0, keys[i]);
  }
  return rows;
}

// Use cuDF's hash result as the partition-membership oracle. Materialize each slice to rebase
// nested offsets, and compare multisets because intra-partition row order is unspecified.
template <typename Rows>
std::vector<Rows> reference_partition_rows(cudf::table_view input,
                                           cudf::table_view keys,
                                           int num_partitions,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr,
                                           Rows (*copy_rows)(cudf::table_view))
{
  auto reference = cudf::hash_partition(
    input, keys, num_partitions, cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, stream, mr);
  REQUIRE(reference.second.size() == static_cast<std::size_t>(num_partitions) + 1);

  std::vector<cudf::size_type> slice_indices;
  slice_indices.reserve(static_cast<std::size_t>(num_partitions) * 2);
  for (int i = 0; i < num_partitions; ++i) {
    slice_indices.push_back(reference.second[static_cast<std::size_t>(i)]);
    slice_indices.push_back(reference.second[static_cast<std::size_t>(i) + 1]);
  }

  auto views = cudf::slice(reference.first->view(), slice_indices, stream);
  std::vector<Rows> rows;
  rows.reserve(static_cast<std::size_t>(num_partitions));
  for (auto const& view : views) {
    cudf::table canonical{view, stream, mr};
    rows.push_back(copy_rows(canonical.view()));
  }
  return rows;
}

void require_canonical_schema(cudf::table_view actual, cudf::table_view expected)
{
  REQUIRE(actual.num_columns() == expected.num_columns());
  for (cudf::size_type column = 0; column < actual.num_columns(); ++column) {
    REQUIRE(actual.column(column).type() == expected.column(column).type());
    REQUIRE(actual.column(column).offset() == 0);
  }
}

// Nonempty fixed-width partitions must be consecutive offset-zero aliases into each fixed column's
// combined allocation. Batch sizes are logical slice bytes; mixed batches add copied column
// storage.
void require_fixed_aliases_and_sizes(const std::vector<std::shared_ptr<data_batch>>& batches,
                                     const std::vector<cudf::size_type>& fixed_indices,
                                     bool all_columns_are_fixed)
{
  std::vector<std::byte const*> bases(fixed_indices.size(), nullptr);
  cudf::size_type preceding_rows = 0;

  for (auto const& batch : batches) {
    auto ro                 = batch->to_read_only();
    auto const view         = sirius::get_cudf_table_view(ro);
    std::size_t fixed_bytes = 0;

    for (cudf::size_type column = 0; column < view.num_columns(); ++column) {
      REQUIRE(view.column(column).offset() == 0);
    }
    for (std::size_t i = 0; i < fixed_indices.size(); ++i) {
      auto const column = view.column(fixed_indices[i]);
      auto const width  = cudf::size_of(column.type());
      fixed_bytes += static_cast<std::size_t>(view.num_rows()) * width;
      if (view.num_rows() == 0) { continue; }

      auto const* head = static_cast<std::byte const*>(column.head());
      if (bases[i] == nullptr) { bases[i] = head; }
      REQUIRE(head == bases[i] + static_cast<std::size_t>(preceding_rows) * width);
    }

    if (all_columns_are_fixed) {
      REQUIRE(ro.get_data()->get_size_in_bytes() == fixed_bytes);
    } else {
      REQUIRE(ro.get_data()->get_size_in_bytes() >= fixed_bytes);
    }
    preceding_rows += view.num_rows();
  }
}

using fixed_rows = std::multiset<std::vector<std::uint8_t>>;
using list_rows  = std::multiset<std::vector<int32_t>>;

std::vector<cudf::data_type> copy_schema(cudf::table_view table)
{
  std::vector<cudf::data_type> types;
  types.reserve(static_cast<std::size_t>(table.num_columns()));
  for (auto const& column : table) {
    types.push_back(column.type());
  }
  return types;
}

void require_canonical_column_tree(cudf::column_view const& column)
{
  REQUIRE(column.offset() == 0);
  for (auto child = column.child_begin(); child != column.child_end(); ++child) {
    require_canonical_column_tree(*child);
  }
}

void require_schema_and_layout(cudf::table_view table, std::vector<cudf::data_type> const& schema)
{
  REQUIRE(table.num_columns() == static_cast<cudf::size_type>(schema.size()));
  for (cudf::size_type column = 0; column < table.num_columns(); ++column) {
    REQUIRE(table.column(column).type() == schema[static_cast<std::size_t>(column)]);
    require_canonical_column_tree(table.column(column));
  }
}

template <typename T>
std::unique_ptr<cudf::column> make_fixed_test_column(cudf::data_type type,
                                                     std::vector<T> const& values,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  auto column = cudf::make_fixed_width_column(
    type, static_cast<cudf::size_type>(values.size()), cudf::mask_state::UNALLOCATED, stream, mr);
  if (!values.empty()) {
    auto const status = cudaMemcpyAsync(column->mutable_view().template data<T>(),
                                        values.data(),
                                        values.size() * sizeof(T),
                                        cudaMemcpyHostToDevice,
                                        stream.value());
    REQUIRE(status == cudaSuccess);
  }
  return column;
}

// Compare heterogeneous fixed-width rows by raw storage bytes so boolean, temporal, decimal, and
// 128-bit payloads share one lossless representation; callers verify type metadata separately.
fixed_rows copy_fixed_rows(cudf::table_view table)
{
  std::vector<std::vector<std::uint8_t>> columns;
  columns.reserve(static_cast<std::size_t>(table.num_columns()));
  std::size_t row_width = 0;
  for (auto const& column : table) {
    REQUIRE(cudf::is_fixed_width(column.type()));
    REQUIRE(column.num_children() == 0);
    auto const width = cudf::size_of(column.type());
    row_width += width;
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(column.size()) * width);
    if (!bytes.empty()) {
      auto const* source = static_cast<std::byte const*>(column.head()) +
                           static_cast<std::size_t>(column.offset()) * width;
      REQUIRE(cudaMemcpy(bytes.data(), source, bytes.size(), cudaMemcpyDeviceToHost) ==
              cudaSuccess);
    }
    columns.push_back(std::move(bytes));
  }

  fixed_rows rows;
  for (cudf::size_type row = 0; row < table.num_rows(); ++row) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(row_width);
    for (cudf::size_type column = 0; column < table.num_columns(); ++column) {
      auto const width   = cudf::size_of(table.column(column).type());
      auto const& source = columns[static_cast<std::size_t>(column)];
      auto const begin   = source.begin() + static_cast<std::size_t>(row) * width;
      bytes.insert(bytes.end(), begin, begin + width);
    }
    rows.insert(std::move(bytes));
  }
  return rows;
}

list_rows copy_list_rows(cudf::table_view table)
{
  REQUIRE(table.num_columns() == 1);
  auto const column = table.column(0);
  REQUIRE(column.type().id() == cudf::type_id::LIST);
  list_rows rows;
  if (column.size() == 0) { return rows; }

  cudf::lists_column_view lists{column};
  auto const offsets =
    sirius::test::operator_utils::copy_column_to_host<cudf::size_type>(lists.offsets());
  auto const values = sirius::test::operator_utils::copy_column_to_host<int32_t>(lists.child());
  REQUIRE(offsets.size() == static_cast<std::size_t>(column.size()) + 1);
  for (cudf::size_type row = 0; row < column.size(); ++row) {
    auto const begin = static_cast<std::size_t>(offsets[static_cast<std::size_t>(row)]);
    auto const end   = static_cast<std::size_t>(offsets[static_cast<std::size_t>(row) + 1]);
    rows.emplace(values.begin() + begin, values.begin() + end);
  }
  return rows;
}

std::shared_ptr<data_batch> make_list_batch(memory_space& space,
                                            std::vector<cudf::size_type> const& offsets,
                                            std::vector<int32_t> const& values)
{
  REQUIRE(offsets.size() >= 2);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space.get_default_allocator();
  auto offsets_column =
    make_fixed_test_column(cudf::data_type{cudf::type_id::INT32}, offsets, stream, mr);
  auto values_column =
    make_fixed_test_column(cudf::data_type{cudf::type_id::INT32}, values, stream, mr);
  auto list_column = cudf::make_lists_column(static_cast<cudf::size_type>(offsets.size() - 1),
                                             std::move(offsets_column),
                                             std::move(values_column),
                                             0,
                                             rmm::device_buffer{});
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(list_column));
  return make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                         space,
                         stream,
                         telemetry::batch_telemetry_info{});
}

}  // namespace

TEST_CASE("Hash partition basic", "[operator][hash_partition]")
{
  auto* mem_space                           = get_shared_mem_space();
  constexpr size_t num_input_rows           = 100;
  constexpr size_t num_partitions           = 4;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> partition_key_idx        = {0, 1};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  auto input_batch =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  std::vector<std::shared_ptr<data_batch>> output_batches;
  {
    auto ro        = input_batch->to_read_only();
    output_batches = gpu_partition_impl::hash_partition(
      ro, partition_key_idx, num_partitions, cudf::get_default_stream(), *mem_space);
  }
  validate_hash_partition(*input_batch, output_batches, num_partitions);
}

TEST_CASE("Hash partition with invalid input", "[operator][hash_partition]")
{
  auto* mem_space                           = get_shared_mem_space();
  constexpr size_t num_input_rows           = 100;
  constexpr size_t num_partitions           = 1;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> partition_key_idx        = {0, 1};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  auto input_batch =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  {
    auto ro = input_batch->to_read_only();
    REQUIRE_THROWS_AS(
      gpu_partition_impl::hash_partition(
        ro, partition_key_idx, num_partitions, cudf::get_default_stream(), *mem_space),
      std::runtime_error);
  }
}

TEST_CASE("Hash partition validates key metadata", "[operator][hash_partition]")
{
  auto* mem_space                    = get_shared_mem_space();
  std::vector<cudf::data_type> types = {cudf::data_type{cudf::type_id::INT32},
                                        cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges(types.size(), std::pair{0, 4});
  auto input = create_batch_with_random_data(8, types, ranges, *mem_space);
  auto ro    = input->to_read_only();

  std::vector<int> empty_keys;
  REQUIRE_THROWS_AS(
    gpu_partition_impl::hash_partition(ro, empty_keys, 2, cudf::get_default_stream(), *mem_space),
    std::invalid_argument);

  std::vector<int> two_keys{0, 1};
  std::vector<cudf::data_type> one_cast{cudf::data_type{cudf::type_id::EMPTY}};
  REQUIRE_THROWS_AS(gpu_partition_impl::hash_partition(
                      ro, two_keys, one_cast, 2, cudf::get_default_stream(), *mem_space),
                    std::invalid_argument);

  std::vector<int> negative_key{-1};
  REQUIRE_THROWS_AS(
    gpu_partition_impl::hash_partition(ro, negative_key, 2, cudf::get_default_stream(), *mem_space),
    std::out_of_range);

  std::vector<int> past_end_key{2};
  REQUIRE_THROWS_AS(
    gpu_partition_impl::hash_partition(ro, past_end_key, 2, cudf::get_default_stream(), *mem_space),
    std::out_of_range);
}

TEST_CASE("Hash partition with empty input", "[operator][hash_partition]")
{
  auto* mem_space                           = get_shared_mem_space();
  constexpr size_t num_input_rows           = 0;
  constexpr size_t num_partitions           = 4;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> partition_key_idx        = {0, 1};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  auto input_batch =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  std::vector<std::shared_ptr<data_batch>> output_batches;
  {
    auto ro        = input_batch->to_read_only();
    output_batches = gpu_partition_impl::hash_partition(
      ro, partition_key_idx, num_partitions, cudf::get_default_stream(), *mem_space);
  }
  validate_hash_partition(*input_batch, output_batches, num_partitions);
}

TEST_CASE("Hash partition with all the same partitioning keys", "[operator][hash_partition]")
{
  auto* mem_space                                        = get_shared_mem_space();
  constexpr size_t num_input_rows                        = 100;
  constexpr size_t num_partitions                        = 4;
  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::INT32},
                                                            cudf::data_type{cudf::type_id::INT64},
                                                            cudf::data_type{cudf::type_id::INT32},
                                                            cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> partition_key_idx                     = {0, 1};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>({0, 0}),
    std::optional<std::pair<int, int>>({1, 1}),
    std::nullopt,
    std::nullopt};

  auto input_batch =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  std::vector<std::shared_ptr<data_batch>> output_batches;
  {
    auto ro        = input_batch->to_read_only();
    output_batches = gpu_partition_impl::hash_partition(
      ro, partition_key_idx, num_partitions, cudf::get_default_stream(), *mem_space);
  }
  validate_hash_partition(*input_batch, output_batches, num_partitions);
}

TEST_CASE("Hash partition with num partitions larger than input size", "[operator][hash_partition]")
{
  auto* mem_space                           = get_shared_mem_space();
  constexpr size_t num_input_rows           = 10;
  constexpr size_t num_partitions           = 20;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> partition_key_idx        = {0, 1};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  auto input_batch =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  std::vector<std::shared_ptr<data_batch>> output_batches;
  {
    auto ro        = input_batch->to_read_only();
    output_batches = gpu_partition_impl::hash_partition(
      ro, partition_key_idx, num_partitions, cudf::get_default_stream(), *mem_space);
  }
  validate_hash_partition(*input_batch, output_batches, num_partitions);
  std::size_t empty_count = 0;
  for (auto const& batch : output_batches) {
    auto ro = batch->to_read_only();
    if (sirius::get_cudf_table_view(ro).num_rows() == 0) {
      ++empty_count;
      REQUIRE(ro.get_data()->get_size_in_bytes() == 0);
    }
  }
  REQUIRE(empty_count >= num_partitions - num_input_rows);
}

TEST_CASE("Hash partition aliases fixed columns in a mixed schema",
          "[operator][hash_partition][zero_copy]")
{
  auto* mem_space             = get_shared_mem_space();
  constexpr int partitions    = 4;
  std::vector<int32_t> keys   = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int64_t> values = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111};
  std::vector<std::string> strings;
  strings.reserve(keys.size());
  for (auto key : keys) {
    strings.push_back("row_" + std::to_string(key));
  }

  auto input      = make_mixed_batch(*mem_space, keys, strings, values);
  auto input_ro   = input->to_read_only();
  auto input_view = sirius::get_cudf_table_view(input_ro);
  std::vector<cudf::column_view> key_views{input_view.column(0)};
  auto expected = reference_partition_rows<mixed_rows>(input_view,
                                                       cudf::table_view{key_views},
                                                       partitions,
                                                       cudf::get_default_stream(),
                                                       mem_space->get_default_allocator(),
                                                       copy_mixed_rows);

  auto output = gpu_partition_impl::hash_partition(
    input_ro, {0}, partitions, cudf::get_default_stream(), *mem_space);
  REQUIRE(output.size() == static_cast<std::size_t>(partitions));
  require_fixed_aliases_and_sizes(output, {0, 2}, false);

  mixed_rows all_output;
  for (std::size_t partition = 0; partition < output.size(); ++partition) {
    auto ro         = output[partition]->to_read_only();
    auto const view = sirius::get_cudf_table_view(ro);
    require_canonical_schema(view, input_view);
    auto expected_bytes =
      static_cast<std::size_t>(view.num_rows()) * (sizeof(int32_t) + sizeof(int64_t));
    if (view.num_rows() != 0) {
      cudf::column copied_string{
        view.column(1), cudf::get_default_stream(), mem_space->get_default_allocator()};
      expected_bytes += copied_string.alloc_size();
    }
    REQUIRE(ro.get_data()->get_size_in_bytes() == expected_bytes);

    auto const actual = copy_mixed_rows(view);
    REQUIRE(actual == expected[partition]);
    all_output.insert(actual.begin(), actual.end());
  }
  REQUIRE(all_output == copy_mixed_rows(input_view));
}

TEST_CASE("Hash partition fixed-only batches have zero-copy logical sizes",
          "[operator][hash_partition][zero_copy]")
{
  auto* mem_space                                        = get_shared_mem_space();
  std::vector<cudf::data_type> types                     = {cudf::data_type{cudf::type_id::INT32},
                                                            cudf::data_type{cudf::type_id::INT64},
                                                            cudf::data_type{cudf::type_id::INT32}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::pair{0, 15}, std::pair{100, 200}, std::pair{300, 400}};
  auto input = create_batch_with_random_data(64, types, ranges, *mem_space);

  std::vector<std::shared_ptr<data_batch>> output;
  {
    auto ro = input->to_read_only();
    output = gpu_partition_impl::hash_partition(ro, {0}, 4, cudf::get_default_stream(), *mem_space);
  }

  require_fixed_aliases_and_sizes(output, {0, 1, 2}, true);
  numeric_rows rows;
  for (auto const& batch : output) {
    auto ro         = batch->to_read_only();
    auto const part = copy_numeric_rows(sirius::get_cudf_table_view(ro));
    rows.insert(part.begin(), part.end());
  }
  REQUIRE(rows == copy_numeric_rows(sirius::get_cudf_table_view(*input)));
}

TEST_CASE("Hash partition string-only fallback owns canonical partitions",
          "[operator][hash_partition][fallback]")
{
  auto* mem_space                 = get_shared_mem_space();
  std::vector<std::string> values = {"zero", "one", "two", "three", "four", "five", "six", "seven"};
  auto input = sirius::test::operator_utils::make_string_batch(*mem_space, values);

  std::vector<string_rows> expected;
  std::vector<std::shared_ptr<data_batch>> output;
  {
    auto ro               = input->to_read_only();
    auto const input_view = sirius::get_cudf_table_view(ro);
    std::vector<cudf::column_view> key_views{input_view.column(0)};
    expected = reference_partition_rows<string_rows>(input_view,
                                                     cudf::table_view{key_views},
                                                     4,
                                                     cudf::get_default_stream(),
                                                     mem_space->get_default_allocator(),
                                                     copy_string_rows);
    output = gpu_partition_impl::hash_partition(ro, {0}, 4, cudf::get_default_stream(), *mem_space);
  }

  auto const survivor_it = std::find_if(output.begin(), output.end(), [](auto const& batch) {
    auto ro = batch->to_read_only();
    return sirius::get_cudf_table_view(ro).num_rows() != 0;
  });
  REQUIRE(survivor_it != output.end());
  auto const survivor_index = static_cast<std::size_t>(std::distance(output.begin(), survivor_it));
  auto survivor             = *survivor_it;

  string_rows all_output;
  for (auto const& batch : output) {
    auto ro         = batch->to_read_only();
    auto const view = sirius::get_cudf_table_view(ro);
    REQUIRE(view.num_columns() == 1);
    REQUIRE(view.column(0).type() == cudf::data_type{cudf::type_id::STRING});
    REQUIRE(view.column(0).offset() == 0);
    auto const rows = copy_string_rows(view);
    all_output.insert(rows.begin(), rows.end());
  }

  output.clear();
  input.reset();
  auto survivor_ro         = survivor->to_read_only();
  auto const survivor_view = sirius::get_cudf_table_view(survivor_ro);
  REQUIRE(survivor_view.column(0).offset() == 0);
  REQUIRE(copy_string_rows(survivor_view) == expected[survivor_index]);
  string_rows const expected_all(values.begin(), values.end());
  REQUIRE(all_output == expected_all);
}

TEST_CASE("Hash partition copies nullable fixed columns but aliases eligible siblings",
          "[operator][hash_partition][fallback]")
{
  using namespace sirius::test::operator_utils;
  auto* mem_space                      = get_shared_mem_space();
  std::vector<int32_t> nullable_values = {10, 11, 12, 13, 14, 15, 16, 17};
  std::vector<bool> validity           = {true, false, true, true, false, true, false, true};
  std::vector<int64_t> keys            = {0, 1, 2, 3, 4, 5, 6, 7};
  auto nullable =
    make_numeric_batch_with_nulls(*mem_space, nullable_values, validity, cudf::type_id::INT32);
  auto key   = make_numeric_batch(*mem_space, keys, cudf::type_id::INT64);
  auto input = concatenate_batches_horizontal({nullable, key}, *mem_space);

  auto input_ro         = input->to_read_only();
  auto const input_view = sirius::get_cudf_table_view(input_ro);
  std::vector<cudf::column_view> key_views{input_view.column(1)};
  auto expected = reference_partition_rows<nullable_rows>(input_view,
                                                          cudf::table_view{key_views},
                                                          4,
                                                          cudf::get_default_stream(),
                                                          mem_space->get_default_allocator(),
                                                          copy_nullable_rows);

  auto output =
    gpu_partition_impl::hash_partition(input_ro, {1}, 4, cudf::get_default_stream(), *mem_space);
  require_fixed_aliases_and_sizes(output, {1}, false);
  for (std::size_t partition = 0; partition < output.size(); ++partition) {
    auto ro         = output[partition]->to_read_only();
    auto const view = sirius::get_cudf_table_view(ro);
    require_canonical_schema(view, input_view);
    REQUIRE(copy_nullable_rows(view) == expected[partition]);
  }
}

TEST_CASE("Hash partition cast keys stay out of the payload",
          "[operator][hash_partition][cast_key]")
{
  using namespace sirius::test::operator_utils;
  auto* mem_space = get_shared_mem_space();
  std::vector<int32_t> keys;
  std::vector<int64_t> payload;
  for (int32_t key = 0; key < 64; ++key) {
    keys.push_back(key);
    payload.push_back(1'000 + key);
  }
  auto key_batch     = make_numeric_batch(*mem_space, keys, cudf::type_id::INT32);
  auto payload_batch = make_numeric_batch(*mem_space, payload, cudf::type_id::INT64);
  auto input         = concatenate_batches_horizontal({key_batch, payload_batch}, *mem_space);

  auto input_ro         = input->to_read_only();
  auto const input_view = sirius::get_cudf_table_view(input_ro);
  auto cast_key         = cudf::cast(input_view.column(0),
                             cudf::data_type{cudf::type_id::INT64},
                             cudf::get_default_stream(),
                             mem_space->get_default_allocator());
  std::vector<cudf::column_view> key_views{cast_key->view()};
  auto expected = reference_partition_rows<numeric_rows>(input_view,
                                                         cudf::table_view{key_views},
                                                         4,
                                                         cudf::get_default_stream(),
                                                         mem_space->get_default_allocator(),
                                                         copy_numeric_rows);
  std::vector<cudf::column_view> original_key_views{input_view.column(0)};
  auto uncast_partitions =
    reference_partition_rows<numeric_rows>(input_view,
                                           cudf::table_view{original_key_views},
                                           4,
                                           cudf::get_default_stream(),
                                           mem_space->get_default_allocator(),
                                           copy_numeric_rows);
  REQUIRE(expected != uncast_partitions);

  auto output = gpu_partition_impl::hash_partition(input_ro,
                                                   {0},
                                                   {cudf::data_type{cudf::type_id::INT64}},
                                                   4,
                                                   cudf::get_default_stream(),
                                                   *mem_space);
  for (std::size_t partition = 0; partition < output.size(); ++partition) {
    auto ro         = output[partition]->to_read_only();
    auto const view = sirius::get_cudf_table_view(ro);
    require_canonical_schema(view, input_view);
    REQUIRE(view.num_columns() == 2);
    REQUIRE(copy_numeric_rows(view) == expected[partition]);
  }
}

TEST_CASE("View-backed hash partition materializes safely on another stream",
          "[operator][hash_partition][zero_copy][stream]")
{
  auto* mem_space = get_shared_mem_space();
  rmm::cuda_stream producer_stream{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream consumer_stream{rmm::cuda_stream::flags::non_blocking};
  std::vector<int32_t> keys   = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int64_t> values = {50, 51, 52, 53, 54, 55, 56, 57};
  std::vector<std::string> strings;
  for (auto key : keys) {
    strings.push_back("value_" + std::to_string(key));
  }
  auto input = make_mixed_batch(*mem_space, keys, strings, values);
  // Finish fixture writes on the default stream so this test isolates the output representation's
  // producer-to-consumer writer-event handoff.
  cudf::get_default_stream().synchronize();

  std::vector<mixed_rows> expected;
  std::vector<std::shared_ptr<data_batch>> output;
  {
    auto ro               = input->to_read_only();
    auto const input_view = sirius::get_cudf_table_view(ro);
    std::vector<cudf::column_view> key_views{input_view.column(0)};
    expected = reference_partition_rows<mixed_rows>(input_view,
                                                    cudf::table_view{key_views},
                                                    4,
                                                    cudf::get_default_stream(),
                                                    mem_space->get_default_allocator(),
                                                    copy_mixed_rows);
    output   = gpu_partition_impl::hash_partition(ro, {0}, 4, producer_stream.view(), *mem_space);
  }

  auto const survivor_it = std::find_if(output.begin(), output.end(), [](auto const& batch) {
    auto ro = batch->to_read_only();
    return sirius::get_cudf_table_view(ro).num_rows() != 0;
  });
  REQUIRE(survivor_it != output.end());
  auto const survivor_index = static_cast<std::size_t>(std::distance(output.begin(), survivor_it));
  auto survivor             = *survivor_it;
  output.clear();

  std::unique_ptr<cudf::table> materialized;
  {
    auto mut     = survivor->to_mutable();
    auto& repr   = mut.get_data()->cast<cucascade::gpu_table_representation>();
    materialized = repr.release_table(consumer_stream.view());
  }

  survivor.reset();
  input.reset();
  REQUIRE(materialized != nullptr);
  for (auto const& column : materialized->view()) {
    REQUIRE(column.offset() == 0);
  }
  REQUIRE(copy_mixed_rows(materialized->view()) == expected[survivor_index]);
}

TEST_CASE("View-backed hash partition clones safely on another stream",
          "[operator][hash_partition][zero_copy][stream]")
{
  auto* mem_space = get_shared_mem_space();
  rmm::cuda_stream producer_stream{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream consumer_stream{rmm::cuda_stream::flags::non_blocking};
  std::vector<int32_t> keys   = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int64_t> values = {50, 51, 52, 53, 54, 55, 56, 57};
  std::vector<std::string> strings;
  for (auto key : keys) {
    strings.push_back("value_" + std::to_string(key));
  }
  auto input = make_mixed_batch(*mem_space, keys, strings, values);
  // Finish fixture writes on the default stream so this test isolates the output representation's
  // producer-to-consumer writer-event handoff.
  cudf::get_default_stream().synchronize();

  std::vector<mixed_rows> expected;
  std::vector<std::shared_ptr<data_batch>> output;
  {
    auto ro               = input->to_read_only();
    auto const input_view = sirius::get_cudf_table_view(ro);
    std::vector<cudf::column_view> key_views{input_view.column(0)};
    expected = reference_partition_rows<mixed_rows>(input_view,
                                                    cudf::table_view{key_views},
                                                    4,
                                                    cudf::get_default_stream(),
                                                    mem_space->get_default_allocator(),
                                                    copy_mixed_rows);
    output   = gpu_partition_impl::hash_partition(ro, {0}, 4, producer_stream.view(), *mem_space);
  }

  auto const survivor_it = std::find_if(output.begin(), output.end(), [](auto const& batch) {
    auto ro = batch->to_read_only();
    return sirius::get_cudf_table_view(ro).num_rows() != 0;
  });
  REQUIRE(survivor_it != output.end());
  auto const survivor_index = static_cast<std::size_t>(std::distance(output.begin(), survivor_it));
  auto survivor             = *survivor_it;
  output.clear();

  std::unique_ptr<idata_representation> cloned;
  {
    auto mut = survivor->to_mutable();
    cloned   = mut.get_data()->clone(consumer_stream.view());
  }

  survivor.reset();
  input.reset();
  REQUIRE(cloned != nullptr);
  auto const& cloned_repr = cloned->cast<cucascade::gpu_table_representation>();
  auto const cloned_view  = cloned_repr.get_table_view();
  for (auto const& column : cloned_view) {
    REQUIRE(column.offset() == 0);
  }
  REQUIRE(copy_mixed_rows(cloned_view) == expected[survivor_index]);
}

TEST_CASE("Non-first mixed partition survives registry GPU host GPU round-trip",
          "[operator][hash_partition][conversion][zero_copy]")
{
  auto manager     = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space  = manager->get_memory_space(Tier::GPU, 0);
  auto* host_space = manager->get_memory_space(Tier::HOST, 0);
  REQUIRE(gpu_space != nullptr);
  REQUIRE(host_space != nullptr);

  std::vector<int32_t> keys(128);
  std::iota(keys.begin(), keys.end(), int32_t{0});
  std::vector<int64_t> payload(keys.size());
  std::vector<std::string> strings;
  strings.reserve(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i) {
    payload[i] = 10'000 + static_cast<int64_t>(i);
    strings.push_back("round_trip_" + std::to_string(i));
  }

  auto input              = make_mixed_batch(*gpu_space, keys, strings, payload);
  auto const setup_stream = cudf::get_default_stream();
  std::vector<mixed_rows> expected;
  std::vector<cudf::data_type> schema;
  {
    auto ro         = input->to_read_only();
    auto const view = sirius::get_cudf_table_view(ro);
    schema          = copy_schema(view);
    std::vector<cudf::column_view> key_views{view.column(0)};
    expected = reference_partition_rows<mixed_rows>(view,
                                                    cudf::table_view{key_views},
                                                    4,
                                                    setup_stream,
                                                    gpu_space->get_default_allocator(),
                                                    copy_mixed_rows);
  }
  setup_stream.synchronize();

  rmm::cuda_stream producer_stream{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream consumer_stream{rmm::cuda_stream::flags::non_blocking};
  std::vector<std::shared_ptr<data_batch>> output;
  {
    auto ro = input->to_read_only();
    output  = gpu_partition_impl::hash_partition(ro, {0}, 4, producer_stream.view(), *gpu_space);
  }

  // Select a later partition because raw non-first cuDF slices can carry nonzero offsets;
  // converters require canonical offset-zero columns.
  auto survivor_it = std::find_if(std::next(output.begin()), output.end(), [](auto const& batch) {
    auto ro = batch->to_read_only();
    return sirius::get_cudf_table_view(ro).num_rows() != 0;
  });
  REQUIRE(survivor_it != output.end());
  auto const survivor_index = static_cast<std::size_t>(std::distance(output.begin(), survivor_it));
  REQUIRE(survivor_index > 0);
  auto survivor = *survivor_it;

  output.clear();

  auto round_trip = [&]<typename HostRepresentation>() {
    auto& registry = sirius::converter_registry::get();
    std::unique_ptr<HostRepresentation> host;
    {
      auto mut = survivor->to_mutable();
      host =
        registry.convert<HostRepresentation>(*mut.get_data(), host_space, consumer_stream.view());
    }
    survivor.reset();
    input.reset();
    REQUIRE(host != nullptr);

    auto back = registry.convert<cucascade::gpu_table_representation>(
      *host, gpu_space, consumer_stream.view());
    host.reset();
    consumer_stream.synchronize();
    REQUIRE(back != nullptr);
    require_schema_and_layout(back->get_table_view(), schema);
    REQUIRE(copy_mixed_rows(back->get_table_view()) == expected[survivor_index]);
  };

  SECTION("packed host representation")
  {
    round_trip.operator()<cucascade::host_data_packed_representation>();
  }
  SECTION("direct host representation")
  {
    round_trip.operator()<cucascade::host_data_representation>();
  }
}

TEST_CASE("Non-first mixed partition survives registry GPU peer conversion",
          "[operator][hash_partition][conversion][zero_copy][multi_gpu]")
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
    WARN("skipping: hash partition peer conversion requires at least two GPUs");
    (void)cudaGetLastError();
    return;
  }

  auto manager = sirius::test::operator_utils::initialize_memory_manager(2);
  auto* gpu0   = manager->get_memory_space(Tier::GPU, 0);
  auto* gpu1   = manager->get_memory_space(Tier::GPU, 1);
  REQUIRE(gpu0 != nullptr);
  REQUIRE(gpu1 != nullptr);

  rmm::cuda_set_device_raii source_device{rmm::cuda_device_id{0}};
  std::vector<int32_t> keys(128);
  std::iota(keys.begin(), keys.end(), int32_t{0});
  std::vector<int64_t> payload(keys.size());
  std::vector<std::string> strings;
  strings.reserve(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i) {
    payload[i] = 20'000 + static_cast<int64_t>(i);
    strings.push_back("peer_" + std::to_string(i));
  }
  auto input              = make_mixed_batch(*gpu0, keys, strings, payload);
  auto const setup_stream = cudf::get_default_stream();
  std::vector<mixed_rows> expected;
  std::vector<cudf::data_type> schema;
  {
    auto ro         = input->to_read_only();
    auto const view = sirius::get_cudf_table_view(ro);
    schema          = copy_schema(view);
    std::vector<cudf::column_view> key_views{view.column(0)};
    expected = reference_partition_rows<mixed_rows>(view,
                                                    cudf::table_view{key_views},
                                                    4,
                                                    setup_stream,
                                                    gpu0->get_default_allocator(),
                                                    copy_mixed_rows);
  }
  setup_stream.synchronize();

  rmm::cuda_stream producer_stream{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream consumer_stream{rmm::cuda_stream::flags::non_blocking};
  std::vector<std::shared_ptr<data_batch>> output;
  {
    auto ro = input->to_read_only();
    output  = gpu_partition_impl::hash_partition(ro, {0}, 4, producer_stream.view(), *gpu0);
  }

  // Select a later partition because raw non-first cuDF slices can carry nonzero offsets;
  // converters require canonical offset-zero columns.
  auto survivor_it = std::find_if(std::next(output.begin()), output.end(), [](auto const& batch) {
    auto ro = batch->to_read_only();
    return sirius::get_cudf_table_view(ro).num_rows() != 0;
  });
  REQUIRE(survivor_it != output.end());
  auto const survivor_index = static_cast<std::size_t>(std::distance(output.begin(), survivor_it));
  REQUIRE(survivor_index > 0);
  auto survivor = *survivor_it;
  output.clear();

  std::unique_ptr<cucascade::gpu_table_representation> peer_copy;
  {
    auto mut  = survivor->to_mutable();
    peer_copy = sirius::converter_registry::get().convert<cucascade::gpu_table_representation>(
      *mut.get_data(), gpu1, consumer_stream.view());
  }
  survivor.reset();
  input.reset();
  REQUIRE(peer_copy != nullptr);
  REQUIRE(peer_copy->get_device_id() == 1);

  {
    rmm::cuda_set_device_raii target_device{rmm::cuda_device_id{1}};
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    require_schema_and_layout(peer_copy->get_table_view(), schema);
    REQUIRE(copy_mixed_rows(peer_copy->get_table_view()) == expected[survivor_index]);
    peer_copy.reset();
  }
}

TEST_CASE("Fixed-only partition preserves representative fixed-width types",
          "[operator][hash_partition][zero_copy][fixed_fidelity]")
{
  auto* mem_space                 = get_shared_mem_space();
  auto const stream               = cudf::get_default_stream();
  auto const mr                   = mem_space->get_default_allocator();
  constexpr std::size_t row_count = 96;

  std::vector<int32_t> keys(row_count);
  std::vector<int8_t> bools(row_count);
  std::vector<int8_t> int8s(row_count);
  std::vector<int16_t> int16s(row_count);
  std::vector<int64_t> int64s(row_count);
  std::vector<__int128_t> decimal128s(row_count);
  std::vector<int64_t> timestamps(row_count);
  std::vector<int64_t> durations(row_count);
  std::vector<int64_t> decimal64s(row_count);
  for (std::size_t i = 0; i < row_count; ++i) {
    keys[i]        = static_cast<int32_t>(i);
    bools[i]       = static_cast<int8_t>(i % 2);
    int8s[i]       = static_cast<int8_t>(i - 48);
    int16s[i]      = static_cast<int16_t>(i * 17);
    int64s[i]      = 30'000 + static_cast<int64_t>(i);
    decimal128s[i] = (static_cast<__int128_t>(1) << 80) + static_cast<__int128_t>(i);
    timestamps[i]  = 1'700'000'000'000'000 + static_cast<int64_t>(i);
    durations[i]   = -10'000 + static_cast<int64_t>(i) * 3;
    decimal64s[i]  = 40'000 + static_cast<int64_t>(i);
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    make_fixed_test_column(cudf::data_type{cudf::type_id::INT32}, keys, stream, mr));
  columns.push_back(
    make_fixed_test_column(cudf::data_type{cudf::type_id::BOOL8}, bools, stream, mr));
  columns.push_back(
    make_fixed_test_column(cudf::data_type{cudf::type_id::INT8}, int8s, stream, mr));
  columns.push_back(
    make_fixed_test_column(cudf::data_type{cudf::type_id::INT16}, int16s, stream, mr));
  columns.push_back(
    make_fixed_test_column(cudf::data_type{cudf::type_id::INT64}, int64s, stream, mr));
  columns.push_back(make_fixed_test_column(
    cudf::data_type{cudf::type_id::DECIMAL128, -7}, decimal128s, stream, mr));
  columns.push_back(make_fixed_test_column(
    cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}, timestamps, stream, mr));
  columns.push_back(make_fixed_test_column(
    cudf::data_type{cudf::type_id::DURATION_NANOSECONDS}, durations, stream, mr));
  columns.push_back(
    make_fixed_test_column(cudf::data_type{cudf::type_id::DECIMAL64, -3}, decimal64s, stream, mr));

  auto input = make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                               *mem_space,
                               stream,
                               telemetry::batch_telemetry_info{});
  stream.synchronize();
  std::vector<cudf::data_type> schema;
  fixed_rows expected;
  {
    auto ro         = input->to_read_only();
    auto const view = sirius::get_cudf_table_view(ro);
    schema          = copy_schema(view);
    expected        = copy_fixed_rows(view);
  }
  REQUIRE(schema[5] == cudf::data_type(cudf::type_id::DECIMAL128, -7));
  REQUIRE(cudf::size_of(schema[5]) == 16);
  REQUIRE(schema[6] == cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS});
  REQUIRE(schema[7] == cudf::data_type{cudf::type_id::DURATION_NANOSECONDS});
  REQUIRE(schema[8] == cudf::data_type(cudf::type_id::DECIMAL64, -3));

  std::vector<std::shared_ptr<data_batch>> output;
  {
    auto ro = input->to_read_only();
    output  = gpu_partition_impl::hash_partition(ro, {0}, 4, stream, *mem_space);
  }
  stream.synchronize();
  std::vector<cudf::size_type> fixed_indices(schema.size());
  std::iota(fixed_indices.begin(), fixed_indices.end(), cudf::size_type{0});
  require_fixed_aliases_and_sizes(output, fixed_indices, true);

  auto const row_width = std::accumulate(
    schema.begin(), schema.end(), std::size_t{0}, [](std::size_t total, auto const type) {
      return total + cudf::size_of(type);
    });
  fixed_rows actual;
  for (auto const& batch : output) {
    auto ro         = batch->to_read_only();
    auto const view = sirius::get_cudf_table_view(ro);
    require_schema_and_layout(view, schema);
    REQUIRE(ro.get_data()->get_size_in_bytes() ==
            static_cast<std::size_t>(view.num_rows()) * row_width);
    auto const rows = copy_fixed_rows(view);
    actual.insert(rows.begin(), rows.end());
  }
  REQUIRE(actual == expected);
}

TEST_CASE("Nested-only LIST partitions are canonical owning fallbacks",
          "[operator][hash_partition][fallback][list]")
{
  auto* mem_space = get_shared_mem_space();
  std::vector<cudf::size_type> offsets{0};
  std::vector<int32_t> values;
  for (int32_t row = 0; row < 48; ++row) {
    auto const length = static_cast<cudf::size_type>(row % 5);
    for (cudf::size_type item = 0; item < length; ++item) {
      values.push_back(row * 10 + item);
    }
    offsets.push_back(static_cast<cudf::size_type>(values.size()));
  }
  auto input = make_list_batch(*mem_space, offsets, values);

  std::vector<list_rows> expected;
  std::vector<std::shared_ptr<data_batch>> output;
  std::vector<cudf::data_type> schema;
  list_rows all_input;
  {
    auto ro         = input->to_read_only();
    auto const view = sirius::get_cudf_table_view(ro);
    schema          = copy_schema(view);
    all_input       = copy_list_rows(view);
    std::vector<cudf::column_view> key_views{view.column(0)};
    expected = reference_partition_rows<list_rows>(view,
                                                   cudf::table_view{key_views},
                                                   4,
                                                   cudf::get_default_stream(),
                                                   mem_space->get_default_allocator(),
                                                   copy_list_rows);
    output = gpu_partition_impl::hash_partition(ro, {0}, 4, cudf::get_default_stream(), *mem_space);
  }
  cudf::get_default_stream().synchronize();

  list_rows all_output;
  for (std::size_t partition = 0; partition < output.size(); ++partition) {
    auto ro         = output[partition]->to_read_only();
    auto const view = sirius::get_cudf_table_view(ro);
    require_schema_and_layout(view, schema);
    auto const rows = copy_list_rows(view);
    REQUIRE(rows == expected[partition]);
    all_output.insert(rows.begin(), rows.end());
  }
  REQUIRE(all_output == all_input);

  auto survivor_it = std::find_if(output.begin(), output.end(), [](auto const& batch) {
    auto ro = batch->to_read_only();
    return sirius::get_cudf_table_view(ro).num_rows() != 0;
  });
  REQUIRE(survivor_it != output.end());
  auto const survivor_index = static_cast<std::size_t>(std::distance(output.begin(), survivor_it));
  auto survivor             = *survivor_it;
  output.clear();
  input.reset();

  auto ro         = survivor->to_read_only();
  auto const view = sirius::get_cudf_table_view(ro);
  require_schema_and_layout(view, schema);
  REQUIRE(copy_list_rows(view) == expected[survivor_index]);
}

TEST_CASE("Empty partitions preserve decimal and nested LIST schema",
          "[operator][hash_partition][empty][schema]")
{
  auto* mem_space         = get_shared_mem_space();
  auto const stream       = cudf::get_default_stream();
  auto const mr           = mem_space->get_default_allocator();
  auto const decimal_type = cudf::data_type{cudf::type_id::DECIMAL64, -4};

  auto decimal = make_fixed_test_column(decimal_type, std::vector<int64_t>{12'345}, stream, mr);
  auto offsets = make_fixed_test_column(
    cudf::data_type{cudf::type_id::INT32}, std::vector<cudf::size_type>{0, 2}, stream, mr);
  auto values = make_fixed_test_column(
    cudf::data_type{cudf::type_id::INT32}, std::vector<int32_t>{7, 8}, stream, mr);
  auto list =
    cudf::make_lists_column(1, std::move(offsets), std::move(values), 0, rmm::device_buffer{});

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(decimal));
  columns.push_back(std::move(list));
  auto input = make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                               *mem_space,
                               stream,
                               telemetry::batch_telemetry_info{});

  constexpr int num_partitions = 8;
  std::vector<std::shared_ptr<data_batch>> output;
  {
    auto ro = input->to_read_only();
    output  = gpu_partition_impl::hash_partition(ro, {0}, num_partitions, stream, *mem_space);
  }
  stream.synchronize();

  REQUIRE(output.size() == static_cast<std::size_t>(num_partitions));
  std::size_t empty_count = 0;
  for (auto const& batch : output) {
    auto ro         = batch->to_read_only();
    auto const view = sirius::get_cudf_table_view(ro);
    if (view.num_rows() != 0) { continue; }

    ++empty_count;
    REQUIRE(view.num_columns() == 2);
    REQUIRE(view.column(0).type() == decimal_type);
    REQUIRE(view.column(1).type() == cudf::data_type{cudf::type_id::LIST});
    require_canonical_column_tree(view.column(0));
    require_canonical_column_tree(view.column(1));
    REQUIRE(view.column(1).num_children() == 2);
    REQUIRE(view.column(1).child(0).type() == cudf::data_type{cudf::type_id::INT32});
    REQUIRE(view.column(1).child(1).type() == cudf::data_type{cudf::type_id::INT32});
  }
  REQUIRE(empty_count == static_cast<std::size_t>(num_partitions - 1));
}

TEST_CASE("Empty partitions do not retain the shared fixed-width allocation family",
          "[operator][hash_partition][zero_copy][memory]")
{
  auto manager    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);
  auto* counter =
    gpu_space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(counter != nullptr);
  auto const stream = cudf::get_default_stream();
  stream.synchronize();
  auto const allocated_before = counter->get_total_allocated_bytes();

  std::vector<int32_t> keys(64);
  std::iota(keys.begin(), keys.end(), int32_t{0});
  auto input =
    sirius::test::operator_utils::make_numeric_batch(*gpu_space, keys, cudf::type_id::INT32);
  std::vector<std::shared_ptr<data_batch>> output;
  {
    auto ro = input->to_read_only();
    output  = gpu_partition_impl::hash_partition(ro, {0}, 128, stream, *gpu_space);
  }
  stream.synchronize();
  REQUIRE(counter->get_total_allocated_bytes() > allocated_before);

  std::vector<std::shared_ptr<data_batch>> empty_batches;
  for (auto const& batch : output) {
    auto ro = batch->to_read_only();
    if (sirius::get_cudf_table_view(ro).num_rows() == 0) {
      REQUIRE(ro.get_data()->get_size_in_bytes() == 0);
      empty_batches.push_back(batch);
    }
  }
  REQUIRE(empty_batches.size() >= 64);

  output.clear();
  input.reset();
  stream.synchronize();
  REQUIRE(counter->get_total_allocated_bytes() == allocated_before);
  for (auto const& batch : empty_batches) {
    auto ro = batch->to_read_only();
    REQUIRE(sirius::get_cudf_table_view(ro).num_rows() == 0);
    REQUIRE(ro.get_data()->get_size_in_bytes() == 0);
  }
}

namespace {

void validate_evenly_partition(data_batch& input_batch,
                               const std::vector<std::shared_ptr<data_batch>>& output_batches,
                               int num_partitions)
{
  cudf::table_view input_table_view = sirius::get_cudf_table_view(input_batch);
  std::vector<cudf::table_view> output_table_views;
  for (const auto& output_batch : output_batches) {
    output_table_views.push_back(sirius::get_cudf_table_view(*output_batch));
  }

  // Check metadata
  REQUIRE(output_batches.size() == static_cast<size_t>(num_partitions));
  int actual_num_rows = 0;
  std::unordered_map<int, int> partition_num_rows_cnt;
  for (const auto& output_table : output_table_views) {
    ++partition_num_rows_cnt[output_table.num_rows()];
    actual_num_rows += output_table.num_rows();
    REQUIRE(output_table.num_columns() == input_table_view.num_columns());
    for (int c = 0; c < output_table.num_columns(); ++c) {
      REQUIRE(output_table.column(c).type().id() == input_table_view.column(c).type().id());
    }
  }
  REQUIRE(actual_num_rows == input_table_view.num_rows());
  REQUIRE(partition_num_rows_cnt[input_table_view.num_rows() / num_partitions + 1] ==
          input_table_view.num_rows() % num_partitions);
  REQUIRE(partition_num_rows_cnt[input_table_view.num_rows() / num_partitions] ==
          num_partitions - input_table_view.num_rows() % num_partitions);

  // Check data
  std::vector<std::vector<int64_t>> h_input_rows;
  copy_data_to_host_by_rows(input_table_view, h_input_rows);
  std::vector<std::vector<std::vector<int64_t>>> h_output_rows(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    copy_data_to_host_by_rows(output_table_views[i], h_output_rows[i]);
  }

  std::multiset<std::vector<int64_t>> output_set;
  for (const auto& partition : h_output_rows) {
    output_set.insert(partition.begin(), partition.end());
  }
  std::multiset<std::vector<int64_t>> input_set(h_input_rows.begin(), h_input_rows.end());
  REQUIRE(input_set == output_set);
}

}  // namespace

TEST_CASE("Evenly partition basic", "[operator][evenly_partition]")
{
  auto* mem_space                           = get_shared_mem_space();
  constexpr size_t num_input_rows           = 100;
  constexpr size_t num_partitions           = 4;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  auto input_batch =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  std::vector<std::shared_ptr<data_batch>> output_batches;
  {
    auto ro        = input_batch->to_read_only();
    output_batches = gpu_partition_impl::evenly_partition(
      ro, num_partitions, cudf::get_default_stream(), *mem_space);
  }
  validate_evenly_partition(*input_batch, output_batches, num_partitions);
}

TEST_CASE("Evenly partition basic with empty input", "[operator][evenly_partition]")
{
  auto* mem_space                           = get_shared_mem_space();
  constexpr size_t num_input_rows           = 0;
  constexpr size_t num_partitions           = 4;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  auto input_batch =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  std::vector<std::shared_ptr<data_batch>> output_batches;
  {
    auto ro        = input_batch->to_read_only();
    output_batches = gpu_partition_impl::evenly_partition(
      ro, num_partitions, cudf::get_default_stream(), *mem_space);
  }
  validate_evenly_partition(*input_batch, output_batches, num_partitions);
}

TEST_CASE("Evenly partition basic with num partitions larger than input size",
          "[operator][evenly_partition]")
{
  auto* mem_space                           = get_shared_mem_space();
  constexpr size_t num_input_rows           = 10;
  constexpr size_t num_partitions           = 20;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  auto input_batch =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  std::vector<std::shared_ptr<data_batch>> output_batches;
  {
    auto ro        = input_batch->to_read_only();
    output_batches = gpu_partition_impl::evenly_partition(
      ro, num_partitions, cudf::get_default_stream(), *mem_space);
  }
  validate_evenly_partition(*input_batch, output_batches, num_partitions);
}
