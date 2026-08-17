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

#include "op/partition/gpu_partition_impl.hpp"

#include "data/data_batch_utils.hpp"

#include <cudf/column/column_stream.hpp>
#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius {
namespace op {

namespace {

enum class partition_column_storage { shared_fixed_width, copied_per_partition };

// Maps an original-schema column to its index in the compact table selected by `storage`; iterating
// the plan restores the original column order.
struct partition_column_plan {
  partition_column_storage storage;
  cudf::size_type compact_index;
};

// Copyable owner stored with one mixed partition view. Fixed columns are shared by nonempty sibling
// batches; copied columns, when present, belong only to this partition.
struct partition_view_owner {
  std::shared_ptr<cudf::table const> shared_fixed_columns;
  std::shared_ptr<cudf::table const> copied_columns;
};

// Pointer rebasing is safe only for flat, non-nullable fixed-width storage. Nullable masks and
// nested layouts require owning copies to normalize partition-relative buffers.
[[nodiscard]] bool can_alias_fixed_width(cudf::column_view const& column)
{
  return cudf::is_fixed_width(column.type()) && column.num_children() == 0 && !column.nullable();
}

[[nodiscard]] std::size_t checked_size(cudf::size_type value)
{
  if (value < 0) { throw std::overflow_error("negative cuDF size cannot be converted to size_t"); }
  return static_cast<std::size_t>(value);
}

[[nodiscard]] std::size_t checked_multiply(std::size_t lhs, std::size_t rhs)
{
  if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
    throw std::overflow_error("partition byte size exceeds size_t");
  }
  return lhs * rhs;
}

[[nodiscard]] std::size_t checked_add(std::size_t lhs, std::size_t rhs)
{
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    throw std::overflow_error("partition byte size exceeds size_t");
  }
  return lhs + rhs;
}

// Canonicalizes `[begin, end)` by advancing the data pointer. The returned view is non-owning and
// has offset zero for spill and peer-copy conversion; its combined allocation must outlive it.
[[nodiscard]] cudf::column_view make_offset_zero_fixed_width_alias(
  cudf::column_view const& combined, cudf::size_type begin, cudf::size_type end)
{
  if (begin < 0 || end < begin || end > combined.size()) {
    throw std::out_of_range("invalid hash partition range");
  }

  auto const rows         = end - begin;
  auto const byte_width   = cudf::size_of(combined.type());
  auto const byte_advance = checked_multiply(checked_size(begin), byte_width);
  auto const* data        = static_cast<std::byte const*>(combined.head()) + byte_advance;
  return cudf::column_view{combined.type(), rows, data, nullptr, 0, 0, {}};
}

}  // namespace

std::vector<std::shared_ptr<cucascade::data_batch>> gpu_partition_impl::hash_partition(
  const cucascade::read_only_data_batch& input,
  const std::vector<int>& partition_key_idx,
  const std::vector<cudf::data_type>& partition_key_cast_types,
  int num_partitions,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  // Sanity check.
  if (num_partitions < 2) {
    throw std::runtime_error("`num_partitions` in `hash_partition()` should be at least 2");
  }

  auto input_table = get_cudf_table_view(input);
  if (partition_key_idx.empty()) {
    throw std::invalid_argument("`partition_key_idx` in `hash_partition()` must not be empty");
  }
  if (!partition_key_cast_types.empty() &&
      partition_key_cast_types.size() != partition_key_idx.size()) {
    throw std::invalid_argument(
      "nonempty `partition_key_cast_types` must match `partition_key_idx` size");
  }
  for (auto const key_index : partition_key_idx) {
    if (key_index < 0 || key_index >= input_table.num_columns()) {
      throw std::out_of_range("partition key index is outside the input table");
    }
  }

  // When a join condition has mixed key types (e.g. INT32 vs INT64), cuDF's murmur3 hash produces
  // different values for the same integer in different representations. Build a separate key
  // table so temporary casts never become part of the partitioned payload.
  std::vector<std::unique_ptr<cudf::column>> owned_cast_cols;
  std::vector<cudf::column_view> effective_key_views;
  owned_cast_cols.reserve(partition_key_idx.size());
  effective_key_views.reserve(partition_key_idx.size());
  for (std::size_t i = 0; i < partition_key_idx.size(); ++i) {
    auto const& key = input_table.column(partition_key_idx[i]);
    if (i < partition_key_cast_types.size() &&
        partition_key_cast_types[i].id() != cudf::type_id::EMPTY) {
      auto cast_col =
        cudf::cast(key, partition_key_cast_types[i], stream, memory_space.get_default_allocator());
      effective_key_views.push_back(cast_col->view());
      owned_cast_cols.push_back(std::move(cast_col));
    } else {
      effective_key_views.push_back(key);
    }
  }
  cudf::table_view effective_keys{effective_key_views};

  // cudf::hash_partition's CUB dispatch calls cudaPeekAtLastError(); a stale sticky
  // error from an earlier call would be misattributed to the scan inside hash_partition.
  (void)cudaGetLastError();

  auto partition_result = cudf::hash_partition(input_table,
                                               effective_keys,
                                               num_partitions,
                                               cudf::hash_id::HASH_MURMUR3,
                                               cudf::DEFAULT_HASH_SEED,
                                               stream,
                                               memory_space.get_default_allocator());
  // The casts and their only consumer are ordered on stream, so stream-ordered deallocation can
  // release the transient key buffers before any per-partition payload copies are allocated.
  owned_cast_cols.clear();

  auto const& offsets = partition_result.second;
  if (offsets.size() != static_cast<std::size_t>(num_partitions) + 1) {
    throw std::runtime_error("cudf::hash_partition returned an invalid offset count");
  }

  auto combined_columns = partition_result.first->release();
  std::vector<partition_column_plan> column_plan;
  std::vector<std::unique_ptr<cudf::column>> fixed_columns;
  std::vector<std::unique_ptr<cudf::column>> unsupported_columns;
  column_plan.reserve(combined_columns.size());
  fixed_columns.reserve(combined_columns.size());
  unsupported_columns.reserve(combined_columns.size());

  // Split the combined result by storage family so fallback source buffers are not pinned by the
  // shared aliases. Rebinding all buffers to `stream` keeps fallback destruction ordered after the
  // per-partition copy reads enqueued below.
  for (auto& column : combined_columns) {
    column = cudf::rebind_stream(std::move(*column), stream);
    if (can_alias_fixed_width(column->view())) {
      column_plan.push_back({partition_column_storage::shared_fixed_width,
                             static_cast<cudf::size_type>(fixed_columns.size())});
      fixed_columns.push_back(std::move(column));
    } else {
      column_plan.push_back({partition_column_storage::copied_per_partition,
                             static_cast<cudf::size_type>(unsupported_columns.size())});
      unsupported_columns.push_back(std::move(column));
    }
  }

  std::shared_ptr<cudf::table const> shared_fixed_columns;
  if (!fixed_columns.empty()) {
    shared_fixed_columns = std::make_shared<cudf::table>(std::move(fixed_columns));
  }

  std::unique_ptr<cudf::table> unsupported_source;
  std::vector<cudf::table_view> unsupported_partition_views;
  if (!unsupported_columns.empty()) {
    // Deep-copy each nonempty slice below so cuDF canonicalizes its top-level and nested offsets
    // before this temporary combined source is released.
    unsupported_source = std::make_unique<cudf::table>(std::move(unsupported_columns));
    std::vector<cudf::size_type> slice_indices;
    slice_indices.reserve(static_cast<std::size_t>(num_partitions) * 2);
    for (int i = 0; i < num_partitions; ++i) {
      slice_indices.push_back(offsets[static_cast<std::size_t>(i)]);
      slice_indices.push_back(offsets[static_cast<std::size_t>(i) + 1]);
    }
    unsupported_partition_views = cudf::slice(unsupported_source->view(), slice_indices, stream);
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    auto const begin = offsets[static_cast<std::size_t>(i)];
    auto const end   = offsets[static_cast<std::size_t>(i) + 1];
    if (begin < 0 || end < begin || end > input_table.num_rows()) {
      throw std::runtime_error("cudf::hash_partition returned an invalid partition range");
    }

    if (begin == end) {
      // Preserve the input schema in an independent empty batch without pinning the shared
      // fixed-width allocation family.
      output_batches.push_back(
        make_data_batch(cudf::empty_like(input_table), memory_space, stream, telemetry_info));
      continue;
    }

    if (!shared_fixed_columns) {
      auto output_partition =
        std::make_unique<cudf::table>(unsupported_partition_views[static_cast<std::size_t>(i)],
                                      stream,
                                      memory_space.get_default_allocator());
      output_batches.push_back(
        make_data_batch(std::move(output_partition), memory_space, stream, telemetry_info));
      continue;
    }

    std::shared_ptr<cudf::table const> copied_columns;
    if (unsupported_source) {
      copied_columns =
        std::make_shared<cudf::table>(unsupported_partition_views[static_cast<std::size_t>(i)],
                                      stream,
                                      memory_space.get_default_allocator());
    }

    auto const fixed_view = shared_fixed_columns->view();
    std::vector<cudf::column_view> output_columns;
    output_columns.reserve(column_plan.size());
    // Attribute the logical fixed-width slice plus this partition's private copied storage. A
    // shared fixed-width allocation remains owned until the last nonempty sibling releases it.
    std::size_t logical_size = 0;
    for (auto const& plan : column_plan) {
      if (plan.storage == partition_column_storage::shared_fixed_width) {
        auto const& combined = fixed_view.column(plan.compact_index);
        output_columns.push_back(make_offset_zero_fixed_width_alias(combined, begin, end));
        logical_size =
          checked_add(logical_size,
                      checked_multiply(checked_size(end - begin), cudf::size_of(combined.type())));
      } else {
        output_columns.push_back(copied_columns->view().column(plan.compact_index));
      }
    }
    if (copied_columns) { logical_size = checked_add(logical_size, copied_columns->alloc_size()); }

    partition_view_owner owner{shared_fixed_columns, copied_columns};
    output_batches.push_back(make_data_batch_from_view(cudf::table_view{output_columns},
                                                       std::move(owner),
                                                       logical_size,
                                                       memory_space,
                                                       stream,
                                                       telemetry_info));
  }

  return output_batches;
}

std::vector<std::shared_ptr<cucascade::data_batch>> gpu_partition_impl::evenly_partition(
  const cucascade::read_only_data_batch& input,
  int num_partitions,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  // Sanity check.
  if (num_partitions < 2) {
    throw std::runtime_error("`num_partitions` in `evenly_partition()` should be at least 2");
  }

  // Compute slice indices
  auto input_table                        = get_cudf_table_view(input);
  cudf::size_type partition_num_rows_base = input_table.num_rows() / num_partitions;
  cudf::size_type remainder               = input_table.num_rows() % num_partitions;
  std::vector<cudf::size_type> slice_indices;
  for (int i = 0; i < num_partitions; ++i) {
    cudf::size_type curr_partition_num_rows = partition_num_rows_base + (i < remainder ? 1 : 0);
    slice_indices.push_back(i == 0 ? 0 : slice_indices.back());
    slice_indices.push_back(slice_indices.back() + curr_partition_num_rows);
  }

  // Slice and create separate partitions
  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  auto sliced_partition_views = cudf::slice(input_table, slice_indices, stream);
  for (int i = 0; i < num_partitions; ++i) {
    auto output_partition = std::make_unique<cudf::table>(
      sliced_partition_views[i], stream, memory_space.get_default_allocator());
    output_batches.push_back(
      make_data_batch(std::move(output_partition), memory_space, stream, telemetry_info));
  }

  return output_batches;
}

}  // namespace op
}  // namespace sirius
