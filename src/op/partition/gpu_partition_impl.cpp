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
#include "helper/numeric_narrowing.hpp"

#include <cudf/partitioning.hpp>

#include <cuda_runtime.h>

namespace sirius {
namespace op {

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

  // When a join condition has mixed key types (e.g. INT32 vs INT64), cuDF's murmur3 hash
  // produces different values for the same integer in different representations. We apply the
  // same cast that the join condition uses so both sides hash identically. Cast key columns are
  // appended to a transient table view used only for hashing; the output retains the original
  // schema.
  std::vector<std::unique_ptr<cudf::column>> owned_cast_cols;
  std::vector<cudf::column_view> all_col_views;
  all_col_views.reserve(input_table.num_columns() + partition_key_cast_types.size());
  for (int i = 0; i < input_table.num_columns(); i++) {
    all_col_views.push_back(input_table.column(i));
  }
  std::vector<int> effective_key_idx = partition_key_idx;
  for (size_t i = 0; i < partition_key_cast_types.size(); i++) {
    if (partition_key_cast_types[i].id() != cudf::type_id::EMPTY) {
      auto cast_col = sirius::cast_through_rep(
        input_table.column(partition_key_idx[i]), partition_key_cast_types[i], stream);
      effective_key_idx[i] = static_cast<int>(all_col_views.size());
      all_col_views.push_back(cast_col->view());
      owned_cast_cols.push_back(std::move(cast_col));
    }
  }
  cudf::table_view effective_table(all_col_views);
  const int orig_num_cols = input_table.num_columns();

  // cudf::hash_partition's CUB dispatch calls cudaPeekAtLastError(); a stale sticky
  // error from an earlier call would be misattributed to the scan inside hash_partition.
  (void)cudaGetLastError();

  auto partition_result = cudf::hash_partition(effective_table,
                                               effective_key_idx,
                                               num_partitions,
                                               cudf::hash_id::HASH_MURMUR3,
                                               cudf::DEFAULT_HASH_SEED,
                                               stream,
                                               memory_space.get_default_allocator());

  // Drop the appended cast columns before slicing. Releasing them here frees their reordered
  // copies; keeping them in the parent would hold that memory for as long as any output batch
  // lives, because the outputs are views into this table.
  auto reordered_columns = partition_result.first->release();
  reordered_columns.resize(orig_num_cols);
  auto reordered = std::make_shared<cudf::table>(std::move(reordered_columns));

  std::vector<cudf::size_type> slice_indices;
  slice_indices.reserve(num_partitions * 2);
  for (int i = 0; i < num_partitions; ++i) {
    slice_indices.push_back(partition_result.second[i]);
    slice_indices.push_back(i == num_partitions - 1 ? input_table.num_rows()
                                                    : partition_result.second[i + 1]);
  }
  auto sliced_partition_views = cudf::slice(reordered->view(), slice_indices, stream);

  // The reordered rows already sit one partition after another, so each partition is a view into
  // that table rather than an allocation of its own. A shared_ptr to the reordered table is the
  // owner: the last surviving partition batch releases it. Memory is therefore reclaimed per
  // reordered table, not per partition, so a single long-lived partition holds all of it.
  auto const reordered_bytes = reordered->alloc_size();
  auto const reordered_rows  = static_cast<std::size_t>(reordered->num_rows());
  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    auto const& partition_view = sliced_partition_views[i];
    auto const partition_rows  = static_cast<std::size_t>(partition_view.num_rows());
    if (partition_rows == 0) {
      // An empty partition owns an empty table instead: it allocates nothing, and a view would
      // pin the whole reordered table for a batch that carries no rows.
      output_batches.push_back(make_data_batch(
        std::make_unique<cudf::table>(partition_view, stream, memory_space.get_default_allocator()),
        memory_space,
        stream,
        telemetry_info));
      continue;
    }
    // Row-proportional share of the reordered table, so the partitions together are charged for
    // it exactly once. Exact for fixed-width columns, an estimate for variable-width ones.
    auto const partition_bytes = reordered_bytes * partition_rows / reordered_rows;
    output_batches.push_back(make_data_batch_from_view(partition_view,
                                                       std::shared_ptr<cudf::table>(reordered),
                                                       partition_bytes,
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
