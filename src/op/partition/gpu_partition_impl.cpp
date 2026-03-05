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

#include <cudf/partitioning.hpp>
#include <cudf/unary.hpp>

namespace sirius {
namespace op {

std::vector<std::shared_ptr<cucascade::data_batch>> gpu_partition_impl::hash_partition(
  const std::shared_ptr<cucascade::data_batch>& input,
  const std::vector<int>& partition_key_idx,
  const std::vector<cudf::data_type>& partition_key_cast_types,
  int num_partitions,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space)
{
  // Sanity check.
  if (num_partitions < 2) {
    throw std::runtime_error("`num_partitions` in `hash_partition()` should be at least 2");
  }

  auto input_table = get_cudf_table_view(*input);

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
      auto cast_col =
        cudf::cast(input_table.column(partition_key_idx[i]), partition_key_cast_types[i], stream);
      effective_key_idx[i] = static_cast<int>(all_col_views.size());
      all_col_views.push_back(cast_col->view());
      owned_cast_cols.push_back(std::move(cast_col));
    }
  }
  cudf::table_view effective_table(all_col_views);
  const int orig_num_cols = input_table.num_columns();

  auto partition_result = cudf::hash_partition(effective_table,
                                               effective_key_idx,
                                               num_partitions,
                                               cudf::hash_id::HASH_MURMUR3,
                                               cudf::DEFAULT_HASH_SEED,
                                               stream,
                                               memory_space.get_default_allocator());

  // Build a column-index list for the original (non-cast) columns.
  std::vector<cudf::size_type> orig_col_indices;
  orig_col_indices.reserve(orig_num_cols);
  for (int i = 0; i < orig_num_cols; i++) {
    orig_col_indices.push_back(i);
  }

  // Slice from the reordered table to create separate table partitions.
  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  std::vector<cudf::size_type> slice_indices;
  slice_indices.reserve(num_partitions * 2);
  for (int i = 0; i < num_partitions; ++i) {
    slice_indices.push_back(partition_result.second[i]);
    slice_indices.push_back(i == num_partitions - 1 ? input_table.num_rows()
                                                    : partition_result.second[i + 1]);
  }
  auto sliced_partition_views = cudf::slice(partition_result.first->view(), slice_indices, stream);
  for (int i = 0; i < num_partitions; ++i) {
    // Drop any appended cast columns from the output.
    auto output_partition =
      std::make_unique<cudf::table>(sliced_partition_views[i].select(orig_col_indices),
                                    stream,
                                    memory_space.get_default_allocator());
    output_batches.push_back(make_data_batch(std::move(output_partition), memory_space));
  }

  return output_batches;
}

std::vector<std::shared_ptr<cucascade::data_batch>> gpu_partition_impl::evenly_partition(
  const std::shared_ptr<cucascade::data_batch>& input,
  int num_partitions,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space)
{
  // Sanity check.
  if (num_partitions < 2) {
    throw std::runtime_error("`num_partitions` in `evenly_partition()` should be at least 2");
  }

  // Compute slice indices
  auto input_table                        = get_cudf_table_view(*input);
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
    output_batches.push_back(make_data_batch(std::move(output_partition), memory_space));
  }

  return output_batches;
}

}  // namespace op
}  // namespace sirius
