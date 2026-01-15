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

#include "operator/partition/gpu_partition_impl.hpp"

#include "data/data_batch_utils.hpp"

#include <cudf/partitioning.hpp>

namespace sirius {
namespace op {

std::vector<std::shared_ptr<cucascade::data_batch>> gpu_partition_impl::hash_partition(
  std::shared_ptr<cucascade::data_batch> input,
  const std::vector<int>& partition_key_idx,
  int num_partitions,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space)
{
  // Sanity check.
  if (num_partitions < 2) {
    throw std::runtime_error("`num_partitions` in `hash_partition()` should be at least 2");
  }

  // Call cudf hash-partition
  auto input_table      = get_cudf_table_view(*input);
  auto partition_result = cudf::hash_partition(input_table,
                                               partition_key_idx,
                                               num_partitions,
                                               cudf::hash_id::HASH_MURMUR3,
                                               cudf::DEFAULT_HASH_SEED,
                                               stream,
                                               memory_space.get_default_allocator());

  // Slice from the reordered table to create separate table partitions
  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  std::vector<cudf::size_type> slice_indices;
  for (int i = 0; i < num_partitions; ++i) {
    slice_indices.push_back(partition_result.second[i]);
    slice_indices.push_back(i == num_partitions - 1 ? input_table.num_rows()
                                                    : partition_result.second[i + 1]);
  }
  auto sliced_partition_views = cudf::slice(partition_result.first->view(), slice_indices, stream);
  for (int i = 0; i < num_partitions; ++i) {
    auto output_partition = std::make_unique<cudf::table>(sliced_partition_views[i]);
    output_batches.push_back(make_data_batch(std::move(output_partition), memory_space));
  }

  return output_batches;
}

std::vector<std::shared_ptr<cucascade::data_batch>> gpu_partition_impl::evenly_partition(
  std::shared_ptr<cucascade::data_batch> input,
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
    auto output_partition = std::make_unique<cudf::table>(sliced_partition_views[i]);
    output_batches.push_back(make_data_batch(std::move(output_partition), memory_space));
  }

  return output_batches;
}

}  // namespace op
}  // namespace sirius
