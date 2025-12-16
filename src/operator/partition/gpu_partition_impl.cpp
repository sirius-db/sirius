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

#include "data/gpu_data_representation.hpp"

#include <cudf/partitioning.hpp>

namespace sirius {
namespace op {

sirius::vector<sirius::unique_ptr<data_batch>> gpu_partition_impl::hash_partition(
  const data_batch_view& input,
  const sirius::vector<int>& partition_key_idx,
  int num_partitions,
  rmm::cuda_stream_view stream,
  memory::memory_space& memory_space,
  data_repository_manager& data_repository_mgr)
{
  // Sanity check.
  if (num_partitions < 2) {
    throw std::runtime_error("`num_partitions` in `hash_partition()` should be at least 2");
  }

  // Call cudf hash-partition
  auto input_table      = input.get_cudf_table_view();
  auto partition_result = cudf::hash_partition(input_table,
                                               partition_key_idx,
                                               num_partitions,
                                               cudf::hash_id::HASH_MURMUR3,
                                               cudf::DEFAULT_HASH_SEED,
                                               stream,
                                               memory_space.get_default_allocator());

  // Slice from the reordered table to create separate table partitions
  sirius::vector<sirius::unique_ptr<data_batch>> output_batches;
  sirius::vector<int> slice_indices;
  for (int i = 0; i < num_partitions; ++i) {
    slice_indices.push_back(partition_result.second[i]);
    slice_indices.push_back(i == num_partitions - 1 ? input_table.num_rows()
                                                    : partition_result.second[i + 1]);
  }
  auto sliced_partition_views = cudf::slice(partition_result.first->view(), slice_indices, stream);
  for (int i = 0; i < num_partitions; ++i) {
    auto output_partition = sirius::make_unique<cudf::table>(sliced_partition_views[i]);
    auto gpu_table_representation =
      sirius::make_unique<sirius::gpu_table_representation>(*output_partition, memory_space);
    output_batches.push_back(
      sirius::make_unique<sirius::data_batch>(data_repository_mgr.get_next_data_batch_id(),
                                              data_repository_mgr,
                                              std::move(gpu_table_representation)));
  }

  return output_batches;
}

sirius::vector<sirius::unique_ptr<data_batch>> gpu_partition_impl::evenly_partition(
  const data_batch_view& input,
  int num_partitions,
  rmm::cuda_stream_view stream,
  memory::memory_space& memory_space,
  data_repository_manager& data_repository_mgr)
{
  // Sanity check.
  if (num_partitions < 2) {
    throw std::runtime_error("`num_partitions` in `evenly_partition()` should be at least 2");
  }

  // Compute slice indices
  auto input_table            = input.get_cudf_table_view();
  int partition_num_rows_base = input_table.num_rows() / num_partitions;
  int remainder               = input_table.num_rows() % num_partitions;
  sirius::vector<int> slice_indices;
  for (int i = 0; i < num_partitions; ++i) {
    int curr_partition_num_rows = partition_num_rows_base + (i < remainder);
    slice_indices.push_back(i == 0 ? 0 : slice_indices.back());
    slice_indices.push_back(slice_indices.back() + curr_partition_num_rows);
  }

  // Slice and create separate partitions
  sirius::vector<sirius::unique_ptr<data_batch>> output_batches;
  auto sliced_partition_views = cudf::slice(input_table, slice_indices, stream);
  for (int i = 0; i < num_partitions; ++i) {
    auto output_partition = sirius::make_unique<cudf::table>(sliced_partition_views[i]);
    auto gpu_table_representation =
      sirius::make_unique<sirius::gpu_table_representation>(*output_partition, memory_space);
    output_batches.push_back(
      sirius::make_unique<sirius::data_batch>(data_repository_mgr.get_next_data_batch_id(),
                                              data_repository_mgr,
                                              std::move(gpu_table_representation)));
  }

  return output_batches;
}

}  // namespace op
}  // namespace sirius
