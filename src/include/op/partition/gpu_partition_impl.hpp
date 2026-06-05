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

#pragma once

#include <cudf/cudf_utils.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <vector>

namespace sirius {
namespace op {

/**
 * @brief Functionalities for partitioning the input data batch into multiple output batches.
 *
 * Provide functionalities including:
 * - Hash partitioning with specified partitioning columns;
 * - Evenly partitioning to evenly split the input table.
 *
 * Require caller to have already upgraded input data batches into `gpu_table_representation`.
 */
class gpu_partition_impl {
 public:
  /**
   * @brief Perform hash partitioning on the input data batch.
   *
   * @param input The input batch to be hash partitioned.
   * @param partition_key_idx Column ids of the partitioning columns.
   * @param partition_key_cast_types Per-key target types for the partition hash.
   * @param num_partitions Number of partitions.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batches.
   */
  static std::vector<std::shared_ptr<cucascade::data_batch>> hash_partition(
    const cucascade::read_only_data_batch& input,
    const std::vector<int>& partition_key_idx,
    const std::vector<cudf::data_type>& partition_key_cast_types,
    int num_partitions,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space);

  /**
   * @brief Hash partition, returning the single reordered table plus zero-copy per-partition
   * views into it (instead of materializing each partition as its own table).
   *
   * cudf::hash_partition already produces ONE reordered table whose rows are grouped by
   * partition; this returns that table together with `num_partitions` `cudf::table_view`
   * slices (with any transient hash-cast columns dropped). The caller MUST keep the returned
   * table alive for as long as it uses the views, and MUST materialize (copy / concatenate)
   * before any view crosses a task boundary — views are not downgradable/spillable and a
   * sliced (offset) view cannot be safely peer-copied. This lets a caller that immediately
   * re-groups the partitions (e.g. cross-GPU shuffle coalescing) materialize exactly once at
   * the concat step rather than copying every fine partition and then concatenating again.
   *
   * @return {reordered_table, per-partition views into reordered_table}. View `i` covers the
   *         rows of partition `i`; views reference `reordered_table`'s device buffers.
   */
  static std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::table_view>>
  hash_partition_sliced(const cucascade::read_only_data_batch& input,
                        const std::vector<int>& partition_key_idx,
                        const std::vector<cudf::data_type>& partition_key_cast_types,
                        int num_partitions,
                        rmm::cuda_stream_view stream,
                        cucascade::memory::memory_space& memory_space);

  /// Overload without cast types (all keys hashed as-is). Kept for backward compatibility.
  static std::vector<std::shared_ptr<cucascade::data_batch>> hash_partition(
    const cucascade::read_only_data_batch& input,
    const std::vector<int>& partition_key_idx,
    int num_partitions,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space)
  {
    return hash_partition(input, partition_key_idx, {}, num_partitions, stream, memory_space);
  }

  /**
   * @brief Perform evenly partitioning on the input data batch.
   *
   * @param input The input batch to be evenly partitioned.
   * @param num_partitions Number of partitions.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batches.
   */
  static std::vector<std::shared_ptr<cucascade::data_batch>> evenly_partition(
    const cucascade::read_only_data_batch& input,
    int num_partitions,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space);
};

}  // namespace op
}  // namespace sirius
