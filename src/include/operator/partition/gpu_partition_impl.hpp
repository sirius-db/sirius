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

#include "data/data_batch_view.hpp"
#include "data/data_repository_manager.hpp"
#include "memory/memory_space.hpp"

#include <cudf/cudf_utils.hpp>

namespace sirius {
namespace op {

/**
 * @brief Functionalities for partitioning the input data batch into multiple output batches.
 *
 * Provide functionalities including:
 * - Hash partitioning with specified partitioning columns;
 * - Evenly partitioning to evenly split the input table.
 *
 * Require caller to have already upgraded input data batches into `gpu_table_representation`
 * (the input data batch views are pinned).
 */
class gpu_partition_impl {
 public:
  /**
   * @brief Perform hash partitioning on the input data batch.
   *
   * @param input The input batches to be hash partitioned.
   * @param partition_key_idx Column ids of the partitioning columns.
   * @param num_partitions Number of partitions.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   * @param data_repository_mgr The data repository manager that the output data batch belongs to.
   *
   * @return The output data batch with ownership.
   */
  static sirius::vector<sirius::unique_ptr<data_batch>> hash_partition(
    const data_batch_view& input,
    const sirius::vector<int>& partition_key_idx,
    int num_partitions,
    rmm::cuda_stream_view stream,
    memory::memory_space& memory_space,
    data_repository_manager& data_repository_mgr);

  /**
   * @brief Perform evenly partitioning on the input data batch.
   *
   * @param input The input batches to be evenly partitioned.
   * @param num_partitions Number of partitions.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   * @param data_repository_mgr The data repository manager that the output data batch belongs to.
   *
   * @return The output data batch with ownership.
   */
  static sirius::vector<sirius::unique_ptr<data_batch>> evenly_partition(
    const data_batch_view& input,
    int num_partitions,
    rmm::cuda_stream_view stream,
    memory::memory_space& memory_space,
    data_repository_manager& data_repository_mgr);
};

}  // namespace op
}  // namespace sirius
