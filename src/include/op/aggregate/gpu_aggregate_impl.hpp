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

#include <data/data_batch.hpp>
#include <memory/memory_space.hpp>

#include <memory>
#include <vector>

namespace sirius {
namespace op {

/**
 * @brief Functionalities for running local aggregation on a data batch.
 *
 * Provide functionalities including:
 * - Local ungrouped aggregation;
 * - Local grouped aggregation
 *
 * Require caller to have already upgraded input data batches into `gpu_table_representation`.
 */
class gpu_aggregate_impl {
 public:
  /**
   * @brief Perform local ungrouped aggregate on the input data batch.
   *
   * @param input The input data batch.
   * @param aggregates The aggregate functions.
   * @param aggregate_idx The aggregate columns, should have the same size as `aggregates`.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batch.
   */
  static std::shared_ptr<cucascade::data_batch> local_ungrouped_aggregate(
    std::shared_ptr<cucascade::data_batch> input,
    const std::vector<cudf::aggregation::Kind>& aggregates,
    const std::vector<int>& aggregate_idx,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space);

  /**
   * @brief Perform local grouped aggregate on the input data batch.
   *
   * @param input The input data batch.
   * @param group_idx The group columns.
   * @param aggregates The aggregate functions.
   * @param aggregate_idx The aggregate columns, should have the same size as `aggregates`.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batch.
   */
  static std::shared_ptr<cucascade::data_batch> local_grouped_aggregate(
    std::shared_ptr<cucascade::data_batch> input,
    const std::vector<int>& group_idx,
    const std::vector<cudf::aggregation::Kind>& aggregates,
    const std::vector<int>& aggregate_idx,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space);
};

}  // namespace op
}  // namespace sirius
