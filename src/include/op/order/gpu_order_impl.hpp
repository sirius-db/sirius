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
 * @brief Functionalities for running local order-by or top-n on a data batch.
 *
 * Require caller to have already upgraded input data batches into `gpu_table_representation`.
 */
class gpu_order_impl {
 public:
  /**
   * @brief Perform local order-by on the input data batch.
   *
   * @param input The input data batch.
   * @param order_key_idx The columns to sort on.
   * @param column_order The desired sort order for each column.
   * @param null_precedence The desired order of null compared to other elements for each column.
   * Should have `order_idx.size() = column_order.size() = null_precedence.size()`.
   * @param projections The columns to construct output based on the sorted order.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batch.
   */
  static std::shared_ptr<cucascade::data_batch> local_order_by(
    std::shared_ptr<cucascade::data_batch> input,
    const std::vector<int>& order_key_idx,
    const std::vector<cudf::order>& column_order,
    const std::vector<cudf::null_order>& null_precedence,
    const std::vector<int>& projections,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space);
};

}  // namespace op
}  // namespace sirius
