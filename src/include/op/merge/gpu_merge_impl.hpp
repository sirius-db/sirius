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
#include <cudf/types.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace sirius {
namespace op {

/**
 * @brief Functionalities for merging multiple data batches into a single one.
 *
 * Provide functionalities including:
 * - Concatenate multiple data batches;
 * - Merge aggregation over multiple data batches (presumably each input data batch is a local
 * aggregation result);
 * - Merge sort over multiple sorted data batches.
 *
 * Require caller to have already upgraded input data batches into `gpu_table_representation`.
 */
class gpu_merge_impl {
 public:
  /**
   * @brief Concatenate multiple data batches.
   *
   * @param input The input batches to be concatenated.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batch.
   */
  static std::shared_ptr<cucascade::data_batch> concat(
    const std::vector<std::shared_ptr<cucascade::data_batch>>& input,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space);

  /**
   * @brief Perform ungrouped merge aggregate on multiple data batches.
   *
   * @param input The input batches to be merged.
   * @param aggregates The aggregate functions, should have the same size as num input columns.
   * @param merge_nth_index When aggregates[i] == NTH_ELEMENT, the nth index to use (e.g. 0 for
   * first).
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batch.
   */
  static std::shared_ptr<cucascade::data_batch> merge_ungrouped_aggregate(
    const std::vector<std::shared_ptr<cucascade::data_batch>>& input,
    const std::vector<cudf::aggregation::Kind>& aggregates,
    const std::vector<std::optional<cudf::size_type>>& merge_nth_index,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space);

  /**
   * @brief Perform grouped merge aggregate on multiple data batches.
   * For each batch, the first `num_group_cols` are the group columns, followed by aggregate columns
   * corresponding to `aggregates`.
   *
   * @param input The input batches to be merged.
   * @param num_group_cols The number of group columns.
   * @param aggregates The aggregate functions. Should satisfy `num_group_cols + group_idx.size() =
   * num input columns`.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batch.
   */
  static std::shared_ptr<cucascade::data_batch> merge_grouped_aggregate(
    const std::vector<std::shared_ptr<cucascade::data_batch>>& input,
    int num_group_cols,
    const std::vector<cudf::aggregation::Kind>& aggregates,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space);

  /**
   * @brief Perform merge order-by on multiple data batches.
   *
   * @param input The input batches to be merged.
   * @param order_key_idx The columns to sort on.
   * @param column_order The desired sort order for each column.
   * @param null_precedence The desired order of null compared to other elements for each column.
   * Should have `order_idx.size() = column_order.size() = null_precedence.size()`, and the three
   * parameters should be consistent to the sorted order of each input batch.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batch.
   */
  static std::shared_ptr<cucascade::data_batch> merge_order_by(
    const std::vector<std::shared_ptr<cucascade::data_batch>>& input,
    const std::vector<int>& order_key_idx,
    const std::vector<cudf::order>& column_order,
    const std::vector<cudf::null_order>& null_precedence,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space);
};

}  // namespace op
}  // namespace sirius
