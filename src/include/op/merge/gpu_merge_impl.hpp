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

#include "op/fold_limits.hpp"
#include "telemetry/data_batch_probe.hpp"

#include <cudf/cudf_utils.hpp>
#include <cudf/types.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace sirius {

namespace telemetry {
class telemetry_context;
}  // namespace telemetry

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
   * Every fold a CONCAT performs passes through here, so this is where INV-FOLD is enforced: the
   * result is one `cudf::table`, whose rows `cudf::size_type` must be able to address. The two
   * aggregate merges below fold as well and carry the same check, at the fixed cuDF limit.
   *
   * @param input The input batches to be concatenated.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   * @param telemetry_info Telemetry context linking the output batch into the query's lineage.
   * @param max_fold_rows Rows the result may hold; see `op/fold_limits.hpp`. Callers that are not
   *                      configurable keep the cuDF limit.
   *
   * @return The output data batch.
   * @throws fold_row_limit_exceeded (message marker `[fold_limit]`) when @p input holds more rows
   *         in total than @p max_fold_rows. A distinct type so a caller can tell an INV-FOLD
   *         violation from the other ways a fold fails -- device OOM, a non-GPU-resident batch.
   */
  static std::shared_ptr<cucascade::data_batch> concat(
    const std::vector<cucascade::read_only_data_batch>& input,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {},
    uint64_t max_fold_rows                                = k_fold_row_limit);

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
   * @throws fold_row_limit_exceeded when @p input holds more rows in total than one cuDF table can
   *         address (INV-FOLD; the merge concatenates before aggregating).
   */
  static std::shared_ptr<cucascade::data_batch> merge_ungrouped_aggregate(
    const std::vector<cucascade::read_only_data_batch>& input,
    const std::vector<cudf::aggregation::Kind>& aggregates,
    const std::vector<std::optional<cudf::size_type>>& merge_nth_index,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {});

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
   * @throws fold_row_limit_exceeded when @p input holds more rows in total than one cuDF table can
   *         address (INV-FOLD; the merge concatenates before aggregating). Unlike a CONCAT fold,
   *         no partition-count floor keeps this group small -- MERGE_GROUP_BY sizes by bytes
   *         alone -- so this check is the whole bound (residual R4).
   */
  static std::shared_ptr<cucascade::data_batch> merge_grouped_aggregate(
    const std::vector<cucascade::read_only_data_batch>& input,
    int num_group_cols,
    const std::vector<cudf::aggregation::Kind>& aggregates,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {});

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
    const std::vector<cucascade::read_only_data_batch>& input,
    const std::vector<int>& order_key_idx,
    const std::vector<cudf::order>& column_order,
    const std::vector<cudf::null_order>& null_precedence,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {});
};

}  // namespace op
}  // namespace sirius
