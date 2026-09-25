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

#include "telemetry/data_batch_probe.hpp"

#include <cudf/cudf_utils.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <vector>

namespace sirius {

namespace telemetry {
class telemetry_context;
}  // namespace telemetry

namespace op {

/**
 * @brief Functionalities for running local aggregation on a data batch.
 *
 * Provide functionalities including:
 * - Local ungrouped aggregation;
 * - Local grouped aggregation
 * - Local one row per key
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
    const cucascade::read_only_data_batch& input,
    const std::vector<cudf::aggregation::Kind>& aggregates,
    const std::vector<int>& aggregate_idx,
    ::cuda::stream_ref stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {});

  /**
   * @brief Perform local grouped aggregate on the input data batch.
   *
   * @throw std::runtime_error if `carried_idx` is non-empty and an entry of `group_idx` or
   * `carried_idx` is outside the input's columns
   *
   * @param input The input data batch.
   * @param group_idx The group columns.
   * @param aggregates The aggregate functions.
   * @param aggregate_idx The aggregate columns, should have the same size as `aggregates`.
   *        For multi-column COUNT DISTINCT (COLLECT_SET), the entry is -1 (sentinel) and
   *        the actual column indices are provided in `aggregate_struct_col_indices`.
   * @param aggregate_struct_col_indices Parallel to `aggregates`. Non-empty entries indicate
   *        a multi-column COLLECT_SET where a struct column is synthesized from those column
   *        indices. Empty entries (or an empty outer vector) use `aggregate_idx` directly.
   * @param carried_idx Input columns emitted after the aggregate results, all taken from one
   *        arbitrary row of each group; empty for none.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batch: the group keys, one column per aggregate, then the carried
   * columns.
   */
  static std::shared_ptr<cucascade::data_batch> local_grouped_aggregate(
    const cucascade::read_only_data_batch& input,
    const std::vector<int>& group_idx,
    const std::vector<cudf::aggregation::Kind>& aggregates,
    const std::vector<int>& aggregate_idx,
    const std::vector<std::vector<int>>& aggregate_struct_col_indices,
    const std::vector<int>& carried_idx,
    ::cuda::stream_ref stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {});

  /**
   * @brief Keep one arbitrary row per distinct key in one batch.
   *
   * Runs `cudf::distinct` over the whole input keyed on `group_idx`, with `KEEP_ANY`, NULL keys
   * equal to each other and every NaN equal to every other NaN, then emits the columns `select`
   * names in that order. `select` comes from `one_row_per_key_select`, so the output is the
   * group keys followed by the carried columns, the layout `PARTITION` and
   * `gpu_merge_impl::merge_one_row_per_key` expect.
   *
   * @throw std::runtime_error if an entry of `group_idx` or `select` is outside the input's columns
   *
   * @param input The input data batch.
   * @param group_idx The key columns, addressing the input.
   * @param select The input columns to emit, in output order. Each column may appear at most once,
   * because its data is moved into the output.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batch.
   */
  static std::shared_ptr<cucascade::data_batch> local_one_row_per_key(
    const cucascade::read_only_data_batch& input,
    const std::vector<int>& group_idx,
    const std::vector<int>& select,
    ::cuda::stream_ref stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {});
};

}  // namespace op
}  // namespace sirius
