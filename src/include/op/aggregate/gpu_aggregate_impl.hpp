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

#include "sirius_config.hpp"
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
 * @brief Controls the runtime sorted-key proof for gpu_aggregate_impl::local_grouped_aggregate
 *
 * Eligible fixed-width keys are checked with `cudf::is_sorted` using the same ascending,
 * nulls-after order passed to the groupby. `sorted::YES` skips hashing and key sorting, but cuDF
 * may still allocate group/order metadata and gathered values.
 */
struct sorted_hint_options {
  /// Try the is_sorted check and take the sorted::YES path when it proves the keys sorted.
  bool enabled = false;
  /// Minimum input rows before the check runs (it costs one pass over the key columns).
  uint64_t min_rows = config::DEFAULT_SORTED_GROUPBY_HINT_MIN_ROWS;
};

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
    const cucascade::read_only_data_batch& input,
    const std::vector<cudf::aggregation::Kind>& aggregates,
    const std::vector<int>& aggregate_idx,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {});

  /**
   * @brief Perform local grouped aggregate on the input data batch.
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
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   * @param telemetry_info Telemetry lineage for the output batch.
   * @param sorted_hint Options for proving that group keys are sorted before aggregation.
   *
   * @return The output data batch.
   */
  static std::shared_ptr<cucascade::data_batch> local_grouped_aggregate(
    const cucascade::read_only_data_batch& input,
    const std::vector<int>& group_idx,
    const std::vector<cudf::aggregation::Kind>& aggregates,
    const std::vector<int>& aggregate_idx,
    const std::vector<std::vector<int>>& aggregate_struct_col_indices,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {},
    const sorted_hint_options& sorted_hint                = {});
};

}  // namespace op
}  // namespace sirius
