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

#include <cudf/column/column.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <utility>
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
   * @param enable_label_remap On the COLLECT_SET label path, produce the group labels with
   *        `cudf::distinct` + `cudf::key_remapping` instead of `cudf::encode`. The labels are
   *        byte-identical either way; see `compute_group_labels_via_remap`.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batch.
   */
  static std::shared_ptr<cucascade::data_batch> local_grouped_aggregate(
    const cucascade::read_only_data_batch& input,
    const std::vector<int>& group_idx,
    const std::vector<cudf::aggregation::Kind>& aggregates,
    const std::vector<int>& aggregate_idx,
    const std::vector<std::vector<int>>& aggregate_struct_col_indices,
    bool enable_label_remap,
    rmm::cuda_stream_view stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {});

  /**
   * @brief Compute dense INT32 group labels for a key table.
   *
   * Drop-in replacement for `cudf::encode(keys)`: returns the distinct key rows in
   * lexicographic order (nulls last) plus, for every input row, the INT32 index of its key
   * in that table. The labels are byte-identical to `cudf::encode`'s, but are produced with
   * `cudf::distinct` + a `cudf::key_remapping` hash probe (one hash + ~1 compare per row)
   * instead of encode's per-row lexicographic binary search, which is ~5x faster at
   * many-rows/few-groups shapes. Null keys group together (`null_equality::EQUAL`) and all
   * NaNs compare equal, matching `cudf::encode`'s semantics.
   *
   * Falls back to `cudf::encode` internally when the remap ids do not form a dense 0..n-1
   * permutation (`cudf::key_remapping` guarantees unique non-negative ids per distinct key
   * but does not promise density).
   *
   * @param keys The group key columns (at least one column required -- zero rows are fine;
   *        nested types unsupported, matching `cudf::encode`).
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used for the returned table and column.
   *
   * @return Pair of {distinct key rows sorted lexicographically (nulls last),
   *         per-row INT32 label column indexing into the table}.
   */
  static std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>>
  compute_group_labels_via_remap(const cudf::table_view& keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);
};

}  // namespace op
}  // namespace sirius
