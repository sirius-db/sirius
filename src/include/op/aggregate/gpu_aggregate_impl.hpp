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
    const telemetry::batch_telemetry_info& telemetry_info = {});

  /**
   * @brief Hash-based drop-in replacement for `cudf::encode`.
   *
   * Returns exactly what `cudf::encode(keys)` returns: the distinct rows of `keys` in
   * ascending lexicographic order (NULLs last, per column), plus one INT32 index per input
   * row into that distinct table (`distinct[indices[i]] == keys[i]`). Row identity follows
   * `null_equality::EQUAL` / `nan_equality::ALL_EQUAL`, matching both `cudf::encode` and
   * cudf's sorted groupby under `null_policy::INCLUDE`.
   *
   * The difference is the cost model: `cudf::encode` assigns indices with a per-row
   * `lower_bound` binary search driven by a lexicographic row comparator
   * (O(N log G) comparator calls of random access over every key column — 119 ms for
   * 59M rows x 3 columns in TPC-H q16), whereas this implementation assigns them with one
   * hash-table probe per row (`cudf::distinct` + a sort at *distinct* cardinality +
   * `cudf::distinct_hash_join::left_join`), which is ~5-10x cheaper for string-heavy keys.
   *
   * @param keys Table of group key columns to encode. Nested (LIST/STRUCT) key columns are
   *        not supported (the label-encode caller gates them off; cudf throws on the
   *        unsupported ones).
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned table/column.
   *
   * @return {distinct key rows in sorted order, per-row INT32 index into them}.
   */
  static std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> hash_encode(
    const cudf::table_view& keys, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
};

}  // namespace op
}  // namespace sirius
