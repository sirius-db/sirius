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
#include <set>
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
    ::cuda::stream_ref stream,
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
    ::cuda::stream_ref stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {});

  /**
   * @brief Perform local grouped aggregate over several grouping sets on the input data batch.
   *
   * This is the local step of ROLLUP, CUBE and GROUPING SETS. It works in two steps:
   * 1. Aggregate the batch grouped by all keys in `group_idx`, like `local_grouped_aggregate()`.
   * 2. For each grouping set, aggregate that result again grouped by only the keys in the set.
   *
   * For example, with keys `(a, b)` and `ROLLUP(a, b)`, the output has these rows:
   * - set `{a, b}`, set id 0: one row per `(a, b)`, as in step 1
   * - set `{a}`, set id 1: one row per `a`, with `b` NULL. The COUNT for `a = 1` is the sum of
   *   the counts of the `(1, b)` rows from step 1
   * - set `{}`, set id 2: one row with `a` and `b` NULL
   *
   * The output stacks one block of rows per grouping set, in the order of `grouping_sets`.
   * Every block has the same columns, in this order:
   * - keys: one column per `group_idx` key, NULL in the rows of a set that leaves the key out
   * - set id: INT32, the position of the set in `grouping_sets`, which keeps the sets apart
   *   in the merge
   * - grouping functions: INT64, one column per `GROUPING()` function, constant within a block
   * - aggregates: the partial aggregates, with the same columns and types as
   *   `local_grouped_aggregate()` emits
   *
   * The empty grouping set always has one row, also for an empty batch. Its counts are then 0
   * and its other aggregates NULL, so a grand total over an empty input still returns a row.
   *
   * @param input The input data batch.
   * @param group_idx The group columns.
   * @param aggregates The aggregate functions.
   * @param aggregate_idx See `local_grouped_aggregate()`.
   * @param aggregate_struct_col_indices See `local_grouped_aggregate()`.
   * @param grouping_sets The grouping sets, as positions in `group_idx`.
   * @param grouping_functions The arguments of each `GROUPING()` function, as positions in
   *        `group_idx`.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batch.
   */
  static std::shared_ptr<cucascade::data_batch> local_grouping_sets_aggregate(
    const cucascade::read_only_data_batch& input,
    const std::vector<int>& group_idx,
    const std::vector<cudf::aggregation::Kind>& aggregates,
    const std::vector<int>& aggregate_idx,
    const std::vector<std::vector<int>>& aggregate_struct_col_indices,
    const std::vector<std::set<std::size_t>>& grouping_sets,
    const std::vector<std::vector<std::size_t>>& grouping_functions,
    ::cuda::stream_ref stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {});
};

}  // namespace op
}  // namespace sirius
