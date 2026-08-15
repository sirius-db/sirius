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

#include "op/groupby_surrogate_deferral.hpp"
#include "op/groupby_surrogate_store.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace sirius::op {

/**
 * @brief GPU mechanisms for surrogate-key group-by finalization (see
 * `op/groupby_surrogate_deferral.hpp` for the module overview and correctness argument).
 *
 * `sirius_physical_grouped_aggregate_merge::finalize_surrogate_groupby` owns the policy (proof,
 * then restore, then re-group, before AVG / COUNT DISTINCT post-processing) and composes the
 * mechanisms here; the surrogate planner pass consults `is_recomposable_aggregate` so its gate
 * and the re-group can never drift apart.
 */
class gpu_surrogate_restore_impl {
 public:
  /// TRUE iff the aggregate kind survives the conservative full-tuple re-group
  /// (MIN / MAX / SUM / COUNT_ALL / COUNT_VALID). The planner gate and the re-group both consult
  /// THIS function -- single source of truth.
  [[nodiscard]] static bool is_recomposable_aggregate(cudf::aggregation::Kind kind) noexcept;

  /// The re-combining groupby aggregation for a recomposable kind (MIN -> min, MAX -> max,
  /// SUM / COUNT_* -> sum). Throws sirius::internal_exception for non-recomposable kinds.
  [[nodiscard]] static std::unique_ptr<cudf::groupby_aggregation> make_recompose_aggregation(
    cudf::aggregation::Kind kind);

  /// Exact fast-path proof: distinct_count over `real_keys` (nulls EQUAL, matching groupby
  /// null_policy::INCLUDE semantics) equals `num_rows`, which proves every full key tuple is
  /// distinct. Never consulted when any real key is FLOAT32/FLOAT64 -- SQL GROUP BY treats all
  /// NaNs as one group, but distinct_count's row comparator's NaN semantics are not
  /// contractually documented, so the proof comparator might be FINER than the grouping's on
  /// floating-point keys (two NaN rows counted distinct would fake a proof and leak a duplicate
  /// group) -- returns false so the caller re-groups.
  [[nodiscard]] static bool tuples_proven_distinct(std::vector<cudf::column_view> const& real_keys,
                                                   cudf::size_type num_rows,
                                                   rmm::cuda_stream_view stream);

  /// Materialize one restore group's string columns into `cols` at their key slots: snapshot
  /// the group's side of `store`, wait each source's writer event (STREAM-LINEAGE: the retained
  /// batches were written on the deferral join's streams), concatenate the sources in base
  /// order (which reproduces the absolute rowid address space exactly, see the store's
  /// invariants), and gather at the merged rowids. Handles the zero-row case with typed empty
  /// columns.
  static void restore_deferred_keys(std::vector<std::unique_ptr<cudf::column>>& cols,
                                    surrogate_restore_plan::restore_group const& group,
                                    surrogate_deferral_store const& store,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

  /// Conservative path: re-group by the full restored tuple (the first `num_key_cols` of
  /// `cols`), recombining the composable partial aggregates per `kinds`.
  [[nodiscard]] static std::vector<std::unique_ptr<cudf::column>> regroup_full_tuple(
    std::vector<std::unique_ptr<cudf::column>> cols,
    std::size_t num_key_cols,
    std::vector<cudf::aggregation::Kind> const& kinds,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);
};

}  // namespace sirius::op
