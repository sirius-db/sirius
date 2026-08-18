/*
 * Copyright 2026, Sirius Contributors.
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

#include "expression/ast/node.hpp"
#include "telemetry/data_batch_probe.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/data_batch.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace sirius {
namespace op {

/**
 * Clustered merge bypass for MERGE_GROUP_BY (config: enable_clustered_merge_bypass).
 *
 * When a grouped aggregate's input is clustered on its leading group key (e.g. a lineitem scan
 * grouped by l_orderkey, whose delta-bitpack encoding implies near-sorted keys), the per-batch
 * partial aggregates have (near-)disjoint key ranges and the PARTITION + MERGE_GROUP_BY chain
 * re-hashes every partial row to deduplicate almost nothing. This bypass detects that shape AT
 * RUNTIME from the actual batch data, skips the partition/merge re-hash, pushes the downstream
 * HAVING filter to each partial, and re-groups only the tiny key window shared by adjacent
 * batches.
 *
 * PROOF OBLIGATION (why this is exact, and why a wrong gate decision cannot corrupt results):
 *
 *  Let P_1..P_n be the partial-aggregate batches with observed leading-key ranges
 *  R_i = [min_i, max_i] (computed by cudf::reduce over the real batch data — never inferred from
 *  table names, encodings, or plan heuristics). A group key can occur in two batches only if it
 *  lies in R_i ∩ R_j. `analyze_partial_ranges` proves, from the sorted ranges, that only
 *  ADJACENT ranges intersect; the union of those pairwise intersections is the region list of
 *  the plan. Therefore:
 *
 *   1. A key OUTSIDE every region occurs in exactly one batch, so its partial row already equals
 *      the row the full merge would emit (partial batches are group-by outputs: one row per key
 *      per batch). Applying the downstream filter predicate to it is exactly what the plan would
 *      do after the merge.
 *   2. A key INSIDE a region may be split across (adjacent) batches. Every row whose key lies in
 *      a region is kept unconditionally — regardless of the filter — and fed to a re-group that
 *      uses the SAME merge combine (`merge_grouped_aggregate`) the normal path uses, so its
 *      combined row is identical to the full merge's row.
 *   3. All survivors flow through the re-group and then through the downstream FILTER operator,
 *      which re-applies the predicate: idempotent for case-1 rows (the merge combine is an
 *      identity on singletons for SUM/MIN/MAX/COUNT — the only kinds admitted), and exactly the
 *      post-merge filter for case-2 rows.
 *
 *  The disjointedness structure is REQUIRED for arming the bypass; the additional overlap-width
 *  gate is purely economic. Every reachable branch — including the defensive fall-backs inside
 *  `execute_bypass` — computes the exact merge semantics, so gate mistakes on non-clustered data
 *  can only cost time, never correctness. Memory stays bounded in every branch because the
 *  re-group hash-partitions its input first whenever it exceeds `hash_partition_bytes`.
 */
namespace clustered_bypass {

/// One key window shared by two adjacent partial batches (inclusive bounds, host-widened).
struct region {
  __int128 lo;
  __int128 hi;
};

/// The armed bypass plan: the proven overlap regions plus, per input batch id, the indices of
/// the regions that intersect that batch's key range (at most two under the adjacency proof).
struct plan {
  cudf::data_type key_type{cudf::type_id::EMPTY};
  std::vector<region> regions;
  std::unordered_map<uint64_t, std::vector<int>> batch_regions;
};

/// Whether the bypass supports `t` as a leading group key. Fixed-width integral and timestamp
/// types only: their min/max widen losslessly to __int128 for the host-side range proof, and
/// cudf AST literals exist for the device-side region membership test.
[[nodiscard]] bool supported_key_type(cudf::data_type t);

/**
 * @brief Runtime range proof over the partial batches waiting on the PARTITION input port.
 *
 * Computes each batch's leading-key min/max on the GPU, then checks on the host that the sorted
 * ranges intersect only adjacently and that each adjacent overlap is small (see
 * `max_overlap_fraction`; a small absolute floor admits tiny key spans, which is safe because a
 * partial batch cannot hold more rows than its key span holds distinct keys).
 *
 * @return The armed plan, or std::nullopt when the input is not proven clustered (mixed or
 *         unsupported key types, null keys, non-GPU-resident batches, fewer than two non-empty
 *         batches, or ranges that fail the disjointedness structure).
 */
[[nodiscard]] std::optional<plan> analyze_partial_ranges(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& batches,
  int key_column_index,
  double max_overlap_fraction);

/**
 * @brief Execute the armed bypass over the merge task's input batches.
 *
 * Per batch: evaluate the downstream filter predicate into a boolean mask, OR it (SQL
 * null-or semantics) with the region-membership mask, and gather the survivors. Survivors are
 * then re-grouped with the merge combine — hash-partitioned first when they exceed
 * `hash_partition_bytes` — unless the plan has no regions at all, in which case the filtered
 * partials are already final and are emitted directly.
 *
 * Defensive fall-back: a batch id the plan does not know (it was not on the partition port when
 * the proof ran) voids the per-batch filtering, and ALL input rows are re-grouped through the
 * partitioned merge combine instead — still exact, still memory-bounded.
 *
 * Always returns at least one batch (an empty batch of the input schema when nothing survives).
 */
[[nodiscard]] std::vector<std::shared_ptr<cucascade::data_batch>> execute_bypass(
  const std::vector<cucascade::read_only_data_batch>& batches,
  const plan& bypass_plan,
  const sirius::ast::node* filter_expression,
  const std::vector<int>& group_indices,
  const std::vector<cudf::aggregation::Kind>& merge_aggregates,
  uint64_t hash_partition_bytes,
  int num_gpus,
  rmm::cuda_stream_view stream,
  const telemetry::batch_telemetry_info& telemetry_info);

}  // namespace clustered_bypass
}  // namespace op
}  // namespace sirius
