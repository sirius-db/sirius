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

#include "op/sirius_physical_operator.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::planner {

/**
 * @file
 * @brief Twin-scan fusion: fuse two near-duplicate probe-side scans of the same table into one
 * fan-out twin-scan pipeline.
 *
 * `sirius_physical_plan_generator::create_plan` invokes `fuse_twin_scans` (under the
 * `fuse_twin_scans` setting, read at the call site) while the scans are still `TABLE_SCAN` nodes,
 * before `insert_gpu_pipeline_operators` rewrites the fused scan into `GPU_SCAN ->
 * DYNAMIC_FILTER`. The returned `twin_scan_fusion_report` is stored on the generator and exposed
 * through `twin_scan_report()` as diagnostic state; the match conditions, the semantic-safety
 * proof, and the rewrite live in `sirius_plan_twin_scan_fusion.cpp`.
 */

/// Why a same-table candidate pair was not fused. Values are append-only; the evaluation order
/// (geometry checks before channel checks before proof checks) is part of the contract -- tests
/// pin exact reasons. See `sirius_plan_twin_scan_fusion.cpp` for the semantics of each check.
enum class twin_scan_rejection_reason : uint8_t {
  // match stage -- scan geometry (I1, I3-static)
  columns_not_strict_prefix,  // column_ids: size or content
  output_layout_not_prefix,   // projection_ids / identity layout
  output_types_not_prefix,    // includes width-consistency internal checks
  physical_carriers_differ,   // sidecar presence or shared-prefix carriers
  static_filters_differ,
  // prove stage -- channels (I4)
  channel_missing,
  channel_shared,  // both scans hold the same channel object
  channel_without_producer,
  channel_unscoped_producer,
  channel_multi_target,
  channel_target_invalid,  // out of range or rowid
  channel_targets_differ,  // different underlying table columns
  // prove stage -- delim-chain subsumption (I2)
  producer_join_not_unique,  // zero, or more than one, publishing hash join
  producer_joins_identical,
  build_not_delim_replay,
  delim_joins_identical,
  delim_chain_not_direct,    // B's delim join does not directly consume A's
  join_back_not_row_subset,  // A's join-back is not RIGHT_SEMI / RIGHT_ANTI
  delim_distinct_missing,
  delim_key_refs_differ,  // group_idx / types mismatch
  producer_key_not_single_equality,
  producer_keys_differ,
  producer_key_outside_delim_output,
};

/// The snake_case name of @p reason (matching its enumerator), as it appears in rejection logs.
[[nodiscard]] std::string_view to_string(twin_scan_rejection_reason reason) noexcept;

/// One rejected same-table pair: the reason plus both sites' one-line geometry summaries (column
/// ids, projection ids, output width, physical-sidecar presence, channel state).
struct twin_scan_rejection {
  twin_scan_rejection_reason reason;
  std::string site_a;
  std::string site_b;
};

/// What the pass did to one plan. `same_table_rejections` records only same-table candidate pairs
/// (a handful per query at most) -- cross-table pairs are noise and are not recorded.
struct twin_scan_fusion_report {
  std::size_t fused_pairs = 0;
  std::vector<twin_scan_rejection> same_table_rejections;
};

/// Fuse near-duplicate probe-side scans of the same table into one fan-out twin-scan pipeline.
/// Must run while scans are still TABLE_SCAN nodes (before `insert_gpu_pipeline_operators`).
/// Purely structural: reads no settings.
[[nodiscard]] twin_scan_fusion_report fuse_twin_scans(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan);

}  // namespace sirius::planner
