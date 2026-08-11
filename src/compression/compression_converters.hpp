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

#include "decode_pushdown.hpp"

#include <cucascade/data/representation_converter.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace simpatico {
class compressed_table;
}  // namespace simpatico

namespace sirius {

/**
 * @brief Register Simpatico compression/decompression converters into @p registry.
 *
 * Registers:
 *   compressed_host_representation → gpu_table_representation
 *     (read the serialised .hpln file, select columns if projected, decompress)
 *
 * Called from converter_registry::initialize().
 */
void register_compression_converters(cucascade::representation_converter_registry& registry);

// ── Fused scan-filter decode directives (env gate SIRIUS_EXP_FUSED_SCAN_FILTER) ──
//
// The numeric analog of decode_equality_pushdown: instead of substituting one
// column with a BOOL8 mask, every range conjunct of a scan contributes a
// per-column selection mask during decode (wave 1), the masks AND together, and
// wave 2 decodes output columns against the combined mask. decode_range_entry /
// decode_range_pushdown live in decode_pushdown.hpp so the scan can attach them
// to a projected compressed representation without including this interface.
// The env gate is read by the orchestrator, not here.

/// Wave-2 output tier for one selected column: the plan-shape TAXONOMY, not a
/// capability claim. Whether a tier can actually decode compacted on this
/// build is answered by @c simpatico::plan_supports_selection_decode — the
/// probe lives next to the decode implementation, so new masked variants
/// (W1's delta mask-consume, K5 dict gather) light their tiers up the moment
/// they land, with no constant here to go stale.
enum class decode_output_tier : std::uint8_t {
  /// Plain bitpack leaf: K3 mask-consume writes compacted output directly
  /// (survivor-count-first allocation).
  tier_a,
  /// `input -> delta -> differences -> bitpack` (e.g. l_orderkey): compacted
  /// decode via the delta mask-consume variant once available.
  tier_a_delta,
  /// Dictionary-rooted string plan (`dictionary -> ... indices -> bitpack`):
  /// compacted decode via the K5 masked key gather once available.
  tier_dict_k5,
  /// `str_split` string plan (bitpack offsets + raw chars): compacted decode
  /// via the K6 masked offsets + survivor-only chars gather once available.
  tier_str_k6,
  /// Everything else: full-width decode as today, survivors gathered
  /// afterwards with mask-derived indices (scan side).
  tier_b,
};

/// Column-vs-column conjunct attached by the scan (iteration 5, q12-class
/// `colA OP colB`): positions index the representation's selected column list.
/// @c op numerically mirrors @c sirius::codegen::pair_compare_op
/// (lt=0, le=1, gt=2, ge=3, eq=4, ne=5) — spelled out so this header stays
/// simpatico-free, like the rest of the carrier types. The directive builder
/// validates both sides are K1m2-capable bitpack leaves and folds each side's
/// constant range into the pair (see codegen::pair_predicate); an invalid pair
/// is DROPPED (covers_whole_filter cleared) rather than emitted.
struct decode_pair_entry {
  std::size_t column_a = 0;
  std::size_t column_b = 0;
  std::uint8_t op      = 0;
};
using decode_pair_pushdown = std::vector<decode_pair_entry>;

/// The per-batch directive package for the wave orchestrator.
///
/// @c enabled: at least one range conjunct survives as a wave-1 mask source on
/// this chunk. Iteration 3 relaxes iteration 1's all-or-nothing rule: a
/// PARTIAL mask (some conjuncts unconvertible or dropped) is still sound —
/// mask conjuncts are conjunctive, so rows it drops are rows the full filter
/// would drop — as long as the batch is NOT tagged row-filtered and the scan
/// re-runs the full filter on the compacted output.
///
/// @c covers_whole_filter: the mask carries EVERY restricting conjunct of the
/// scan's pushed-down filter (extraction converted everything AND nothing was
/// dropped at directive build). Only then may the converter hand out a
/// batch with @c decode_outcome::row_filtered set; a partial mask must leave the
/// batch untagged so post_filter_and_project evaluates the residual.
struct fused_scan_directives {
  bool enabled             = false;
  bool covers_whole_filter = false;
  decode_range_pushdown ranges;                  ///< parallel to selected columns
  std::vector<decode_output_tier> output_tiers;  ///< parallel to selected columns
  /// Per selected column: its plan decodes compacted on THIS build
  /// (simpatico::plan_supports_selection_decode). Parallel to output_tiers;
  /// tier_b entries are always false.
  std::vector<std::uint8_t> compact_capable;
  /// Validated col-vs-col mask sources (K1m2), post drop. A column whose
  /// constant range is folded into a kept pair side has its standalone K1
  /// participation cleared (the pair is the source; one kernel does both).
  decode_pair_pushdown pairs;
};

/**
 * @brief Build the wave orchestrator's directive package for one compressed batch.
 *
 * @param table                     The batch's cached compressed table.
 * @param selected_columns          Physical column indices being decompressed, in
 *                                  output order (the representation's selection).
 * @param attached_ranges           The scan's range pushdown, parallel to
 *                                  @p selected_columns (shorter = inactive tail).
 * @param all_conjuncts_convertible The extraction gate
 *                                  (@c numeric_range_extraction::all_conjuncts_convertible).
 *
 * An active range on a column that is not a fusable bitpack leaf is DROPPED
 * from the mask (sound: the mask under-filters and the residual filter still
 * runs) and clears @c covers_whole_filter; the package is disabled only when
 * no mask source survives. Output tiers are tagged for ALL selected columns
 * (plan-shape taxonomy) with @c compact_capable answering per column whether
 * this build can decode it compacted; a pure-filter column's compacted output
 * is simply dropped by the scan's projection as usual.
 *
 * @p has_bool8_mask_sources declares that the caller will also contribute
 * dict-code equality conjuncts (scan_filter_request::bool8_filters) to the
 * scan mask; @p has_membership_mask_sources likewise for dynamic-membership
 * probes (scan_filter_request::membership_filters, Phase A). Either one lets
 * the package enable with zero range sources, and either forces
 * @c covers_whole_filter false — the extraction gate only speaks for the
 * static numeric view (dynamic filters are never whole-filter), so such
 * batches always keep the residual post-filter.
 *
 * @throws std::runtime_error if @p attached_ranges is wider than
 *         @p selected_columns (a wiring bug, mirroring the equality pushdown).
 */
fused_scan_directives build_fused_scan_directives(const simpatico::compressed_table& table,
                                                  std::span<const std::size_t> selected_columns,
                                                  const decode_range_pushdown& attached_ranges,
                                                  bool all_conjuncts_convertible,
                                                  bool has_bool8_mask_sources = false,
                                                  const decode_pair_pushdown& attached_pairs = {},
                                                  bool has_membership_mask_sources = false);

}  // namespace sirius
