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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace cudf {
class column;
class column_view;
}  // namespace cudf

namespace sirius {

// ── Fused scan-filter decode directives (env gate SIRIUS_EXP_FUSED_SCAN_FILTER) ──
//
// Shared between the scan (which attaches ranges to a projected compressed
// representation, sirius_scan_manager) and the converter (which builds the wave
// orchestrator's directive package from them, compression_converters). Lives in
// its own header so compressed_representation.hpp can carry the pushdown without
// pulling in the converter interface, exactly as decode_equality_pushdown stays
// a plain string vector there.

/// Numeric-range decode directive for one selected column.
///
/// @c lo / @c hi mirror @c sirius::codegen::range_predicate field-for-field;
/// they are spelled out here (rather than embedding the type) so this header —
/// which reaches test TUs through data/sirius_converter_registry.hpp — stays
/// free of the simpatico include tree, exactly like decode_equality_pushdown
/// keeps plain strings instead of simpatico::decode_predicate.
struct decode_range_entry {
  /// False ⇒ no range on this column; it decodes as its role dictates.
  bool active = false;
  /// Inclusive [lo,hi] in the DECODED integer domain (dates = stored day count,
  /// decimals = unscaled int at the column's scale). lo > hi is a provably
  /// empty range and selects nothing. Kernel parameters, never JIT constants.
  std::int64_t lo = 0;
  std::int64_t hi = 0;
  /// True when this column's decode-time mask ANDs into the scan-wide
  /// selection mask (wave 1 / K1). Set by build_fused_scan_directives, not by
  /// the scan: it implies the column's plan can evaluate the range in-decode.
  bool participates_in_scan_mask = false;
};

/// Parallel to the representation's selected column list, like
/// decode_equality_pushdown: entry @c i describes the @c i-th column the
/// converter will decompress. May be shorter (missing tail = inactive).
using decode_range_pushdown = std::vector<decode_range_entry>;

/// Type-erased membership probe for dynamic join filters (Phase A): evaluates
/// the published device structure (small_in_list / cuco set / Bloom) over the
/// decoded key column and returns a BOOL8 keep-mask. Wraps
/// sirius_dynamic_filter::compute_mask with the device id bound; the closure
/// co-owns the filter (shared_ptr capture) for the call's duration and must
/// enqueue only on the handed stream.
using membership_probe_fn = std::function<std::unique_ptr<cudf::column>(
  cudf::column_view const&, rmm::cuda_stream_view, rmm::device_async_resource_ref)>;

/// One probe plus its cap-ordering signal. The converter sorts the request's
/// membership sources by ascending EXPECTED keep-rate before the engine's
/// membership cap truncates the list (q21 lesson: a prefix cap must see the
/// strong filters first). Best static signal, in priority order:
///   kind_rank — 0 = small_in_list, 1 = cuco in_list set, 2 = Bloom (the set
///   forms are exact; Bloom over-keeps by construction); 255 = unknown, last.
///   num_keys  — build keys when the filter exposes it (fewer ⇒ stronger);
///   0 = unknown. Ties keep channel (publication) order.
struct decode_membership_probe {
  membership_probe_fn probe;
  std::uint8_t kind_rank = 255;
  std::uint64_t num_keys = 0;
};

/// Membership pushdown for one selected column: every probe ANDs into the
/// scan mask. Parallel to the selected column list like the ranges; an entry
/// with no probes decodes normally.
struct decode_membership_entry {
  std::vector<decode_membership_probe> probes;
};
using decode_membership_pushdown = std::vector<decode_membership_entry>;

}  // namespace sirius
