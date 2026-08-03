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

#include <cstdint>
#include <vector>

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

}  // namespace sirius
