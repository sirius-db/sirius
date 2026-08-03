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

#include <cucascade/data/representation_converter.hpp>

#include "decode_pushdown.hpp"

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

/// Wave-2 output tier for one selected column.
enum class decode_output_tier : std::uint8_t {
  /// Bitpack-leaf plan: decode consumes the selection mask and writes
  /// compacted output directly (K3, survivor-count-first allocation).
  tier_a,
  /// Everything else: full-width decode as today, survivors gathered
  /// afterwards with mask-derived indices (scan side).
  tier_b,
};

/// The per-batch directive package for the wave orchestrator.
///
/// @c enabled is the iteration-1 gate: true iff every restricting conjunct of
/// the scan resolves during decode ON THIS CHUNK — the extraction converted the
/// whole filter (@c numeric_range_extraction::all_conjuncts_convertible) AND
/// every range column's plan here is a bitpack leaf. Only then may decode drop
/// rows (compacted output, post-decompress filter skipped). When false the
/// package is empty and the batch takes today's path unchanged.
struct fused_scan_directives {
  bool enabled = false;
  decode_range_pushdown ranges;                  ///< parallel to selected columns
  std::vector<decode_output_tier> output_tiers;  ///< parallel to selected columns
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
 * Returns a disabled (empty) package unless every active range column's plan is
 * a bitpack leaf of a fusable lane type — one non-fusable filter column sends
 * the whole batch down today's path (iteration 1: no mixed-mask combine).
 * Output tiers are tagged for ALL selected columns; a pure-filter column's
 * tier-A output is simply dropped by the scan's projection as usual.
 *
 * @throws std::runtime_error if @p attached_ranges is wider than
 *         @p selected_columns (a wiring bug, mirroring the equality pushdown).
 */
fused_scan_directives build_fused_scan_directives(const simpatico::compressed_table& table,
                                                  std::span<const std::size_t> selected_columns,
                                                  const decode_range_pushdown& attached_ranges,
                                                  bool all_conjuncts_convertible);

}  // namespace sirius
