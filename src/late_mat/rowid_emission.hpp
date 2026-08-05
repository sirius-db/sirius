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

// Scan-side pin-order rowid emission (SIRIUS_EXP_LATE_MAT; u32 width and its
// consumers additionally gated by SIRIUS_EXP_LATE_MAT_V2).
//
// CACHE COMPATIBILITY IS THE DESIGN INVARIANT: the rowid is synthesized
// POST-SERVE in the scan's execute(), from metadata the cached provider
// already stamps (the batch's origin range, and for fused-compacted batches
// the captured wave-1 mask). It is NEVER part of the scan's projection
// request — cache_entry_info::column_projection_for refuses rowid/virtual
// columns ("never cached"), so a projected rowid would push the scan off the
// pinned cache onto disk. This entry point cannot cause that by
// construction: it takes no part in column binding, pushdown coverage, or
// cache matching; it only manufactures a column for an already-served batch.
//
// Width: u64 always valid; u32 ONLY when the whole pinned table's rows fit
// 32 bits (the planner asserts the table total; this function re-checks the
// batch's own span and throws — never a silent overflow). The narrow width
// is what makes count-on-deferred pay: a 4-byte ride vs the 8-byte column it
// replaces.

#include "codegen/selection/selection.hpp"
#include "late_mat/column_origin.hpp"

#include <cudf/column/column.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <cstdlib>
#include <memory>

namespace sirius::late_mat {

// The v2 sub-gate reader lives in late_mat/column_origin.hpp (included above)
// beside late_mat_enabled() — single reader, never fork it. NOTE: the shared
// reader IMPLIES the main gate (v2 can never be on while SIRIUS_EXP_LATE_MAT
// is off), which is strictly tighter than a bare sub-gate check.

enum class rowid_width : std::uint8_t { u64 = 0, u32 = 1 };

struct rowid_emission_request {
  row_range range;                       ///< the served batch's global span
  rowid_width width = rowid_width::u64;  ///< u32 iff pinned-table rows < 2^32
  /// Fused-compacted batches: the captured wave-1 selection over the chunk's
  /// full row range (survivor_count == the batch's emitted rows). Null for a
  /// dense (whole-chunk) batch.
  sirius::codegen::selection_mask const* mask = nullptr;
};

/// Emit the pin-order rowid column for one served batch:
///   dense (mask == null):  out[k] = range.start + k          (k in [0, n_rows))
///   compacted (mask set):  out[k] = range.start + survivor_k (ascending)
/// Requirements (checked; throws std::runtime_error — a stamped scan whose
/// batch matches neither shape must fail loudly, never emit inconsistent
/// placeholder batches):
///   dense:     n_rows == range.rows
///   compacted: n_rows == mask->survivor_count and mask->chunk_offsets set
///   u32 width: range.start + range.rows <= 2^32
/// Stream-ordered on `stream` (one shipped mask->indices kernel + one
/// sequence/binary op); no host sync.
std::unique_ptr<cudf::column> emit_rowid_column(rowid_emission_request const& req,
                                                std::int64_t n_rows,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr);

}  // namespace sirius::late_mat
