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

// Reading a pinned entry the way the late materializer needs it (env gate:
// SIRIUS_EXP_LATE_MAT).
//
// This is the only code on either side of the seam that knows how a pinned
// entry stores its chunks — device columns per name for a plain GPU pin,
// per-chunk device_pin_chunks that may interleave compressed and uncompressed
// storage, or host_chunks in the opposite orientation (one representation per
// chunk holding every pinned column). Everything above it works in layouts and
// column views, which is what lets the materializer be tested without a scan
// manager and the scan manager change storage without touching the
// materializer.
//
// Both entry points return nullopt rather than throwing, because every reason
// they can fail is a reason to simply not defer: a stale origin, a compressed
// origin, a host pin the device cannot address in place, chunks that disagree
// on their carrier width. A deferral that cannot be resolved must degrade to
// the ordinary path, not to an error.
//
// The views are NON-OWNING and valid for the query's lifetime, which pin/unpin
// serialization against query execution is what guarantees.

#include "late_mat/column_origin.hpp"
#include "late_mat/materialize.hpp"
#include "late_mat/prepared_selection.hpp"

#include <cstddef>
#include <optional>

namespace sirius::scan_manager {

struct pinned_entry;  // scan_manager/sirius_scan_manager.hpp

/// The origin table's batch layout, in pin order — how many rows each chunk
/// holds, which is all a selection needs to be split across them.
///
/// nullopt when the origin is stale (its generation no longer matches), when the
/// entry is on the DISK tier, or when a host-tier entry holds a chunk this path
/// cannot read (a compressed one).
[[nodiscard]] std::optional<late_mat::pinned_table_layout> resolve_pinned_layout(
  late_mat::column_origin const& origin);

/// The per-chunk sources of one origin column — a compressed table and column
/// index, a device column view, or a host chunk's block-scattered buffers, per
/// chunk.
///
/// Positionally consistent with resolve_pinned_layout's batches, which is the
/// invariant the materializer checks its inputs against: a mismatch there would
/// read the right number of rows out of the wrong chunks.
///
/// nullopt for a stale origin, a column position the entry does not have, a
/// COMPRESSED origin (the decode routes write values only, with no output
/// validity buffer), or a column whose chunks were narrowed to different carrier
/// widths — the view carries one dtype and the gather reads every batch at it.
/// An uncompressed origin MAY be nullable: every such gather shape propagates
/// validity.
///
/// A HOST-tier origin resolves to blocks read in place; it is refused unless
/// @ref host_pinned_column_is_addressable holds for it.
[[nodiscard]] std::optional<late_mat::pinned_column_view> resolve_pinned_column(
  late_mat::column_origin const& origin);

/// Whether a HOST-tier entry's column at @p column_position can be gathered
/// where it lies, without staging it to the device first.
///
/// Four things have to hold, and each is a refusal the install gate has to make
/// for the same reasons the resolver does — a deferral installed over a column
/// the resolver will later decline throws at the port, with the values already
/// gone:
///
///  * the device must be able to read registered pinned host memory directly
///    (unified addressing plus cudaDevAttrCanUseHostPointerForRegisteredMem);
///  * every chunk must be an uncompressed host representation, since a
///    Simpatico-compressed host blob has no addressable per-row layout;
///  * the column must be FIXED WIDTH in every chunk, at one carrier width, since
///    the gather translates a row to a byte offset by multiplying;
///  * the translation must land on element and mask-word boundaries, which the
///    block size and the recorded buffer offsets decide.
///
/// False for a GPU-tier entry: this asks about host addressability only.
[[nodiscard]] bool host_pinned_column_is_addressable(pinned_entry const& entry,
                                                     std::size_t column_position);

}  // namespace sirius::scan_manager
