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
// entry stores its chunks — device columns per name for a plain GPU pin, or
// per-chunk device_pin_chunks that may interleave compressed and uncompressed
// storage. Everything above it works in layouts and column views, which is
// what lets the materializer be tested without a scan manager and the scan
// manager change storage without touching the materializer.
//
// Both entry points return nullopt rather than throwing, because every reason
// they can fail is a reason to simply not defer: a stale origin, an entry
// living on the host, a nullable column. A deferral that cannot be resolved
// must degrade to the ordinary path, not to an error.
//
// The views are NON-OWNING and valid for the query's lifetime, which pin/unpin
// serialization against query execution is what guarantees.

#include "late_mat/column_origin.hpp"
#include "late_mat/materialize.hpp"
#include "late_mat/prepared_selection.hpp"

#include <optional>

namespace sirius::scan_manager {

/// The origin table's batch layout, in pin order — how many rows each chunk
/// holds, which is all a selection needs to be split across them.
///
/// nullopt when the origin is stale (its generation no longer matches) or the
/// entry is not device-resident: v1 materializes from GPU-tier pins only, since
/// a host chunk would have to be staged before any of this applies.
[[nodiscard]] std::optional<late_mat::pinned_table_layout> resolve_pinned_layout(
  late_mat::column_origin const& origin);

/// The per-chunk sources of one origin column — a compressed table and column
/// index, or a device column view, per chunk.
///
/// Positionally consistent with resolve_pinned_layout's batches, which is the
/// invariant the materializer checks its inputs against: a mismatch there would
/// read the right number of rows out of the wrong chunks.
///
/// nullopt for a stale or host-resident origin, a column position the entry
/// does not have, or a nullable column — nulls would need their validity
/// gathered alongside the values, which no materialization route does.
[[nodiscard]] std::optional<late_mat::pinned_column_view> resolve_pinned_column(
  late_mat::column_origin const& origin);

}  // namespace sirius::scan_manager
