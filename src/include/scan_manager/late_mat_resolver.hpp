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

// Late-mat origin resolver (SIRIUS_EXP_LATE_MAT): turns a generation-checked
// column_origin into the layout/column views the late materializer consumes.
// This is the scan-manager side of the seam — the only code that knows how
// pinned entries store their chunks. The returned views are NON-OWNING and
// valid for the query lifetime (pin/unpin is query-lifecycle-serialized).

#include "late_mat/column_origin.hpp"
#include "late_mat/late_materializer.hpp"

#include <optional>

namespace sirius::scan_manager {

/// The origin table's layout (per-chunk rows + row starts, pin order), or
/// nullopt when the origin is stale (generation) or unsupported (HOST tier —
/// v1 materializes from device-resident pins only).
[[nodiscard]] std::optional<late_mat::pinned_table_layout> resolve_pinned_layout(
  late_mat::column_origin const& origin);

/// The per-chunk source views of one origin column (compressed table pointer
/// or uncompressed column view per chunk), or nullopt when the origin is
/// stale/unsupported or the column is nullable (v1 refusal, fused-path
/// parity).
[[nodiscard]] std::optional<late_mat::pinned_column_view> resolve_pinned_column(
  late_mat::column_origin const& origin);

}  // namespace sirius::scan_manager
