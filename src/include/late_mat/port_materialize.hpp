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

// The far end of a deferral: turn the rowid back into columns (env gate:
// SIRIUS_EXP_LATE_MAT).
//
// The scan replaced a bundle of columns with one UINT64 pin-order rowid and
// INT8 placeholders, and everything in between carried them as ordinary data.
// This puts the values back, in the batch's own row order, at the positions
// they were taken from — so the operator about to read them sees the table it
// would have seen had nothing been deferred.
//
// MATCHED BY THE WHOLE SCHEMA, NOT BY POSITION. An operator can receive batches
// from more than one producer, and materializing against the wrong one reads
// arbitrary rows of the pinned table — plausible values in the right shape,
// which is the worst kind of wrong. A batch that does not match is declined and
// passes through untouched.
//
// A DECLINE IS NOT A FAILURE, BUT A FAILED RESOLUTION IS. Once a batch matches,
// its values exist nowhere else: the scan threw them away against this
// directive's promise. So a stale origin, a host-resident entry or a nullable
// origin column throws rather than degrading — there is no ordinary path left
// to degrade to. The install side is what keeps those cases from arising.

#include "late_mat/defer_directive.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace sirius::late_mat {

/// Whether @p batch is the one @p directive was installed for.
[[nodiscard]] bool port_directive_matches(port_materialize_directive const& directive,
                                          cudf::table_view const& batch);

/// Restore @p directive's columns into @p batch.
///
/// The result has the batch's rows, in the batch's order, with each deferred
/// position holding its values again and every other position copied through.
/// That copy is over the batch as it rides — narrow, since the wide columns are
/// exactly the ones not there yet — which is why splicing by copy is affordable
/// here and would not have been at the scan.
///
/// @pre port_directive_matches(directive, batch). Callers check first; this
/// throws rather than guessing, since a mismatched batch has values of its own
/// that must not be overwritten.
[[nodiscard]] std::unique_ptr<cudf::table> materialize_at_port(
  port_materialize_directive const& directive,
  cudf::table_view const& batch,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::late_mat
