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

// Late-mat defer policy (SIRIUS_EXP_LATE_MAT): the plan-time walk that decides
// which pinned-scan payload columns ride as rowids and where they
// re-materialize. BAIL-EVERYWHERE BY CONSTRUCTION: any pipeline shape, join
// type, projection, filter, or origin the walk cannot prove transparent means
// NO deferral is installed and the query runs exactly as today. When it does
// install, it installs a PAIR atomically — the scan's substitution directive
// and the consuming operator's port-materialization directive.
//
// v1 transparent set (positions tracked through each):
//   DYNAMIC_FILTER / PARTITION / CONCAT  — identity column mapping;
//   single-op INNER HASH_JOIN pipelines  — payload pass (deferred columns must
//     not appear in the entry side's join-key references; positions remap
//     through lhs/rhs output projections; columns not projected out simply
//     drop from candidacy).
// Anything else stops the walk: the port through which the deferred columns
// enter that pipeline is the materialization port.
//
// Policy gates (env, read once): SIRIUS_EXP_LATE_MAT_DEFER (default on when
// the late-mat gate is on; "0" disables deferral while keeping annotations),
// SIRIUS_EXP_LATE_MAT_COMPRESSED (default OFF: compressed-origin scans defer
// only once the wave-seam capture is live — a stamped scan whose fused
// batch lacks a captured selection fails loudly at execute),
// SIRIUS_LATE_MAT_MIN_BOUNDARIES (default 4: minimum port crossings the ride
// must save — q10 customer = 6, q9 lineitem = 8; the q3/q4/q20-class <=3 ms
// rides stay out per R1's no-go table),
// SIRIUS_LATE_MAT_MIN_VALUE_BYTES (default 32: deferred-value floor,
// Σ(real deferred widths) − rowid ≥ T; the gate-on attribution measured the port-side
// id canonicalization at ~60 B/row break-even on the sort path, so the
// n_name/s_name-class 11–25 B dimension rides that cost +61 ms on q9's 800 M
// row port are rejected while the 154.6 B customer bundle and the 50 B
// supplier pair install). Consumer-slot arbitration is widest-bundle-wins
// with atomic eviction (first-install-wins let a
// 25-row nation ride lock out the flagship customer bundle); every rejected
// or evicted candidate logs one line for the census.

#include "scan_manager/sirius_scan_manager.hpp"

#include <span>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {
class sirius_gpu_scan_operator;
}

namespace sirius::scan_manager {

/// Analyze the plan downstream of @p scan_op and, when every check passes,
/// install the deferral pair. No-op (and cheap) in every other case.
/// Preconditions the CALLER guarantees (the prepare_for_query handoff):
/// gate on, entry has a live late-mat handle, no MVCC masks, no insert-delta
/// splits, single-GPU topology.
/// Per-query origin registry for the v3 FD closure: every gate-eligible
/// cached assignment's (entry, selected columns), built by prepare_for_query
/// BEFORE any install so rider origins resolve regardless of scan order.
struct late_mat_defer_context {
  struct origin_info {
    pinned_entry const* entry{nullptr};
    std::vector<std::size_t> columns;  ///< materialized order -> entry positions
  };
  std::unordered_map<op::scan::sirius_gpu_scan_operator*, origin_info> by_scan;
};

void try_install_late_mat_deferral(op::scan::sirius_gpu_scan_operator* scan_op,
                                   pinned_entry const& entry,
                                   std::span<std::size_t const> selected_columns,
                                   late_mat_defer_context const* context = nullptr);

}  // namespace sirius::scan_manager
