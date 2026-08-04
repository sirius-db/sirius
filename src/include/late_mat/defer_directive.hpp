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

// Late-materialization deferral directives (env gate: SIRIUS_EXP_LATE_MAT).
//
// Installed IN PAIRS by the defer policy (scan_manager/late_mat_defer_policy)
// at query prepare — one deferred_scan_output on the producing scan, one
// port_materialize_directive on the consuming operator. Between the two, the
// deferred columns ride as ordinary data: a UINT64 pin-order rowid column at
// the first deferred position and 1-byte INT8 zero placeholders at the rest
// (arity and positions preserved, so no operator between the pair changes at
// all). Materialization happens at the consuming pipeline's input prepare
// (pipelineable_operator_data::prepare_for_processing), per the
// scheduling design: the Pk-finish/Pm-prepare boundary, on the task's stream,
// inside its reservation envelope.
//
// Both structs are immutable after install and shared by shared_ptr; when the
// gate is off neither is ever constructed.

#include "late_mat/column_origin.hpp"
#include "late_mat/late_materializer.hpp"  // pinned_table_layout / pinned_column_view

#include <cudf/types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sirius::op::scan {
class sirius_gpu_scan_operator;
}  // namespace sirius::op::scan

namespace sirius::late_mat {

/// Stamped on the producing GPU_SCAN: substitute these OUTPUT positions with
/// the rowid/placeholder columns at execute(). Positions are scan-output
/// positions (== materialized slots, the materialized-order mapping invariant).
struct deferred_scan_output {
  std::vector<std::size_t> output_positions;  ///< ascending; front() carries the rowid
  /// v2 count-on-deferred: emit the rowid as UINT32 (pinned rows < 2^32) —
  /// the counted ride never materializes, so the narrow width is pure ride
  /// savings. v1 and every port-materialized bundle stay UINT64.
  bool narrow_rowid{false};
  /// v2 count-on-deferred under a static scan filter: substitute BEFORE
  /// post_filter_and_project (view-splice; the filter then compacts the
  /// rowid column with the batch), valid because the policy proved the
  /// filter references no deferred position.
  bool pre_filter{false};
  [[nodiscard]] std::size_t rowid_position() const { return output_positions.front(); }
};

/// Stamped on the consuming operator: batches arriving at its input whose
/// table matches the placeholder signature below are transformed at prepare —
/// prepare_selection over the rowid column's ids (unsorted u64, gather
/// semantics), then materialize each deferred column and splice it in.
struct port_materialize_directive {
  /// Full expected post-substitution schema (review-hardened matcher input) —
  /// producer's planned types with every deferred position swapped to its
  /// placeholder type (UINT64/UINT32 rowid at each bundle's rowid position,
  /// INT8 elsewhere).
  std::size_t expected_arity{0};
  std::vector<cudf::type_id> expected_types;

  /// One materialization bundle per ORIGIN table contributing deferred
  /// columns at this port. v1/v2 installs carry exactly one; the v3
  /// FD/multi-origin ride carries the nominated origin plus one rider bundle
  /// per additional origin. Each bundle is independently the v2 contract:
  /// u64 ids at rowid_position (u32 riders are widened port-side with a cast
  /// temporary the materializer owns) -> prepare_selection -> materialize per
  /// column.
  struct origin_bundle {
    std::size_t rowid_position{0};       ///< this origin's rowid column at the port
    std::vector<std::size_t> positions;  ///< deferred positions at this port, ascending
    std::vector<column_origin> origins;  ///< parallel to positions
    pinned_table_layout layout;
    std::vector<pinned_column_view> columns;  ///< parallel to positions
  };
  std::vector<origin_bundle> bundles;

  /// Bundle value = Σ(real deferred widths, all bundles) − rowid bytes, in
  /// B/row (the attribution currency). Used by the consumer-slot arbitration:
  /// the widest install per consumer wins.
  double bundle_value_bytes{0.0};
  /// The scan this directive pairs with (nominated origin for v3 rides) — an
  /// arbitration eviction must clear the loser's scan-side substitution
  /// directives atomically with replacing this one.
  op::scan::sirius_gpu_scan_operator* source_scan{nullptr};
  /// Rider origins' scans (v3): their scan-side directives are cleared on
  /// eviction together with source_scan's.
  std::vector<op::scan::sirius_gpu_scan_operator*> rider_scans;
};

}  // namespace sirius::late_mat
