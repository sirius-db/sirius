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
  /// Placeholder signature of a pre-materialization batch. Hardened (review finding F1):
  /// the matcher requires the FULL expected post-substitution schema — every
  /// column's cudf type id, i.e. the producer's planned types with deferred
  /// positions swapped to UINT64 (rowid) / INT8 (placeholders) — plus a
  /// content check on the rowid column (min/max within the pinned row range)
  /// before any gather. A batch that does not match is passed through
  /// untouched (e.g. the other port's batches in a mixed join task input);
  /// a silent-wrong now needs a whole-schema AND in-range coincidence.
  std::size_t expected_arity{0};
  std::vector<cudf::type_id> expected_types;  ///< full port schema, post-substitution
  std::vector<std::size_t> positions;  ///< deferred positions at this port, ascending
  std::size_t rowid_position{0};       ///< == positions.front() mapped to this port

  /// Origins parallel to `positions` (generation-checked at every use).
  std::vector<column_origin> origins;
  /// Bundle value = Σ(real deferred widths) − rowid bytes, in B/row (the
  /// attribution currency). Used by the consumer-slot arbitration: the
  /// widest bundle per consumer wins, so a 25-row dimension string can never
  /// lock out a 154 B/row payload bundle again.
  double bundle_value_bytes{0.0};
  /// The scan this directive pairs with — an arbitration eviction must clear
  /// the loser's scan-side substitution directive atomically with replacing
  /// this one (installs are plan-time, pre-execution, single-threaded).
  op::scan::sirius_gpu_scan_operator* source_scan{nullptr};
  /// Resolved once at install (pin storage is stable for the query): the
  /// origin table's layout and the per-column source views the late materializer
  /// consumes. `columns` is parallel to `positions`.
  pinned_table_layout layout;
  std::vector<pinned_column_view> columns;
};

}  // namespace sirius::late_mat
