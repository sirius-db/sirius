// SPDX-License-Identifier: Apache-2.0
//
// Immutable caller policy shared by planning, reservations and decoding.

#pragma once

#include <cstddef>

namespace sirius::codegen {

/// Connection-local policy copied into each scan and decode request. A decoder
/// using a looser selectivity threshold could exceed its memory reservation.
struct fused_scan_policy {
  /// Promote decision traces to INFO / stderr.
  bool diagnostics = false;
  /// Maximum surviving-row fraction for compactable outputs. Dictionary
  /// gathers may bypass this ceiling, so their forecasts remain unbounded.
  double max_selectivity = 0.35;
  /// Full-width decode plus gather pays off only at low selectivity.
  double full_route_max_selectivity = 0.10;
  /// Prefer survivor-index enumeration below this fraction. A tiny positive
  /// value effectively disables the index walk.
  double index_walk_max_selectivity = 0.15;
  /// Maximum join probes carried into wave 1, ordered strongest first.
  /// Dropping the tail only under-filters: the join remains authoritative.
  std::size_t max_membership_sources = 1;
};

}  // namespace sirius::codegen
