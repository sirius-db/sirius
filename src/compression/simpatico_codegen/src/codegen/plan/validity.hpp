// SPDX-License-Identifier: Apache-2.0
//
// Transparent validity handling.
//
// Nulls are stripped off a column before the compress walk and reattached after
// the decode walk, so the plan DSL never mentions validity and no operator ever
// sees a nullable input. The stripped bitmask rides along as a sidecar on the
// column's PlanTree (see plan_tree.hpp) rather than as a routed channel.
//
// Two shapes cost nothing to carry and are special-cased:
//   * no nulls  -- no sidecar, no allocation, no copy; the input view is passed
//                  through untouched, so an all-valid column is byte-for-byte
//                  the same work it was before validity was supported at all.
//   * all nulls -- the bitmask is a constant, so only the kind is recorded; it
//                  is regenerated on decode and occupies no payload bytes.

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>

namespace simpatico {

/// How a column's validity is carried alongside its compression plan.
/// Serialized as a uint8 in the .hpln header -- do not renumber.
enum class validity_kind : std::uint8_t {
  all_valid = 0,  ///< No nulls. Nothing stored.
  all_null  = 1,  ///< Every row null. Nothing stored; the mask is regenerated on decode.
  mask      = 2,  ///< Mixed. `mask` holds the bitmask bytes for rows [0, size).
};

/// A column's validity, detached from its data.
struct validity_sidecar {
  validity_kind kind      = validity_kind::all_valid;
  std::int64_t null_count = 0;
  /// Bitmask bytes, populated iff kind == mask. Always rebased so bit i
  /// describes row i of the logical column, whatever the source view's offset.
  rmm::device_buffer mask;

  /// True when there is nothing to reattach (the common, all-valid case).
  bool empty() const { return kind == validity_kind::all_valid; }
};

/// Detach @p input's validity into @p out and return the same column with its
/// null mask dropped.
///
/// The returned view keeps @p input's offset and children, so it is exactly what
/// the walk would have received had the column never been nullable -- no
/// operator behaviour changes. It borrows @p input's data, so @p input must
/// outlive it.
cudf::column_view strip_validity(cudf::column_view input,
                                 validity_sidecar& out,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/// Reattach @p v to a freshly decoded column. A no-op when @p v is all_valid.
/// @p v is not consumed: a PlanTree can be decoded more than once, so the mask
/// is copied rather than moved.
void attach_validity(cudf::column& col,
                     validity_sidecar const& v,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr);

}  // namespace simpatico
