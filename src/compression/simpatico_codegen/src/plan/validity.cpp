// SPDX-License-Identifier: Apache-2.0
//
// strip_validity / attach_validity: the two ends of the transparent null path.
// See validity.hpp for the contract.

#include "codegen/plan/validity.hpp"

#include <cudf/null_mask.hpp>

#include <vector>

namespace simpatico {

cudf::column_view strip_validity(cudf::column_view input,
                                 validity_sidecar& out,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  auto const n     = input.size();
  auto const nulls = input.null_count();

  // Fast path: nothing to carry. No allocation, no copy, no kernel -- the view
  // reaches the walk exactly as it arrived. An all-valid column pays nothing for
  // nullability support, including in the .hpln header (see push_validity).
  if (nulls == 0 || n == 0) {
    out = validity_sidecar{};
    return input;
  }

  if (nulls == n) {
    // Fast path: every row is null, so the bitmask is a constant. Record the
    // kind alone -- nothing is copied here and nothing is serialized later;
    // attach_validity regenerates the mask from the row count.
    out.kind       = validity_kind::all_null;
    out.null_count = n;
    out.mask       = rmm::device_buffer{};
  } else {
    // copy_bitmask rebases a sliced view's mask to start at row 0, so the
    // sidecar always describes rows [0, size) of the logical column regardless
    // of input.offset().
    out.kind       = validity_kind::mask;
    out.null_count = nulls;
    out.mask       = cudf::copy_bitmask(input, stream, mr);
  }

  // The same column with its validity buffer dropped. Offset and children are
  // preserved, so stripping is behaviourally neutral: every operator sees what
  // it would have seen had the column arrived all-valid.
  std::vector<cudf::column_view> children;
  children.reserve(static_cast<std::size_t>(input.num_children()));
  for (cudf::size_type i = 0; i < input.num_children(); ++i) {
    children.push_back(input.child(i));
  }
  return cudf::column_view(input.type(),
                           n,
                           input.head<void>(),
                           /*null_mask=*/nullptr,
                           /*null_count=*/0,
                           input.offset(),
                           children);
}

void attach_validity(cudf::column& col,
                     validity_sidecar const& v,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  switch (v.kind) {
    case validity_kind::all_valid: return;

    case validity_kind::all_null:
      col.set_null_mask(cudf::create_null_mask(col.size(), cudf::mask_state::ALL_NULL, stream, mr),
                        col.size());
      return;

    case validity_kind::mask:
      // Copied, not moved: decompress_column takes the tree by const reference
      // and the same tree may be decoded repeatedly (e.g. a pinned chunk served
      // to several queries).
      col.set_null_mask(rmm::device_buffer(v.mask.data(), v.mask.size(), stream, mr),
                        static_cast<cudf::size_type>(v.null_count));
      return;
  }
}

}  // namespace simpatico
