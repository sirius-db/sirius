// SPDX-License-Identifier: Apache-2.0
#ifndef SIMPATICO_PLAN_COLUMN_COPY_HPP
#define SIMPATICO_PLAN_COLUMN_COPY_HPP

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <memory>

namespace simpatico {

/// Deep-copy a column_view into an owning column (for identity leaves).
/// Fixed-width types are copied via make_fixed_width_column + memcpy; STRING/LIST
/// are copied via an identity gather. For a LIST view with >=2 children the list's
/// element child (child 1) is copied.
std::unique_ptr<cudf::column> copy_column_view(cudf::column_view const& view,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr);

/// Copy a column view as raw UINT8 (for dictionary keys_chars when view
/// type/layout may vary). Copies the full byte size (size() * element_size) so
/// INT32 or other fixed-width types work.
std::unique_ptr<cudf::column> copy_column_view_as_uint8(cudf::column_view const& view,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr);

}  // namespace simpatico

#endif  // SIMPATICO_PLAN_COLUMN_COPY_HPP
