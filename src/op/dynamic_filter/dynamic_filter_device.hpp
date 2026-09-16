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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <helper/numeric_narrowing.hpp>

#include <memory>

namespace sirius::op::detail {

// Negative IDs resolve to the current device; lookup failure returns -1.
[[nodiscard]] inline int resolve_dynamic_filter_device_id(int device_id) noexcept
{
  if (device_id >= 0) { return device_id; }
  int current = -1;
  return cudaGetDevice(&current) == cudaSuccess ? current : -1;
}

/// @p probe restored to @p want, or null when no restoration applies (the
/// caller then probes @p probe itself and declines if it is still the wrong
/// type).
///
/// A pinned chunk may store a join key NARROWED -- pin-time compressed
/// materialization casts each column to the narrowest carrier its values fit --
/// while the filter was published at the key's native carrier. Declining that
/// mismatch costs the chunk its whole decode-side compaction, so restore
/// instead. This is the same value-preserving widening scan normalization
/// applies after decode (@ref sirius::can_restore_to guards the carrier family,
/// signedness and scale), so a restored probe answers exactly as the native
/// column would have.
///
/// Only the widening direction is restored: narrowing a probe could map values
/// outside @p want onto keys that are in the set, which would be a false
/// positive rather than a declined probe.
[[nodiscard]] inline std::unique_ptr<cudf::column> restore_probe_to(
  cudf::column_view const& probe,
  cudf::data_type want,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (probe.type() == want || !sirius::can_restore_to(probe.type(), want)) { return nullptr; }
  return sirius::cast_through_rep(probe, want, stream, mr);
}

}  // namespace sirius::op::detail
