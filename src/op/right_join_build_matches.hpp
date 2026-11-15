/*
 * Copyright 2025, Sirius Contributors.
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

#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream_ref>

#include <cstdint>
#include <memory>

namespace sirius::op {

/// Set `matched[row]` for every build row in `build_rows`. Concurrent calls over one array are
/// safe: every write stores the same value.
void mark_build_matches(rmm::device_uvector<std::uint8_t>& matched,
                        rmm::device_uvector<cudf::size_type> const& build_rows,
                        ::cuda::stream_ref stream);

/// The build rows never marked, ascending; every row when `matched` is null.
std::unique_ptr<rmm::device_uvector<cudf::size_type>> unmatched_build_rows(
  rmm::device_uvector<std::uint8_t> const* matched,
  cudf::size_type num_rows,
  ::cuda::stream_ref stream);

/// Add one to `counts[build_rows[i]]` for every pair `i` whose probe row matched and, given
/// `counted_mask`, holds a valid value at bit `probe_rows[i] + counted_offset`. Concurrent calls
/// over one array are safe.
void count_build_matches(rmm::device_uvector<std::int64_t>& counts,
                         rmm::device_uvector<cudf::size_type> const& probe_rows,
                         rmm::device_uvector<cudf::size_type> const& build_rows,
                         cudf::bitmask_type const* counted_mask,
                         cudf::size_type counted_offset,
                         ::cuda::stream_ref stream);

/// Replace each zero count by `unmatched_fill`.
void fill_unmatched_counts(rmm::device_uvector<std::int64_t>& counts,
                           std::int64_t unmatched_fill,
                           ::cuda::stream_ref stream);

/// `count` gather indices of `cudf::JoinNoMatch`, which a NULLIFY gather turns into NULL rows.
std::unique_ptr<rmm::device_uvector<cudf::size_type>> no_match_rows(std::size_t count,
                                                                     ::cuda::stream_ref stream);

}  // namespace sirius::op
