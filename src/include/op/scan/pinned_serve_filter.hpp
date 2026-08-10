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

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <memory>

namespace sirius::op {
class sirius_dynamic_filter_set;
}  // namespace sirius::op

namespace sirius::op::scan {

/// Outcome of a serve-side filtered upload attempt (all counts are per split).
struct pinned_serve_filter_result {
  bool applied              = false;  ///< the batch now holds a filtered GPU representation
  std::size_t rows_in       = 0;
  std::size_t rows_out      = 0;
  std::size_t bytes_moved   = 0;  ///< bytes that crossed to the device (key column + survivors)
  std::size_t bytes_avoided = 0;  ///< bulk-upload bytes that never crossed
};

/// A serve-side filter keeping more than this fraction of a split's rows is not worth the
/// key-column upload + random host gather; the split falls back to the bulk upload. Tighter than
/// the post-decode gate's 0.9 because the gather reads host memory at random, which only beats a
/// sequential bulk copy at real selectivity.
inline constexpr double k_serve_filter_keep_threshold = 0.5;

/**
 * @brief Pre-transfer dynamic filtering of a host-pinned cached split — upload the filter's key
 * column, probe the published membership filter on device, then gather only surviving rows of the
 * remaining columns straight out of the mapped pinned blocks. On success the batch's data is
 * replaced with the filtered GPU table (same schema, fewer rows).
 *
 * Correctness mirrors the post-decode DYNAMIC_FILTER checkpoint: the applied predicate is a
 * conjunctive superset of the join condition, applied per split against an opportunistic channel
 * snapshot, so dropping rows here is exactly as safe as dropping them one operator later. Only one
 * membership filter is applied; any further channel filters still run at the post-decode
 * checkpoint on the reduced rows.
 *
 * Falls back (returns applied=false, batch untouched) whenever the fast path does not hold:
 * non-HOST tier or compressed representation, any nullable / string / nested / non-fixed-width
 * column, no device-local mask-capable filter for a served column, a filter/key type mismatch, or
 * a keep ratio above @ref k_serve_filter_keep_threshold.
 */
pinned_serve_filter_result try_serve_filtered_upload(
  cucascade::data_batch& batch,
  sirius_dynamic_filter_set const& filters,
  cucascade::memory::memory_space const* target_space,
  rmm::cuda_stream_view stream);

}  // namespace sirius::op::scan
