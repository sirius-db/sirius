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

#include <cstddef>
#include <optional>
#include <vector>

namespace sirius {
namespace parallel {

/**
 * @brief Pick the next spill candidate for a downgrade request.
 *
 * Candidates arrive in eviction-policy order (most-recently-added first). Without a byte target
 * the pick is just the first not-yet-dispatched candidate. With one, the final pick is best-fit:
 * the smallest candidate that still covers the remaining deficit, so the tail of a request
 * spills the smallest batch that closes the gap instead of the next whole multi-GB partition.
 *
 * Within a repository this trades some of the newest-first re-materialization preference for
 * spilled-byte proportionality; the cross-repository sweep order is unchanged.
 *
 * @param candidate_bytes Byte size of each candidate in the source space, in policy order.
 * @param dispatched      Parallel flags; true = already dispatched, skip.
 * @param remaining_bytes Remaining deficit toward the request's byte target, if known.
 * @return Index of the candidate to dispatch next, or nullopt when none remain.
 */
inline std::optional<std::size_t> select_next_spill_candidate(
  const std::vector<std::size_t>& candidate_bytes,
  const std::vector<bool>& dispatched,
  std::optional<std::size_t> remaining_bytes)
{
  std::optional<std::size_t> first_unused;
  std::optional<std::size_t> best_fit;
  for (std::size_t i = 0; i < candidate_bytes.size(); ++i) {
    if (dispatched[i]) { continue; }
    if (!first_unused.has_value()) { first_unused = i; }
    if (remaining_bytes.has_value() && candidate_bytes[i] >= *remaining_bytes &&
        (!best_fit.has_value() || candidate_bytes[i] < candidate_bytes[*best_fit])) {
      best_fit = i;
    }
  }
  if (!remaining_bytes.has_value()) { return first_unused; }
  // No candidate covers the deficit alone: fall back to policy order and keep going.
  return best_fit.has_value() ? best_fit : first_unused;
}

}  // namespace parallel
}  // namespace sirius
