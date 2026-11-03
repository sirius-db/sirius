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

#include <cucascade/memory/memory_reservation_manager.hpp>

#include <algorithm>
#include <chrono>
#include <functional>
#include <thread>

namespace sirius::memory {

// Query backing may be retained indefinitely after unconfirmed GPU completion.
// Try each eligible NUMA node without blocking inside cuCascade, and let the
// caller surface its original cancellation or runtime error between attempts.
inline std::unique_ptr<cucascade::memory::reservation> reserve_host_memory(
  cucascade::memory::memory_reservation_manager& manager,
  cucascade::memory::any_memory_space_in_tier_with_preference const& request,
  std::size_t bytes,
  std::function<void()> const& check_running)
{
  auto candidates = request.get_candidates(manager);
  std::erase_if(candidates, [bytes](auto* space) { return bytes > space->get_max_memory(); });
  if (candidates.empty()) return nullptr;
  for (;;) {
    if (check_running) check_running();
    for (auto* space : candidates) {
      if (auto reservation = space->make_reservation_or_null(bytes)) return reservation;
    }
    // If only ordinary allocations consume capacity, no reservation holder can
    // signal progress. Preserve cuCascade's idle fallback: the block allocator
    // can charge the unreserved portion against the space's allocation limit.
    for (auto* space : candidates) {
      if (space->get_active_reservation_count() == 0) return space->make_reservation_upto(bytes);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
}  // namespace sirius::memory
