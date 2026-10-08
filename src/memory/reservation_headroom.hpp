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

#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

namespace sirius::memory {

/// Snapshot of GPU reservation headroom. The allocator's tracked bytes include live
/// allocations AND reservations. get_available_memory() instead uses physical capacity,
/// which can exceed the configured reservation limit. Only a reservation guarantees space.
inline std::size_t gpu_reservation_headroom(const cucascade::memory::memory_space& space)
{
  auto used =
    space.get_memory_resource_of<cucascade::memory::Tier::GPU>()->get_total_allocated_bytes();
  auto limit = space.get_max_memory();
  return used < limit ? limit - used : 0;
}

}  // namespace sirius::memory
