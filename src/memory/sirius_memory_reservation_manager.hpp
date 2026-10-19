
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

#include <cuda/memory_resource>

#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <optional>
#include <vector>

namespace sirius {
namespace memory {

class sirius_memory_reservation_manager : public cucascade::memory::memory_reservation_manager {
 public:
  using allocation_tracking_scope =
    cucascade::memory::reservation_aware_resource_adaptor::AllocationTrackingScope;

  explicit sirius_memory_reservation_manager(
    const std::vector<cucascade::memory::memory_space_config>& configs);

  ~sirius_memory_reservation_manager();

  /**
   * @brief Look up the configured allocation tracking scope of a live manager's GPU space.
   *
   * Lookups are synchronized; the caller must keep @p space alive. cuCascade does not expose this
   * configuration, so Sirius records it during manager construction.
   *
   * @param space Memory space to look up
   * @return The registered scope, or std::nullopt if the space is unregistered
   */
  [[nodiscard]] static std::optional<allocation_tracking_scope> gpu_allocation_tracking_scope(
    cucascade::memory::memory_space const& space);

 private:
  // Previous cuDF device resources, saved in constructor and restored in destructor.
  // Stored as owning any_resource (not as non-owning device_async_resource_ref) because
  // rmm 26.06 moved the per-device resource map to store any_resource by value: calling
  // set_current_device_resource_ref() moves the old map entry out, so any previously
  // captured resource_ref into that entry dangles. Capturing the *return value* of
  // set_current_device_resource_ref (the evicted old resource) avoids this.
  std::vector<::cuda::mr::any_resource<::cuda::mr::device_accessible>> prev_device_mrs_;
};

}  // namespace memory
}  // namespace sirius
