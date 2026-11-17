
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

#include "memory/runtime_stream_pool.hpp"

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
  std::vector<runtime_stream_pool::registration> runtime_stream_pools_;
  void restore_device_resources() noexcept;

  struct device_resource_registration {
    int device_id;
    ::cuda::mr::any_resource<::cuda::mr::device_accessible> previous;
  };
  // Keep owning resources: references into RMM's default-resource map are
  // invalidated when its entries are replaced.
  std::vector<device_resource_registration> prev_device_mrs_;
};

}  // namespace memory
}  // namespace sirius
