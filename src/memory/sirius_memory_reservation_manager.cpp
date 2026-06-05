
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

#include "memory/sirius_memory_reservation_manager.hpp"

#include "cucascade/memory/common.hpp"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/memory_reservation_manager.hpp>

#include <memory>

namespace sirius {
namespace memory {

sirius_memory_reservation_manager::sirius_memory_reservation_manager(
  const std::vector<cucascade::memory::memory_space_config>& configs)
  : cucascade::memory::memory_reservation_manager(configs)
{
  auto gpu_spaces = this->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw std::runtime_error("At least one GPU memory space must be configured");
  }
  for (const auto* space : gpu_spaces) {
    int const dev        = space->get_device_id();
    auto const device_mr = space->get_default_allocator();
    rmm::cuda_set_device_raii set_device{rmm::cuda_device_id{dev}};
    prev_device_mrs_.push_back(cudf::get_current_device_resource_ref());
    cudf::set_current_device_resource_ref(device_mr);
    // FIX: install a forwarding shim as the LEGACY current device resource. cuDF default
    // allocations resolve through get_current_device_resource_ref, which (when no _ref is
    // effective) wraps the legacy resource — so this routes them to the shim. The shim
    // forwards to device_mr (== space->get_default_allocator()), which is cuCascade's
    // reservation_aware_resource_adaptor. This preserves reservation/spill tracking AND
    // keeps allocations stream-ordered (no raw, device-syncing cudaMalloc), because the
    // adaptor's upstream is cuCascade's stream-ordered pool.
    auto shim = std::make_unique<cucascade_forwarding_resource>(device_mr);
    rmm::mr::device_memory_resource* prev_legacy = rmm::mr::set_current_device_resource(shim.get());
    prev_legacy_mrs_.push_back({dev, prev_legacy});
    legacy_forwarding_mrs_.push_back(std::move(shim));
  }
}

sirius_memory_reservation_manager::~sirius_memory_reservation_manager()
{
  auto gpu_spaces = this->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  // Restore the previous cuDF device resources saved in the constructor.
  // Calling reset_current_device_resource_ref() would leave cuDF with a null/invalid
  // resource that crashes subsequent allocations in other tests or code paths.
  //
  // Before restoring the device resource refs, drain each GPU so any pending
  // stream-ordered frees forwarded through our shims to cuCascade's allocator
  // have completed. Without this drain, callers that leave async deallocations
  // un-synchronized (e.g., a TEST_CASE that lets its cuda_stream + data_batches
  // fall out of scope without an explicit sync) can leave in-flight work
  // referencing resources that are about to be torn down. The cost is a single
  // device sync per managed GPU at teardown.
  for (std::size_t i = 0; i < gpu_spaces.size() && i < prev_device_mrs_.size(); ++i) {
    rmm::cuda_set_device_raii set_device{rmm::cuda_device_id{gpu_spaces[i]->get_device_id()}};
    cudaDeviceSynchronize();
    cudf::set_current_device_resource_ref(prev_device_mrs_[i]);
  }
  // Restore the legacy current-device resource before our forwarding shims are destroyed
  // (members destruct after this body), so nothing points at a freed resource.
  for (auto const& [dev, prev_legacy] : prev_legacy_mrs_) {
    rmm::cuda_set_device_raii set_device{rmm::cuda_device_id{dev}};
    cudaDeviceSynchronize();
    if (prev_legacy) { rmm::mr::set_current_device_resource(prev_legacy); }
  }
}

}  // namespace memory
}  // namespace sirius
