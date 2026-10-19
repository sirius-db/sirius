
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

#include <cuda_runtime_api.h>

#include <cucascade/memory/memory_reservation_manager.hpp>

#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <variant>

namespace sirius {
namespace memory {

namespace {

using allocation_tracking_scope = sirius_memory_reservation_manager::allocation_tracking_scope;

// Register live GPU-space addresses after construction and remove them before base-class teardown.
struct gpu_tracking_scope_registry {
  std::shared_mutex mutex;
  std::unordered_map<cucascade::memory::memory_space const*, allocation_tracking_scope> scopes;
};

gpu_tracking_scope_registry& gpu_tracking_scopes()
{
  static gpu_tracking_scope_registry registry;
  return registry;
}

}  // namespace

sirius_memory_reservation_manager::sirius_memory_reservation_manager(
  const std::vector<cucascade::memory::memory_space_config>& configs)
  : cucascade::memory::memory_reservation_manager(configs)
{
  auto gpu_spaces = this->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw std::runtime_error("At least one GPU memory space must be configured");
  }
  for (const auto* space : gpu_spaces) {
    auto const device_mr = space->get_default_allocator();
    rmm::cuda_set_device_raii set_device{rmm::cuda_device_id{space->get_device_id()}};
    // Capture the old resource by value (not by ref) — see comment in header for why.
    // Wrap device_async_resource_ref in any_resource to satisfy the non-deprecated API.
    prev_device_mrs_.push_back(cudf::set_current_device_resource(
      ::cuda::mr::any_resource<::cuda::mr::device_accessible>{device_mr}));
  }

  auto& registry = gpu_tracking_scopes();
  std::unique_lock lock{registry.mutex};
  try {
    for (auto const& config : configs) {
      auto const* gpu = std::get_if<cucascade::memory::gpu_memory_space_config>(&config);
      if (gpu == nullptr) { continue; }
      registry.scopes.insert_or_assign(
        get_memory_space(cucascade::memory::Tier::GPU, gpu->device_id),
        gpu->per_stream_reservation ? allocation_tracking_scope::PER_STREAM
                                    : allocation_tracking_scope::PER_THREAD);
    }
  } catch (...) {
    for (auto const* space : gpu_spaces) {
      registry.scopes.erase(space);
    }
    throw;
  }
}

sirius_memory_reservation_manager::~sirius_memory_reservation_manager()
{
  auto gpu_spaces = this->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  {
    auto& registry = gpu_tracking_scopes();
    std::unique_lock lock{registry.mutex};
    for (auto const* space : gpu_spaces) {
      registry.scopes.erase(space);
    }
  }
  // Restore the previous cuDF device resources saved in the constructor.
  // Calling reset_current_device_resource_ref() would leave cuDF with a null/invalid
  // resource that crashes subsequent allocations in other tests or code paths.
  //
  // Before restoring the device resource refs, drain each GPU so any pending
  // stream-ordered frees (cudaFreeAsync) against the cuda_async_memory_resource
  // pool we are about to destroy have completed. Without this drain, callers
  // that leave async deallocations un-synchronized (e.g., a TEST_CASE that lets
  // its cuda_stream + data_batches fall out of scope without an explicit sync)
  // can corrupt the driver's per-device pool list, which then crashes the next
  // sirius_memory_reservation_manager that constructs a fresh pool on the same
  // device. The cost is a single device sync per managed GPU at teardown.
  for (std::size_t i = 0; i < gpu_spaces.size() && i < prev_device_mrs_.size(); ++i) {
    rmm::cuda_set_device_raii set_device{rmm::cuda_device_id{gpu_spaces[i]->get_device_id()}};
    cudaDeviceSynchronize();
    cudf::set_current_device_resource(std::move(prev_device_mrs_[i]));
  }
}

std::optional<allocation_tracking_scope>
sirius_memory_reservation_manager::gpu_allocation_tracking_scope(
  cucascade::memory::memory_space const& space)
{
  auto& registry = gpu_tracking_scopes();
  std::shared_lock lock{registry.mutex};
  auto const found = registry.scopes.find(&space);
  if (found == registry.scopes.end()) { return std::nullopt; }
  return found->second;
}

}  // namespace memory
}  // namespace sirius
