
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
#include "log/logging.hpp"
#include "memory/pool_peer_access.hpp"

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
  // Complete grants before installing global resource references.
  for_each_pair_without_pool_access(gpu_spaces, [](int owner_device, int accessor_device) {
    SIRIUS_LOG_INFO(
      "Pool access unavailable: owner GPU {}, accessor GPU {}", owner_device, accessor_device);
    return true;
  });
  // Allocate bookkeeping before changing process-wide resources.
  prev_device_mrs_.reserve(gpu_spaces.size());
  runtime_stream_pools_.reserve(gpu_spaces.size());
  auto& registry = gpu_tracking_scopes();
  try {
    {
      std::unique_lock lock{registry.mutex};
      for (auto const& config : configs) {
        auto const* gpu = std::get_if<cucascade::memory::gpu_memory_space_config>(&config);
        if (gpu == nullptr) { continue; }
        registry.scopes.insert_or_assign(
          get_memory_space(cucascade::memory::Tier::GPU, gpu->device_id),
          gpu->per_stream_reservation ? allocation_tracking_scope::PER_STREAM
                                      : allocation_tracking_scope::PER_THREAD);
      }
    }
    for (const auto* space : gpu_spaces) {
      runtime_stream_pools_.push_back(runtime_stream_pool::install(*space));
      auto const device_id = space->get_device_id();
      auto replacement =
        ::cuda::mr::any_resource<::cuda::mr::device_accessible>{space->get_default_allocator()};
      auto previous =
        rmm::mr::set_per_device_resource(rmm::cuda_device_id{device_id}, std::move(replacement));
      prev_device_mrs_.push_back({device_id, std::move(previous)});
    }
  } catch (...) {
    {
      std::unique_lock lock{registry.mutex};
      for (auto const* space : gpu_spaces) {
        registry.scopes.erase(space);
      }
    }
    restore_device_resources();
    throw;
  }
}

sirius_memory_reservation_manager::~sirius_memory_reservation_manager()
{
  auto gpu_spaces = get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  {
    auto& registry = gpu_tracking_scopes();
    std::unique_lock lock{registry.mutex};
    for (auto const* space : gpu_spaces) {
      registry.scopes.erase(space);
    }
  }
  restore_device_resources();
}

void sirius_memory_reservation_manager::restore_device_resources() noexcept
{
  int previous_device = -1;
  (void)cudaGetDevice(&previous_device);
  for (std::size_t i = prev_device_mrs_.size(); i > 0; --i) {
    auto& registration = prev_device_mrs_[i - 1];
    if (cudaSetDevice(registration.device_id) == cudaSuccess) {
      auto const status = cudaDeviceSynchronize();
      if (status != cudaSuccess) {
        log::log_noexcept([&] {
          SIRIUS_LOG_ERROR("GPU {} teardown synchronization failed: {}; continuing cleanup",
                           registration.device_id,
                           cudaGetErrorString(status));
        });
      }
    }
    runtime_stream_pools_[i - 1].reset();
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{registration.device_id},
                                     std::move(registration.previous));
  }
  prev_device_mrs_.clear();
  runtime_stream_pools_.clear();
  if (previous_device >= 0) { (void)cudaSetDevice(previous_device); }
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
