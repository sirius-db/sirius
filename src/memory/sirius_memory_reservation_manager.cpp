
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
  // Allocate bookkeeping before changing any process-wide resource.
  prev_device_mrs_.reserve(gpu_spaces.size());
  try {
    for (const auto* space : gpu_spaces) {
      auto const device_mr = space->get_default_allocator();
      auto const device_id = space->get_device_id();
      auto replacement     = ::cuda::mr::any_resource<::cuda::mr::device_accessible>{device_mr};
      auto previous =
        rmm::mr::set_per_device_resource(rmm::cuda_device_id{device_id}, std::move(replacement));
      prev_device_mrs_.push_back({device_id, std::move(previous)});
    }
  } catch (...) {
    restore_device_resources();
    throw;
  }
}

sirius_memory_reservation_manager::~sirius_memory_reservation_manager()
{
  restore_device_resources();
}

void sirius_memory_reservation_manager::restore_device_resources() noexcept
{
  int previous_device = -1;
  (void)cudaGetDevice(&previous_device);
  for (auto it = prev_device_mrs_.rbegin(); it != prev_device_mrs_.rend(); ++it) {
    auto& registration = *it;
    // Drain stream-ordered frees before the base class destroys GPU pools.
    // Restore the registration even when CUDA is already in an error state.
    if (cudaSetDevice(registration.device_id) == cudaSuccess) { (void)cudaDeviceSynchronize(); }
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{registration.device_id},
                                     std::move(registration.previous));
  }
  prev_device_mrs_.clear();
  if (previous_device >= 0) { (void)cudaSetDevice(previous_device); }
}

}  // namespace memory
}  // namespace sirius
