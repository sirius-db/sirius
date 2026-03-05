
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
  for (const auto* space : gpu_spaces) {
    auto* device_mr = space->get_default_allocator();
    rmm::cuda_set_device_raii set_device{rmm::cuda_device_id{space->get_device_id()}};
    prev_device_mrs_.push_back(cudf::set_current_device_resource(device_mr));
  }
}

sirius_memory_reservation_manager::~sirius_memory_reservation_manager()
{
  auto gpu_spaces = this->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  // Restore the previous cuDF device resources saved in the constructor.
  // Calling reset_current_device_resource_ref() would leave cuDF with a null/invalid
  // resource that crashes subsequent allocations in other tests or code paths.
  for (std::size_t i = 0; i < gpu_spaces.size() && i < prev_device_mrs_.size(); ++i) {
    rmm::cuda_set_device_raii set_device{rmm::cuda_device_id{gpu_spaces[i]->get_device_id()}};
    cudf::set_current_device_resource(prev_device_mrs_[i]);
  }
}

}  // namespace memory
}  // namespace sirius
