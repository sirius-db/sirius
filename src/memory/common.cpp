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

#include "memory/common.hpp"

#include "memory/null_device_memory_resource.hpp"
#include "memory/numa_region_pinned_host_allocator.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

namespace sirius {

namespace memory {

const char* memory_error_category::name() const noexcept { return "SiriusMemorySystem"; }

std::string memory_error_category::message(int ev) const
{
  switch (static_cast<MemoryError>(ev)) {
    case MemoryError::SUCCESS: return "Success";
    case MemoryError::ALLOCATION_FAILED: return "System allocation failed";
    case MemoryError::LIMIT_EXCEEDED: return "Reservation limit exceeded";
    case MemoryError::POOL_EXHAUSTED: return "Internal memory pool exhausted";
    default: return "Unknown memory error";
  }
}

const memory_error_category& memory_category()
{
  static const memory_error_category instance;
  return instance;
}

std::error_code make_error_code(MemoryError e)
{
  return std::error_code(static_cast<int>(e), memory_category());
}

sirius_out_of_memory::sirius_out_of_memory(std::string_view message,
                                           std::size_t requested_bytes,
                                           std::size_t global_usage)
  : rmm::out_of_memory(message.data()), requested_bytes(requested_bytes), global_usage(global_usage)
{
}

std::unique_ptr<rmm::mr::device_memory_resource> make_default_gpu_memory_resource(int device_id,
                                                                                  size_t limit,
                                                                                  size_t capacity)
{
  rmm::cuda_set_device_raii set_device(rmm::cuda_device_id{device_id});
  return std::make_unique<rmm::mr::cuda_async_memory_resource>(capacity);
}

std::unique_ptr<rmm::mr::device_memory_resource> make_default_host_memory_resource(int numa_node_id,
                                                                                   size_t limit,
                                                                                   size_t capacity)
{
  return std::make_unique<sirius::memory::numa_region_pinned_host_memory_resource>(numa_node_id);
}

DeviceMemoryResourceFactoryFn make_default_allocator_for_tier(Tier tier)
{
  if (tier == Tier::GPU) {
    return make_default_gpu_memory_resource;
  } else if (tier == Tier::HOST) {
    return make_default_host_memory_resource;
  } else {
    return [](int, size_t, size_t) { return std::make_unique<null_device_memory_resource>(); };
  }
}

}  // namespace memory
}  // namespace sirius

namespace std {
size_t hash<sirius::memory::memory_space_id>::operator()(
  const sirius::memory::memory_space_id& p) const
{
  return p.uuid();
}
}  // namespace std
