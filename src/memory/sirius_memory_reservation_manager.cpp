
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

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/device_memory_resource.hpp>

#include <cucascade/memory/memory_reservation_manager.hpp>

#include <unordered_map>

namespace sirius {
namespace memory {

namespace {

std::unordered_map<int, rmm::mr::device_memory_resource*> create_per_device_memory_resource_map(
  cucascade::memory::memory_reservation_manager& manager)
{
  std::unordered_map<int, rmm::mr::device_memory_resource*> result;
  auto spaces = manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  for (auto* space : spaces) {
    if (space->get_tier() == cucascade::memory::Tier::GPU) {
      result.insert({space->get_device_id(), space->get_default_allocator()});
    }
  }
  return result;
}

}  // namespace

class sirius_device_memory_resource : public rmm::mr::device_memory_resource {
 public:
  sirius_device_memory_resource(sirius::memory::sirius_memory_reservation_manager& memory_manager)
    : per_device_device_memory_resource_map_(create_per_device_memory_resource_map(memory_manager))
  {
  }

  ~sirius_device_memory_resource() override = default;

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    auto dev_id = rmm::get_current_cuda_device();
    auto it     = per_device_device_memory_resource_map_.find(dev_id.value());
    if (it == per_device_device_memory_resource_map_.end()) {
      throw std::runtime_error("No device memory resource found for device id: " +
                               std::to_string(dev_id.value()));
    }
    return it->second->allocate(stream, bytes);
  }

  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override
  {
    auto dev_id = rmm::get_current_cuda_device();
    auto it     = per_device_device_memory_resource_map_.find(dev_id.value());
    assert(it != per_device_device_memory_resource_map_.end());
    it->second->deallocate(stream, ptr, bytes);
  }

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return this == &other;
  }
  // device memory resources per device id
  const std::unordered_map<int32_t, rmm::mr::device_memory_resource*>
    per_device_device_memory_resource_map_;
};

sirius_memory_reservation_manager::sirius_memory_reservation_manager(
  const std::vector<cucascade::memory::memory_space_config>& configs)
  : cucascade::memory::memory_reservation_manager(configs)
{
  sirius_device_memory_resource_ = std::make_unique<sirius_device_memory_resource>(*this);
  cudf::set_current_device_resource(sirius_device_memory_resource_.get());
}

sirius_memory_reservation_manager::~sirius_memory_reservation_manager()
{
  cudf::set_current_device_resource(nullptr);
}

}  // namespace memory
}  // namespace sirius
