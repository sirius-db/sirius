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

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace sirius::memory {

template <typename Resource>
class host_device_resource_view {
 public:
  explicit host_device_resource_view(Resource& resource) noexcept
    : resource_{std::addressof(resource)}
  {
  }

  [[nodiscard]] Resource* get() const noexcept { return resource_; }

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    return resource_->allocate(stream, bytes, alignment);
  }

  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    resource_->deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return resource_->allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    resource_->deallocate_sync(ptr, bytes, alignment);
  }

  friend bool operator==(host_device_resource_view const& lhs,
                         host_device_resource_view const& rhs) noexcept
  {
    return lhs.resource_ == rhs.resource_;
  }

  friend void get_property(host_device_resource_view const&, ::cuda::mr::device_accessible) noexcept
  {
  }

  friend void get_property(host_device_resource_view const&, ::cuda::mr::host_accessible) noexcept
  {
  }

 private:
  Resource* resource_;
};

template <typename Resource>
  requires(
    ::cuda::mr::resource_with<Resource, ::cuda::mr::host_accessible, ::cuda::mr::device_accessible>)
host_device_resource_view<Resource> make_host_device_resource_view_checked(Resource* resource)
{
  if (resource == nullptr) { throw std::logic_error("Unexpected null resource pointer."); }
  return host_device_resource_view<Resource>{*resource};
}

}  // namespace sirius::memory
