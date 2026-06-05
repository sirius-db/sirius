
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cucascade/memory/memory_reservation_manager.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace sirius {
namespace memory {

// Forwarding RMM device_memory_resource that routes all allocations to a
// cuCascade reservation-aware allocator (held as an rmm::device_async_resource_ref,
// obtained from memory_space::get_default_allocator()).
//
// Why this exists: cuDF default allocations resolve through
// cudf::get_current_device_resource_ref(). Installing the pool via the
// cudf::set_current_device_resource_ref(...) API proved to be a no-op in this build
// (worker threads kept seeing the raw, device-syncing cuda_memory_resource default,
// causing ~29k synchronous cudaMalloc). The LEGACY API
// rmm::mr::set_current_device_resource(device_memory_resource*) DOES take effect:
// get_current_device_resource_ref() wraps the legacy current resource when no _ref is
// set. So we install an instance of this shim as the legacy current device resource.
//
// Because the shim forwards to the ref returned by get_default_allocator() — which
// points at cuCascade's reservation_aware_resource_adaptor — reservation/spill
// tracking is preserved, and forwarding is stream-ordered (no raw cudaMalloc):
// the adaptor's upstream is cuCascade's stream-ordered pool.
class cucascade_forwarding_resource final : public rmm::mr::device_memory_resource {
 public:
  explicit cucascade_forwarding_resource(rmm::device_async_resource_ref upstream) noexcept
    : upstream_(upstream)
  {
  }

 private:
  // Forward to the upstream reservation-aware allocator on `stream` (async, no sync).
  // Mirrors RMM's own adaptor pattern (e.g. failure_callback_resource_adaptor):
  // device_async_resource_ref::allocate(stream, bytes).
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    return upstream_.allocate(stream, bytes);
  }

  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override
  {
    upstream_.deallocate(stream, ptr, bytes);
  }

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto const* cast = dynamic_cast<cucascade_forwarding_resource const*>(&other);
    if (cast == nullptr) { return false; }
    return upstream_ == cast->upstream_;
  }

  rmm::device_async_resource_ref upstream_;
};

class sirius_memory_reservation_manager : public cucascade::memory::memory_reservation_manager {
 public:
  explicit sirius_memory_reservation_manager(
    const std::vector<cucascade::memory::memory_space_config>& configs);

  ~sirius_memory_reservation_manager();

 private:
  // Previous cuDF device resources, saved in constructor and restored in destructor
  // to prevent dangling references after our custom GPU allocators are torn down.
  std::vector<rmm::device_async_resource_ref> prev_device_mrs_;
  // Per-GPU forwarding shims installed as the LEGACY current device resource. Each
  // shim forwards to cuCascade's reservation-aware allocator for that device. Owned
  // here so they outlive their installation; the destructor restores the previous
  // legacy resource before these are destroyed (members destruct after the dtor body).
  std::vector<std::unique_ptr<cucascade_forwarding_resource>> legacy_forwarding_mrs_;
  std::vector<std::pair<int, rmm::mr::device_memory_resource*>> prev_legacy_mrs_;
};

}  // namespace memory
}  // namespace sirius
