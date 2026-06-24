/*
 * Copyright 2026, Sirius Contributors.
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
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <cstddef>

namespace sirius {

/// cuCollections allocator backed by RMM's stream-ordered pool — the same compute-stream memory
/// resource the rest of the query uses — instead of cuco's default synchronizing
/// cudaMalloc/cudaFree. cuco's allocator interface is stream-aware (allocate(n, stream)), matching
/// RMM's stream-ordered resource. This keeps cuco device structures (Bloom bit arrays, static_set
/// storage) inside the RMM pool and avoids the implicit device sync of a raw cudaMalloc on the
/// hot path. Shared by the dynamic-filter device structures (.cu translation units only).
template <class T>
class rmm_cuco_allocator {
 public:
  using value_type = T;
  explicit rmm_cuco_allocator(rmm::device_async_resource_ref mr) noexcept : _mr{mr} {}
  template <class U>
  rmm_cuco_allocator(rmm_cuco_allocator<U> const& other) noexcept : _mr{other.resource()}
  {
  }

  value_type* allocate(std::size_t n, ::cuda::stream_ref stream)
  {
    return static_cast<value_type*>(
      _mr.allocate(rmm::cuda_stream_view{stream.get()}, n * sizeof(value_type)));
  }
  void deallocate(value_type* p, std::size_t n, ::cuda::stream_ref stream) noexcept
  {
    _mr.deallocate(rmm::cuda_stream_view{stream.get()}, p, n * sizeof(value_type));
  }
  [[nodiscard]] rmm::device_async_resource_ref resource() const noexcept { return _mr; }

 private:
  rmm::device_async_resource_ref _mr;
};

}  // namespace sirius
