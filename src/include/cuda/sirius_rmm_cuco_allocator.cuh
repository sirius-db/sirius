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

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>

namespace sirius {

/// @brief cuCollections allocator backed by RMM's stream-ordered pool allocator.
template <class T>
class rmm_cuco_allocator {
 public:
  using value_type = T;
  explicit rmm_cuco_allocator(rmm::device_async_resource_ref mr) noexcept : _mr{mr} {}
  template <class U>
  rmm_cuco_allocator(rmm_cuco_allocator<U> const& other) noexcept : _mr{other.resource()}
  {
  }

  value_type* allocate(std::size_t n, cuda::stream_ref stream)
  {
    return static_cast<value_type*>(_mr.allocate(
      rmm::cuda_stream_view{stream.get()}, n * sizeof(value_type), rmm::CUDA_ALLOCATION_ALIGNMENT));
  }
  void deallocate(value_type* p, std::size_t n, cuda::stream_ref stream) noexcept
  {
    _mr.deallocate(rmm::cuda_stream_view{stream.get()},
                   p,
                   n * sizeof(value_type),
                   rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  [[nodiscard]] rmm::device_async_resource_ref resource() const noexcept { return _mr; }

 private:
  rmm::device_async_resource_ref _mr;
};

}  // namespace sirius
