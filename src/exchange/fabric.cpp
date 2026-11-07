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

#include "exchange/fabric.hpp"

#include "sirius/exception.hpp"

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <limits>

namespace sirius::exchange {

frame_ring::frame_ring(std::uint32_t frames, std::size_t frame_bytes, ring_memory memory)
  : _frames{frames}, _frame_bytes{frame_bytes}
{
  if (frames == 0) { throw sirius::invalid_input_exception("frame ring: no frames"); }
  if (frame_bytes == 0 || frame_bytes % rmm::CUDA_ALLOCATION_ALIGNMENT != 0) {
    throw sirius::invalid_input_exception(
      "frame ring: a frame of {} bytes is not a positive multiple of {}",
      frame_bytes,
      rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  if (frame_bytes > std::numeric_limits<std::size_t>::max() / frames) {
    throw sirius::invalid_input_exception(
      "frame ring: {} frames of {} bytes overflow", frames, frame_bytes);
  }
  if (memory != ring_memory::plain) {
    int device = 0;
    RMM_CUDA_TRY(cudaGetDevice(&device));
    _fabric = fabric_memory::try_allocate(device, bytes(), _why_not_fabric);
    if (_fabric) {
      _base = _fabric->base();
      _why_not_fabric.clear();
      return;
    }
    if (memory == ring_memory::fabric) {
      throw sirius::internal_exception(
        "frame ring: no fabric memory for {} bytes: {}", bytes(), _why_not_fabric);
    }
  }
  void* base        = nullptr;
  auto const status = cudaMalloc(&base, bytes());
  if (status != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    throw sirius::internal_exception(
      "frame ring: cudaMalloc of {} bytes failed: {}", bytes(), cudaGetErrorString(status));
  }
  _base = static_cast<std::byte*>(base);
}

frame_ring::~frame_ring()
{
  if (!_fabric) { static_cast<void>(cudaFree(_base)); }
}

}  // namespace sirius::exchange
