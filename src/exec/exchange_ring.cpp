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

#include "exec/exchange_ring.hpp"

#include "sirius/exception.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>

namespace sirius::exec {

frame_ring::frame_ring(int device, std::size_t frames, std::size_t frame_bytes)
  : _device{device}, _frames{frames}, _frame_bytes{frame_bytes}
{
  if (frames == 0 || frames > std::numeric_limits<std::uint32_t>::max()) {
    throw sirius::invalid_input_exception("frame ring: {} frames is out of range", frames);
  }
  if (frame_bytes == 0 || frame_bytes % rmm::CUDA_ALLOCATION_ALIGNMENT != 0) {
    throw sirius::invalid_input_exception(
      "frame ring: a frame of {} bytes is not a positive multiple of {}",
      frame_bytes,
      rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  if (frames > std::numeric_limits<std::size_t>::max() / frame_bytes) {
    throw sirius::invalid_input_exception(
      "frame ring: {} frames of {} bytes overflow", frames, frame_bytes);
  }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
  auto const status = cudaMalloc(&_base, bytes());
  if (status != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    throw sirius::internal_exception("frame ring: cudaMalloc of {} bytes on device {} failed: {}",
                                     bytes(),
                                     device,
                                     cudaGetErrorString(status));
  }
  _leased.assign(frames, false);
  _free.reserve(frames);
  // Lowest index on top, so frame 0 is leased first.
  for (std::size_t i = frames; i > 0; --i) {
    _free.push_back(static_cast<std::uint32_t>(i - 1));
  }
}

frame_ring::~frame_ring()
{
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
  static_cast<void>(cudaFree(_base));
}

std::size_t frame_ring::frames_in(std::size_t bytes, std::size_t frame_bytes)
{
  if (frame_bytes == 0) { return 1; }
  return std::max<std::size_t>(1, bytes / frame_bytes);
}

std::optional<std::uint32_t> frame_ring::lease()
{
  std::lock_guard const lock{_mutex};
  if (_free.empty()) { return std::nullopt; }
  auto const frame = _free.back();
  _free.pop_back();
  _leased[frame] = true;
  _peak          = std::max(_peak, _frames - _free.size());
  return frame;
}

void frame_ring::release(std::uint32_t frame)
{
  std::lock_guard const lock{_mutex};
  if (frame >= _frames || !_leased[frame]) {
    throw sirius::invalid_input_exception("frame ring: frame {} is not leased", frame);
  }
  _leased[frame] = false;
  _free.push_back(frame);
}

std::uintptr_t frame_ring::frame_address(std::uint32_t frame) const
{
  if (frame >= _frames) {
    throw sirius::invalid_input_exception(
      "frame ring: frame {} is out of range ({} frames)", frame, _frames);
  }
  return base() + static_cast<std::uintptr_t>(frame) * _frame_bytes;
}

std::size_t frame_ring::leased() const
{
  std::lock_guard const lock{_mutex};
  return _frames - _free.size();
}

std::size_t frame_ring::peak_leased() const
{
  std::lock_guard const lock{_mutex};
  return _peak;
}

}  // namespace sirius::exec
