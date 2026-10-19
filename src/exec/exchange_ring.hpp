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

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace sirius::exec {

/**
 * @brief A ring of fixed-size frames in one `cudaMalloc` region, for the exchange transport.
 *
 * The transport registers only these regions with NIXL: a send ring that batches are copied
 * into, and a receive ring that peers write into. Both are plain `cudaMalloc`, never pool
 * memory: UCX's `cuda_ipc` cannot export `cudaMallocAsync` allocations (it falls back to staged
 * host copies, ~220x slower), and the engine's pool can then trim on OOM. They are allocated
 * before the engine's pool, so the pool's budget is what is left.
 *
 * Frames are leased and returned by index. They all have the same size, so the ring cannot
 * fragment: any returned frame can serve any later lease. A batch larger than one frame streams
 * through as many as it needs.
 *
 * Thread-safe; the region is freed with the ring.
 */
class frame_ring {
 public:
  /// Allocates @p frames frames of @p frame_bytes each on @p device.
  /// @throws sirius::invalid_input_exception on zero frames, or a frame size that is zero or not
  ///         a multiple of 256 bytes.
  /// @throws sirius::internal_exception when the allocation fails, naming its size.
  frame_ring(int device, std::size_t frames, std::size_t frame_bytes);
  ~frame_ring();

  // The base address is registered with NIXL, so the region never moves.
  frame_ring(frame_ring const&)            = delete;
  frame_ring& operator=(frame_ring const&) = delete;
  frame_ring(frame_ring&&)                 = delete;
  frame_ring& operator=(frame_ring&&)      = delete;

  /// The number of whole frames of @p frame_bytes in @p bytes, at least 1.
  [[nodiscard]] static std::size_t frames_in(std::size_t bytes, std::size_t frame_bytes);

  /// A free frame's index, or nullopt when every frame is leased. Never blocks.
  [[nodiscard]] std::optional<std::uint32_t> lease();

  /// Returns a leased frame.
  /// @throws sirius::invalid_input_exception on an index out of range or not leased.
  void release(std::uint32_t frame);

  [[nodiscard]] int device() const noexcept { return _device; }
  [[nodiscard]] std::uintptr_t base() const noexcept
  {
    return reinterpret_cast<std::uintptr_t>(_base);
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return _frames * _frame_bytes; }
  [[nodiscard]] std::size_t frames() const noexcept { return _frames; }
  [[nodiscard]] std::size_t frame_bytes() const noexcept { return _frame_bytes; }
  /// Device address of frame @p frame.
  [[nodiscard]] std::uintptr_t frame_address(std::uint32_t frame) const;

  /// Frames leased now, and the most ever leased at once.
  [[nodiscard]] std::size_t leased() const;
  [[nodiscard]] std::size_t peak_leased() const;

 private:
  int _device;
  std::size_t _frames;
  std::size_t _frame_bytes;
  void* _base{nullptr};

  mutable std::mutex _mutex;
  /// Free frame indices; the most recently returned is leased first, so a lightly used ring
  /// keeps reusing the same few frames.
  std::vector<std::uint32_t> _free;
  std::vector<bool> _leased;
  std::size_t _peak{0};
};

}  // namespace sirius::exec
