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
#include <string>
#include <vector>

namespace sirius::exchange {

/// A process taking part in an exchange: 0 to world size - 1.
using rank_t = std::uint32_t;

/**
 * @brief Fixed-size frames in one `cudaMalloc` region.
 *
 * Each process has two: a send ring that batches are staged into, and a receive ring that peers
 * write into. They are the only memory the fabric registers, once. Plain `cudaMalloc`, never pool
 * memory: UCX's `cuda_ipc` cannot export stream-ordered allocations, and a pool block could be
 * handed to a kernel while a peer's write into it is still landing.
 */
class frame_ring {
 public:
  /// Allocates @p frames frames of @p frame_bytes each on the current device.
  /// @throws sirius::invalid_input_exception on zero frames, or a frame size that is zero or not
  ///         a multiple of 256 bytes.
  /// @throws sirius::internal_exception when the allocation fails, naming its size.
  frame_ring(std::uint32_t frames, std::size_t frame_bytes);
  ~frame_ring();

  // The fabric registers the base address, so the region never moves.
  frame_ring(frame_ring const&)            = delete;
  frame_ring& operator=(frame_ring const&) = delete;

  [[nodiscard]] std::uint32_t frames() const noexcept { return _frames; }
  [[nodiscard]] std::size_t frame_bytes() const noexcept { return _frame_bytes; }
  [[nodiscard]] std::size_t bytes() const noexcept { return _frames * _frame_bytes; }
  [[nodiscard]] std::byte* base() const noexcept { return _base; }
  /// Device address of frame @p frame, which must be in range.
  [[nodiscard]] std::byte* frame(std::uint32_t frame) const noexcept
  {
    return _base + static_cast<std::size_t>(frame) * _frame_bytes;
  }

 private:
  std::uint32_t _frames;
  std::size_t _frame_bytes;
  std::byte* _base{nullptr};
};

/// A notification a peer sent, alone or with a write.
struct fabric_notification {
  rank_t from;
  std::string body;
};

/**
 * @brief What the exchange needs from NIXL: one-sided writes into a peer's receive ring, and
 * small notifications.
 *
 * The transport's engine thread is the only caller. Nothing here blocks.
 */
class fabric {
 public:
  enum class write_status { pending, done, failed };

  virtual ~fabric() = default;

  [[nodiscard]] virtual rank_t rank() const noexcept       = 0;
  [[nodiscard]] virtual rank_t world_size() const noexcept = 0;

  /**
   * @brief Starts writing @p bytes from offset @p src of the local send ring to offset @p dst of
   * @p peer's receive ring.
   *
   * @p peer receives @p notification only once every byte has landed.
   * @return an id for status().
   */
  [[nodiscard]] virtual std::uint64_t write(
    rank_t peer, std::size_t src, std::size_t dst, std::size_t bytes, std::string notification) = 0;

  /// Where write @p id stands; done or failed forgets it. The send-ring bytes may be reused
  /// once it is done.
  [[nodiscard]] virtual write_status status(std::uint64_t id) = 0;

  /// Sends @p peer a notification with no data. Notifications and writes may arrive in any order.
  virtual void notify(rank_t peer, std::string notification) = 0;

  /// Appends the notifications received since the last call to @p out.
  virtual void poll(std::vector<fabric_notification>& out) = 0;
};

}  // namespace sirius::exchange
