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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace sirius {
namespace spill {

/**
 * @brief One pending (dst, src, size) copy operation.
 *
 * Pointers are opaque here: the flush callback decides how to submit them, which keeps the
 * chunking logic CUDA-free and unit-testable.
 */
struct copy_op {
  void* dst;
  const void* src;
  std::size_t size;
};

/**
 * @brief Accumulates copy ops and hands them to a flush callback in bounded-size chunks.
 *
 * An op larger than the remaining chunk capacity is split, so no single submission (and no
 * single op within it) ever exceeds @p chunk_bytes. Flushing incrementally lets the DMA engine
 * start on the first chunk while the caller is still collecting the rest.
 *
 * The caller must invoke flush_pending() after the last add(); the batcher never flushes a
 * partial chunk on its own.
 */
class chunked_copy_batcher {
 public:
  using flush_fn = std::function<void(std::span<const copy_op>)>;

  /**
   * @param chunk_bytes Target bytes per flush; 0 means unbounded (a single flush at the end).
   * @param flush       Callback invoked with each chunk's ops. Must not retain the span.
   */
  chunked_copy_batcher(std::size_t chunk_bytes, flush_fn flush)
    : _chunk_bytes(chunk_bytes == 0 ? std::numeric_limits<std::size_t>::max() : chunk_bytes),
      _flush(std::move(flush))
  {
  }

  /// @brief Queue a copy, splitting it at chunk boundaries and flushing full chunks. Zero-size
  /// and null ops are ignored, mirroring the builtin converter's accumulator.
  void add(void* dst, const void* src, std::size_t size)
  {
    if (size == 0 || dst == nullptr || src == nullptr) { return; }
    _bytes_added += size;
    std::size_t offset = 0;
    while (offset < size) {
      // _pending_bytes < _chunk_bytes here (flush_now resets it), so room > 0.
      std::size_t const room  = _chunk_bytes - _pending_bytes;
      std::size_t const piece = std::min(size - offset, room);
      _pending.push_back(copy_op{
        static_cast<std::byte*>(dst) + offset, static_cast<const std::byte*>(src) + offset, piece});
      _pending_bytes += piece;
      offset += piece;
      if (_pending_bytes >= _chunk_bytes) { flush_now(); }
    }
  }

  /// @brief Flush any partially-filled final chunk. Idempotent.
  void flush_pending()
  {
    if (!_pending.empty()) { flush_now(); }
  }

  /// @brief Number of flush callbacks issued so far.
  [[nodiscard]] std::size_t chunks_flushed() const noexcept { return _chunks_flushed; }

  /// @brief Total bytes passed through add() (before splitting).
  [[nodiscard]] std::size_t bytes_added() const noexcept { return _bytes_added; }

  /// @brief Largest single flush submission in bytes (<= chunk_bytes by construction).
  [[nodiscard]] std::size_t largest_submission_bytes() const noexcept
  {
    return _largest_submission_bytes;
  }

 private:
  void flush_now()
  {
    _largest_submission_bytes = std::max(_largest_submission_bytes, _pending_bytes);
    ++_chunks_flushed;
    _flush(std::span<const copy_op>(_pending.data(), _pending.size()));
    _pending.clear();
    _pending_bytes = 0;
  }

  std::size_t _chunk_bytes;
  flush_fn _flush;
  std::vector<copy_op> _pending;
  std::size_t _pending_bytes{0};
  std::size_t _bytes_added{0};
  std::size_t _chunks_flushed{0};
  std::size_t _largest_submission_bytes{0};
};

}  // namespace spill
}  // namespace sirius
