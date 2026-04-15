/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace sirius::op::scan {

/// @brief Pre-allocated ring buffer for pipelined H2D + GPU decode.
///
/// Each slot holds a pinned host buffer (for gathering block data) and a
/// device staging buffer (for bulk H2D).  Two CUDA events per slot track
/// H2D completion and compute completion for inter-stream synchronization.
///
/// Designed for double-buffering (2 slots): while the copy engine transfers
/// chunk N+1, the compute SMs decode chunk N.
class scan_ring_buffer {
 public:
  enum class slot_state { FREE, TRANSFERRING, COMPUTING };

  struct slot {
    void* host_pinned = nullptr;    ///< Page-locked host memory (cudaHostAlloc)
    void* device_staging = nullptr; ///< GPU staging memory (cudaMalloc)
    cudaEvent_t h2d_done = nullptr;     ///< Recorded after H2D completes on copy_stream
    cudaEvent_t compute_done = nullptr; ///< Recorded after decode completes on compute_stream
    slot_state state = slot_state::FREE;
    size_t used_bytes = 0;          ///< Actual bytes in use this iteration
  };

  /// @brief Construct ring buffer with pre-allocated slots.
  /// @param num_slots Number of slots (typically 2 for double-buffering)
  /// @param slot_bytes Size of each slot in bytes (host and device each)
  scan_ring_buffer(size_t num_slots, size_t slot_bytes);

  /// @brief Destroy ring buffer, freeing all pinned and device memory.
  ~scan_ring_buffer();

  // Non-copyable, non-movable
  scan_ring_buffer(const scan_ring_buffer&) = delete;
  scan_ring_buffer& operator=(const scan_ring_buffer&) = delete;
  scan_ring_buffer(scan_ring_buffer&&) = delete;
  scan_ring_buffer& operator=(scan_ring_buffer&&) = delete;

  /// @brief Acquire the next free slot.  Blocks (backpressure) if the slot
  ///        is still in TRANSFERRING or COMPUTING state by waiting on its
  ///        compute_done event.
  slot& acquire();

  /// @brief Get the slot size in bytes.
  [[nodiscard]] size_t slot_bytes() const noexcept { return slot_bytes_; }

  /// @brief Get the number of slots.
  [[nodiscard]] size_t num_slots() const noexcept { return slots_.size(); }

  /// @brief Access a slot by index (for final drain).
  [[nodiscard]] slot& get_slot(size_t idx) { return slots_[idx % slots_.size()]; }

  /// @brief Index of the last acquired slot.
  [[nodiscard]] size_t last_acquired_idx() const noexcept
  {
    return next_slot_ == 0 ? slots_.size() - 1 : next_slot_ - 1;
  }

 private:
  std::vector<slot> slots_;
  size_t next_slot_ = 0;
  size_t slot_bytes_;
};

}  // namespace sirius::op::scan
