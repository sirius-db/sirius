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

#include "exchange_session.hpp"

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace duckdb {

/// Thread-safe singleton that manages:
/// 1. Shared send staging buffer (set once at startup, pre-registered with nixl)
/// 2. Active ExchangeSession (per-execution state, created/destroyed per fragment)
///
/// The session lifecycle:
/// - begin_session() creates a new ExchangeSession, stores as active
/// - Result collector uses get_active_session() during GPU execution
/// - take_session() moves the session out (Rust takes ownership)
/// - dropping the Rust-owned artifact releases the C++ session resources
class LastGPUBuffers {
 public:
  static LastGPUBuffers& GetInstance() {
    static LastGPUBuffers instance;
    return instance;
  }

  // --- Shared staging buffer (set once at startup) ---

  void SetStagingBuffer(uintptr_t addr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    staging_addr_ = addr;
    staging_size_ = size;
  }

  std::pair<uintptr_t, size_t> GetStagingBuffer() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {staging_addr_, staging_size_};
  }

  // --- Exchange session management ---

  /// Create a new exchange session. Returns session_id.
  /// The session is stored as active until take_session() moves it out.
  uint64_t BeginSession() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Carry forward the staging offset from the previous session so that
    // new sessions pack AFTER the old data. This prevents overwriting
    // staging data that is still referenced by pending_views in exchange tables.
    size_t carry_offset = 0;
    if (active_session_) {
      carry_offset = active_session_->staging_offset;
      completed_sessions_.push_back(std::move(active_session_));
    } else {
      // Check completed sessions for the highest offset.
      for (auto& s : completed_sessions_) {
        carry_offset = std::max(carry_offset, s->staging_offset);
      }
    }
    for (const auto& [session_id, staging_offset] : inflight_staging_offsets_) {
      (void)session_id;
      carry_offset = std::max(carry_offset, staging_offset);
    }
    auto session = std::make_unique<ExchangeSession>();
    session->session_id = next_session_id_++;
    session->staging_offset = carry_offset;
    session->staging_lease_base = staging_addr_;
    session->staging_lease_size = staging_size_;
    auto id = session->session_id;
    active_session_ = std::move(session);
    return id;
  }

  /// Get the active session (only valid while engine lock is held).
  /// Returns nullptr if no session is active.
  ExchangeSession* GetActiveSession() {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_session_.get();
  }

  /// Move the active session out of the singleton.
  /// After this call, active_session_ is nullptr.
  /// The caller (Rust) owns the session and its GPU resources.
  std::unique_ptr<ExchangeSession> TakeSession() {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::move(active_session_);
  }

  void RegisterInflightArtifact(uint64_t session_id, size_t staging_offset) {
    std::lock_guard<std::mutex> lock(mutex_);
    inflight_staging_offsets_[session_id] = staging_offset;
  }

  void ReleaseInflightArtifact(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    inflight_staging_offsets_.erase(session_id);
  }

  bool ShouldRetain() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_session_ != nullptr;
  }

  std::pair<int, std::vector<int>> GetPartitionConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      return {active_session_->partition_num, active_session_->partition_cols};
    }
    return {0, {}};
  }

  std::vector<int> GetProjectionIndices() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      return active_session_->projection_indices;
    }
    return {};
  }

  ExchangeCaptureMode GetCaptureMode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      return active_session_->capture_mode;
    }
    return ExchangeCaptureMode::MaterializeAndCapture;
  }

  void SetPartitionConfig(int num_partitions, std::vector<int> column_indices) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      active_session_->partition_num = num_partitions;
      active_session_->partition_cols = std::move(column_indices);
      active_session_->packed_partitions.clear();
    }
  }

  void SetProjectionIndices(std::vector<int> projection_indices) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      active_session_->projection_indices = std::move(projection_indices);
    }
  }

  void SetCaptureMode(ExchangeCaptureMode capture_mode) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      active_session_->capture_mode = capture_mode;
    }
  }

  /// Atomically reserve a contiguous region in the staging buffer.
  /// Returns the starting offset. If the reservation would exceed max_size,
  /// returns the current offset (caller checks and handles overflow).
  size_t ReserveStagingRegion(size_t aligned_size, size_t max_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_session_) return 0;
    size_t offset = active_session_->staging_offset;
    if (offset + aligned_size <= max_size) {
      active_session_->staging_offset = offset + aligned_size;
    }
    return offset;
  }

  void AccumulatePackedPartitions(std::vector<PackedPartition> partitions) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      active_session_->accumulate_partitions(std::move(partitions));
    }
  }

  void RetainData(std::shared_ptr<void> data) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      active_session_->retain(std::move(data));
    }
  }

  void AccumulatePackedBroadcast(PackedBroadcastEntry entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      active_session_->accumulate_broadcast(std::move(entry));
    }
  }

 private:
  LastGPUBuffers() = default;
  mutable std::mutex mutex_;

  // Shared staging buffer (set once at startup)
  uintptr_t staging_addr_ = 0;
  size_t staging_size_ = 0;

  // Active exchange session (per-execution, moved out before unlock)
  std::unique_ptr<ExchangeSession> active_session_;
  uint64_t next_session_id_ = 1;

  // Completed sessions waiting for end_session cleanup
  std::vector<std::unique_ptr<ExchangeSession>> completed_sessions_;
  std::unordered_map<uint64_t, size_t> inflight_staging_offsets_;
};

}  // namespace duckdb
