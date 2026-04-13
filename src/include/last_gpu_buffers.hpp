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
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {

/// GPU buffer metadata for a single column (legacy — used by detect_execution_location).
struct GPUBufferInfo {
  uintptr_t addr;          ///< GPU device pointer (data buffer).
  size_t len;              ///< Data buffer size in bytes.
  int device_id;           ///< GPU device ID.
  std::string column_name; ///< Column name from the query plan.
  int type_id;             ///< cudf::type_id as int.
  size_t num_rows;         ///< Number of rows in the column.
  uintptr_t null_mask_addr = 0; ///< Null bitmap GPU pointer (0 if all-valid).
  size_t null_mask_len     = 0; ///< Null bitmap size in bytes.
  uintptr_t offsets_addr   = 0; ///< String offsets GPU pointer (0 if not string).
  size_t offsets_len       = 0; ///< Offsets buffer size in bytes.
  int null_count           = 0; ///< Number of null values.
  int scale                = 0; ///< Decimal scale.
};

/// Thread-safe singleton that manages:
/// 1. Shared send staging buffer (set once at startup, pre-registered with nixl)
/// 2. Active ExchangeSession (per-execution state, created/destroyed per fragment)
/// 3. Legacy GPU buffer metadata (for detect_execution_location)
///
/// The session lifecycle:
/// - begin_session() creates a new ExchangeSession, stores as active
/// - Result collector uses get_active_session() during GPU execution
/// - take_session() moves the session out (Rust takes ownership)
/// - end_session(id) releases C++ resources for a completed session
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
    // The offset only resets to 0 on ClearCompletedSessions (release_gpu_buffers).
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

  /// Destroy a session and release its resources.
  /// Called after exchange transfer completes.
  void EndSession(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    // If the session is still active (not taken), clear it.
    if (active_session_ && active_session_->session_id == session_id) {
      active_session_.reset();
    }
    // Completed sessions are also tracked for cleanup.
    completed_sessions_.erase(
      std::remove_if(completed_sessions_.begin(), completed_sessions_.end(),
                     [session_id](const auto& s) { return s->session_id == session_id; }),
      completed_sessions_.end());
    inflight_staging_offsets_.erase(session_id);
  }

  /// Store a taken session for later cleanup (when Rust calls end_session).
  void StoreCompletedSession(std::unique_ptr<ExchangeSession> session) {
    std::lock_guard<std::mutex> lock(mutex_);
    completed_sessions_.push_back(std::move(session));
  }

  void RegisterInflightArtifact(uint64_t session_id, size_t staging_offset) {
    std::lock_guard<std::mutex> lock(mutex_);
    inflight_staging_offsets_[session_id] = staging_offset;
  }

  void ReleaseInflightArtifact(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    inflight_staging_offsets_.erase(session_id);
  }

  // --- Legacy GPU buffer metadata (for detect_execution_location) ---

  void Store(std::vector<GPUBufferInfo> buffers) {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_ = std::move(buffers);
  }

  std::vector<GPUBufferInfo> Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffers_;
  }

  // --- Compatibility shims (transition period) ---
  // These delegate to the active session. Remove after full migration.

  bool ShouldRetain() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_session_ != nullptr;
  }

  void SetRetainNext(bool v) {
    // No-op during transition. Retention is now session-based.
    (void)v;
  }

  std::pair<int, std::vector<int>> GetPartitionConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      return {active_session_->partition_num, active_session_->partition_cols};
    }
    return {0, {}};
  }

  void SetPartitionConfig(int num_partitions, std::vector<int> column_indices) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      active_session_->partition_num = num_partitions;
      active_session_->partition_cols = std::move(column_indices);
      active_session_->packed_partitions.clear();
    }
  }

  size_t GetStagingOffset() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) return active_session_->staging_offset;
    return 0;
  }

  void SetStagingOffset(size_t offset) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) active_session_->staging_offset = offset;
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

  void StorePackedData(std::shared_ptr<void> gpu_data,
                       uintptr_t gpu_addr, size_t gpu_size,
                       std::unique_ptr<std::vector<uint8_t>> metadata) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      active_session_->store_packed(std::move(gpu_data), gpu_addr, gpu_size,
                                    std::move(metadata));
    }
  }

  void RetainData(std::shared_ptr<void> data) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      active_session_->retain(std::move(data));
    }
  }

  void ReleaseRetainedData() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      active_session_->retained_data.clear();
      active_session_->packed_gpu_data.reset();
      active_session_->packed_metadata.reset();
      active_session_->packed_gpu_addr = 0;
      active_session_->packed_gpu_size = 0;
    }
  }

  void ClearPartitionConfig() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      active_session_->partition_num = 0;
      active_session_->partition_cols.clear();
      active_session_->packed_partitions.clear();
    }
  }

  std::vector<PackedPartition> TakePackedPartitions() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      return std::move(active_session_->packed_partitions);
    }
    return {};
  }

  void AccumulatePackedBroadcast(PackedBroadcastEntry entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      active_session_->accumulate_broadcast(std::move(entry));
    }
  }

  std::vector<PackedBroadcastEntry> TakePackedBroadcastEntries() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      return std::move(active_session_->packed_broadcast_entries);
    }
    return {};
  }

  std::pair<uintptr_t, size_t> GetPackedGPU() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_) {
      return {active_session_->packed_gpu_addr, active_session_->packed_gpu_size};
    }
    return {0, 0};
  }

  const std::vector<uint8_t>* GetPackedMetadata() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_ && active_session_->packed_metadata) {
      return active_session_->packed_metadata.get();
    }
    return nullptr;
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_.clear();
    // NOTE: Don't clear active_session_ here. Releasing one fragment's
    // data must not destroy another fragment's session. Sessions are
    // managed via BeginSession/TakeSession/EndSession.
  }

  /// Clear all completed sessions (whose data has been sent).
  /// Called from release_gpu_buffers after exchange forward completes.
  void ClearCompletedSessions() {
    std::lock_guard<std::mutex> lock(mutex_);
    completed_sessions_.clear();
    buffers_.clear();
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

  // Legacy GPU buffer metadata
  std::vector<GPUBufferInfo> buffers_;
};

}  // namespace duckdb
