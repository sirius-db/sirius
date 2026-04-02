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
#include <memory>
#include <mutex>
#include <vector>

namespace duckdb {

/// Per-partition packed GPU buffer from cudf::hash_partition + chunked_pack.
struct PackedPartition {
  size_t staging_offset;
  size_t packed_size;
  std::unique_ptr<std::vector<uint8_t>> metadata;
  int32_t num_rows;

  /// When non-zero, this partition overflowed the staging buffer and was packed
  /// into a separate rmm::device_buffer at this address. The staging_offset
  /// field is unused in this case.
  uintptr_t overflow_gpu_addr = 0;
  /// Owns the rmm::device_buffer backing overflow_gpu_addr (RAII).
  std::shared_ptr<void> overflow_data;
};

/// Packed GPU buffer for a single batch in the broadcast path.
/// Mirrors PackedPartition but is used for the broadcast accumulation path
/// where each input batch is independently packed into the staging buffer.
struct PackedBroadcastEntry {
  size_t staging_offset;
  size_t packed_size;
  std::unique_ptr<std::vector<uint8_t>> metadata;
  int32_t num_rows;

  /// Overflow fields (same semantics as PackedPartition).
  uintptr_t overflow_gpu_addr = 0;
  std::shared_ptr<void> overflow_data;
};

/// Per-execution exchange session holding all GPU buffer state for a single
/// fragment's GPU execution + exchange forward.
///
/// Created by begin_exchange_session(), used by the result_collector during
/// GPU execution, moved out by take_session_results(), destroyed by
/// end_exchange_session().
///
/// This replaces the per-execution state that was in the LastGPUBuffers
/// global singleton. By moving the session OUT of the singleton before
/// releasing the engine lock, concurrent fragment executions each own
/// their own session with no interference.
class ExchangeSession {
 public:
  uint64_t session_id;

  // --- Partition config (set before execution) ---
  int partition_num = 0;
  std::vector<int> partition_cols;

  // --- Packed GPU data (set during execution by result_collector) ---
  // Single packed buffer (broadcast path)
  std::shared_ptr<void> packed_gpu_data;
  uintptr_t packed_gpu_addr = 0;
  size_t packed_gpu_size = 0;
  std::unique_ptr<std::vector<uint8_t>> packed_metadata;

  // Per-partition packed data (accumulated across batches)
  std::vector<PackedPartition> packed_partitions;
  // Per-batch broadcast packed data (accumulated across batches)
  std::vector<PackedBroadcastEntry> packed_broadcast_entries;
  size_t staging_offset = 0;

  // --- Retained data (keeps GPU memory alive until transfer done) ---
  std::vector<std::shared_ptr<void>> retained_data;

  // --- Staging lease ---
  // Base address in the shared send staging buffer where this session packs.
  // Each session gets a contiguous region starting at staging_lease_base.
  uintptr_t staging_lease_base = 0;
  size_t staging_lease_size = 0;

  // --- Methods ---

  /// Store packed data from single-pack or hash_partition path.
  void store_packed(std::shared_ptr<void> data, uintptr_t addr,
                    size_t size, std::unique_ptr<std::vector<uint8_t>> md) {
    packed_gpu_data = std::move(data);
    packed_gpu_addr = addr;
    packed_gpu_size = size;
    packed_metadata = std::move(md);
  }

  /// Accumulate per-partition packed data from a new batch.
  void accumulate_partitions(std::vector<PackedPartition> parts) {
    for (auto& p : parts) {
      packed_partitions.push_back(std::move(p));
    }
  }

  /// Accumulate a single broadcast packed entry from one input batch.
  void accumulate_broadcast(PackedBroadcastEntry entry) {
    packed_broadcast_entries.push_back(std::move(entry));
  }

  /// Keep a type-erased reference to GPU data alive.
  void retain(std::shared_ptr<void> data) {
    retained_data.push_back(std::move(data));
  }
};

}  // namespace duckdb
