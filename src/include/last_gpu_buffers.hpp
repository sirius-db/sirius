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
#include <string>
#include <vector>

namespace duckdb {

/// Metadata for a single GPU column buffer from the last execution.
struct GPUBufferInfo {
  uintptr_t addr;          ///< GPU device pointer (data buffer).
  size_t len;              ///< Data buffer size in bytes.
  int device_id;           ///< GPU device ID.
  std::string column_name; ///< Column name from the query plan.
  int type_id;             ///< cudf::type_id as int (not GPUColumnTypeId).
  size_t num_rows;         ///< Number of rows in the column.

  // Extended buffer info for complete Arrow reconstruction:
  uintptr_t null_mask_addr = 0; ///< Null bitmap GPU pointer (0 if all-valid).
  size_t null_mask_len     = 0; ///< Null bitmap size in bytes.
  uintptr_t offsets_addr   = 0; ///< String offsets GPU pointer (0 if not string).
  size_t offsets_len       = 0; ///< Offsets buffer size in bytes.
  int null_count           = 0; ///< Number of null values.
  int scale                = 0; ///< Decimal scale (from cudf data_type::scale()).
};

/// Thread-safe singleton that stores GPU buffer metadata from the most recent
/// GPU execution. Populated by ConvertGPUTableToCPUCollection, read by the
/// sirius_get_last_gpu_buffers() table function.
class LastGPUBuffers {
 public:
  static LastGPUBuffers& GetInstance() {
    static LastGPUBuffers instance;
    return instance;
  }

  void Store(std::vector<GPUBufferInfo> buffers) {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_ = std::move(buffers);
  }

  std::vector<GPUBufferInfo> Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffers_;
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_.clear();
    retained_gpu_data_.clear();
  }

  /// Keep a type-erased reference to GPU data alive (prevents deallocation).
  /// Used to retain cucascade data_batch objects for nixl transfer.
  void RetainData(std::shared_ptr<void> data) {
    std::lock_guard<std::mutex> lock(mutex_);
    retained_gpu_data_.push_back(std::move(data));
  }

  /// Release all retained GPU data references.
  void ReleaseRetainedData() {
    std::lock_guard<std::mutex> lock(mutex_);
    retained_gpu_data_.clear();
    packed_gpu_data_.reset();
    packed_metadata_.reset();
    packed_gpu_addr_ = 0;
    packed_gpu_size_ = 0;
  }

  /// Store packed GPU data (from cudf::pack). The device_buffer is wrapped
  /// in shared_ptr<void> to avoid exposing rmm headers.
  void StorePackedData(std::shared_ptr<void> gpu_data,
                       uintptr_t gpu_addr, size_t gpu_size,
                       std::unique_ptr<std::vector<uint8_t>> metadata) {
    std::lock_guard<std::mutex> lock(mutex_);
    packed_gpu_data_ = std::move(gpu_data);
    packed_gpu_addr_ = gpu_addr;
    packed_gpu_size_ = gpu_size;
    packed_metadata_ = std::move(metadata);
  }

  /// Get the packed GPU buffer address and size (0 if not packed).
  std::pair<uintptr_t, size_t> GetPackedGPU() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {packed_gpu_addr_, packed_gpu_size_};
  }

  /// Get the packed metadata bytes (nullptr if not packed).
  const std::vector<uint8_t>* GetPackedMetadata() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return packed_metadata_.get();
  }

  /// Set the pre-registered nixl staging buffer address and size.
  /// Called once at startup after C++ cudaMalloc + nixl registration.
  void SetStagingBuffer(uintptr_t addr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    staging_addr_ = addr;
    staging_size_ = size;
  }

  /// Get the staging buffer for chunked_pack to write into.
  std::pair<uintptr_t, size_t> GetStagingBuffer() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {staging_addr_, staging_size_};
  }

  void SetRetainNext(bool v) {
    std::lock_guard<std::mutex> lock(mutex_);
    retain_next_ = v;
  }

  bool ShouldRetain() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return retain_next_;
  }

  /// Set hash partition config for the next GPU execution.
  /// The result collector will partition the GPU result before packing.
  void SetPartitionConfig(int num_partitions, std::vector<int> column_indices) {
    std::lock_guard<std::mutex> lock(mutex_);
    partition_num_ = num_partitions;
    partition_cols_ = std::move(column_indices);
    packed_partitions_.clear();
    staging_offset_ = 0;
  }

  /// Get partition config (0 = no partitioning).
  std::pair<int, std::vector<int>> GetPartitionConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {partition_num_, partition_cols_};
  }

  void ClearPartitionConfig() {
    std::lock_guard<std::mutex> lock(mutex_);
    partition_num_ = 0;
    partition_cols_.clear();
    packed_partitions_.clear();
  }

  /// Per-partition packed result from GPU hash_partition + chunked_pack.
  struct PackedPartition {
    size_t staging_offset;
    size_t packed_size;
    std::unique_ptr<std::vector<uint8_t>> metadata;
    int32_t num_rows;
  };

  void StorePackedPartitions(std::vector<PackedPartition> partitions) {
    std::lock_guard<std::mutex> lock(mutex_);
    packed_partitions_ = std::move(partitions);
  }

  /// Accumulate per-partition packed data from a new batch into existing
  /// per-destination entries. On the first call, stores partitions as-is.
  /// On subsequent calls, merges: for each destination, accumulate rows and
  /// update metadata to the latest batch (each batch's data is at a new
  /// staging offset; the exchange sender handles multi-region transfers).
  void AccumulatePackedPartitions(std::vector<PackedPartition> partitions) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (packed_partitions_.empty()) {
      packed_partitions_ = std::move(partitions);
    } else {
      // For each destination, keep the FIRST entry's offset but accumulate
      // additional batch entries as new entries in a flat list.
      // The Rust exchange code handles multiple entries per destination.
      for (auto& p : partitions) {
        packed_partitions_.push_back(std::move(p));
      }
    }
  }

  /// Track cumulative staging offset for multi-batch packing.
  size_t GetStagingOffset() const { std::lock_guard<std::mutex> lock(mutex_); return staging_offset_; }
  void SetStagingOffset(size_t offset) { std::lock_guard<std::mutex> lock(mutex_); staging_offset_ = offset; }

  std::vector<PackedPartition> TakePackedPartitions() {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::move(packed_partitions_);
  }

 private:
  LastGPUBuffers() = default;
  mutable std::mutex mutex_;
  std::vector<GPUBufferInfo> buffers_;
  std::vector<std::shared_ptr<void>> retained_gpu_data_;
  /// Packed GPU data (contiguous buffer from cudf::pack).
  std::shared_ptr<void> packed_gpu_data_;
  uintptr_t packed_gpu_addr_ = 0;
  size_t packed_gpu_size_ = 0;
  std::unique_ptr<std::vector<uint8_t>> packed_metadata_;
  /// Pre-registered nixl staging buffer (set once at startup).
  uintptr_t staging_addr_ = 0;
  size_t staging_size_ = 0;
  bool retain_next_ = false;
  /// Hash partition config for exchange.
  int partition_num_ = 0;
  std::vector<int> partition_cols_;
  std::vector<PackedPartition> packed_partitions_;
  size_t staging_offset_ = 0;
};

}  // namespace duckdb
