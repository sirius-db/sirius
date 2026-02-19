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
#include <mutex>
#include <string>
#include <vector>

namespace duckdb {

/// Metadata for a single GPU column buffer from the last execution.
struct GPUBufferInfo {
  uintptr_t addr;          ///< GPU device pointer.
  size_t len;              ///< Buffer size in bytes.
  int device_id;           ///< GPU device ID.
  std::string column_name; ///< Column name from the query plan.
  int type_id;             ///< GPUColumnTypeId as int.
  size_t num_rows;         ///< Number of rows in the column.
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
  }

 private:
  LastGPUBuffers() = default;
  mutable std::mutex mutex_;
  std::vector<GPUBufferInfo> buffers_;
};

}  // namespace duckdb
