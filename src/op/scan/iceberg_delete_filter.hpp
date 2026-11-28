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

#include "op/scan/iceberg_delete_set.hpp"
#include "op/scan/parquet_batch_layout.hpp"

#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
/// The row provenance of one decoded batch, in batch row order.
using batch_layout = std::span<batch_row_run const>;

/// One delete filter stage: V2 positional deletes (which V3 deletion vectors merge into) or V2
/// equality deletes. Applied per decoded batch by @ref iceberg_gpu_ingestible.
class iceberg_delete_filter {
 public:
  virtual ~iceberg_delete_filter() = default;

  /// Called on a GPU worker thread on the task-local stream. @p layout is used by position-keyed
  /// filters and ignored by equality deletes. Returns @p tbl unchanged if nothing was deleted.
  virtual std::unique_ptr<cudf::table> apply(std::unique_ptr<cudf::table> tbl,
                                             batch_layout layout,
                                             ::cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr) = 0;
};

struct IcebergDeleteData;  // iceberg_metadata_reader.hpp

/// Applies V2 positional deletes: binary-searches the batch's range in a per-file map of sorted
/// row positions, then applies a boolean mask.
class positional_delete_filter : public iceberg_delete_filter {
 public:
  explicit positional_delete_filter(std::shared_ptr<const IcebergDeleteData> delete_data);
  explicit positional_delete_filter(iceberg_delete_sets delete_sets);

  std::unique_ptr<cudf::table> apply(std::unique_ptr<cudf::table> tbl,
                                     batch_layout layout,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) override;

  /// Whether any run in @p layout names a file this filter holds deletes for. Lets the
  /// caller skip the mask build entirely for batches drawn only from undeleted files.
  [[nodiscard]] bool affects(batch_layout layout) const;

 private:
  [[nodiscard]] block_span<int64_t const> positions_for(std::string const& path) const;
  iceberg_delete_sets _delete_sets;
};

/// Applies one equality-delete group: probes its prebuilt hash join with the batch's key columns
/// and applies an anti-join mask. The caller decides which groups apply; the layout is ignored.
class equality_delete_filter : public iceberg_delete_filter {
 public:
  /// @param device The device the batch is on. The group's tables built there are probed.
  /// @param data_key_indices Batch column indices of the group's key columns, in the group's
  ///                         key order.
  equality_delete_filter(std::shared_ptr<const IcebergDeleteData> delete_data,
                         rmm::cuda_device_id device,
                         size_t group_index,
                         std::vector<cudf::size_type> data_key_indices);

  std::unique_ptr<cudf::table> apply(std::unique_ptr<cudf::table> tbl,
                                     batch_layout layout,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) override;

 private:
  std::shared_ptr<const IcebergDeleteData> _delete_data;
  rmm::cuda_device_id _device;
  size_t _group_index;
  std::vector<cudf::size_type> _data_key_indices;
};

}  // namespace sirius::op::scan
