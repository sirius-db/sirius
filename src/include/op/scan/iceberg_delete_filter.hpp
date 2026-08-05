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

#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
/**
 * @brief One contiguous run of decoded rows, mapped back to its source file rows.
 *
 * Positional deletes are keyed on @c (data_file_path, row_position_within_that_file). A batch is
 * the concatenation of the selected row groups of the split's files, so the mapping is a LIST of
 * runs, not one @c (path, first_row) pair: a split can span files, and pruning leaves gaps.
 *
 * Only valid while nothing drops rows between decode and here — which is why the iceberg path
 * disables reader-side pushdown. A row removed in the reader shifts every later position.
 */
struct batch_row_run {
  /// Data file these rows came from; the key into IcebergDeleteData::positional_deletes.
  std::string data_file_path;
  /// Row index within that data file of the first row of this run.
  int64_t file_row_offset{0};
  /// Row index within the decoded batch of the first row of this run.
  int64_t batch_row_offset{0};
  /// Number of rows in the run.
  int64_t num_rows{0};
};

/// The row provenance of one decoded batch, in batch row order.
using batch_layout = std::span<batch_row_run const>;

/// One delete filter stage: V2 positional deletes (which V3 deletion vectors merge into) or V2
/// equality deletes. Applied per decoded batch by @ref iceberg_delete_pipeline.
class iceberg_delete_filter {
 public:
  virtual ~iceberg_delete_filter() = default;

  /// Called on a GPU worker thread on the task-local stream. @p layout is used by position-keyed
  /// filters and ignored by equality deletes. Returns @p tbl unchanged if nothing was deleted.
  virtual std::unique_ptr<cudf::table> apply(std::unique_ptr<cudf::table> tbl,
                                             batch_layout layout,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr) = 0;
};

struct IcebergDeleteData;  // iceberg_metadata_reader.hpp

/// Applies V2 positional deletes: binary-searches the batch's range in a per-file map of sorted
/// row positions, then applies a boolean mask.
class positional_delete_filter : public iceberg_delete_filter {
 public:
  explicit positional_delete_filter(std::shared_ptr<const IcebergDeleteData> delete_data);

  std::unique_ptr<cudf::table> apply(std::unique_ptr<cudf::table> tbl,
                                     batch_layout layout,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) override;

  /// Whether any run in @p layout names a file this filter holds deletes for. Lets the
  /// caller skip the mask build entirely for batches drawn only from undeleted files.
  [[nodiscard]] bool affects(batch_layout layout) const;

 private:
  std::shared_ptr<const IcebergDeleteData> _delete_data;
};

/// Applies V2 equality deletes: probes the group's prebuilt hash join with the batch's key
/// columns and applies an anti-join mask. Entirely on device.
class equality_delete_filter : public iceberg_delete_filter {
 public:
  /// @param data_key_indices Batch column indices of the group's equality key columns.
  equality_delete_filter(std::shared_ptr<const IcebergDeleteData> delete_data,
                         size_t group_index,
                         std::vector<cudf::size_type> data_key_indices);

  std::unique_ptr<cudf::table> apply(std::unique_ptr<cudf::table> tbl,
                                     batch_layout layout,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) override;

 private:
  std::shared_ptr<const IcebergDeleteData> _delete_data;
  size_t _group_index;
  std::vector<cudf::size_type> _data_key_indices;
};

/**
 * @brief Applies an ordered list of delete filters to a batch, then strips extra columns.
 *
 * Equality-delete key columns may not be in the query's projection, so the scan is widened to
 * append them; once every filter has run they are stripped again. Order matters — they are
 * appended at the tail, and stripping cuts from the tail.
 */
class iceberg_delete_pipeline {
 public:
  /// Applied in insertion order.
  void add_filter(std::shared_ptr<iceberg_delete_filter> filter);

  /// Extra columns appended for delete-key projection, stripped after all filters run.
  void set_extra_column_count(size_t n) { _extra_column_count = n; }

  [[nodiscard]] size_t extra_column_count() const { return _extra_column_count; }

  /// True if no filters have been added.
  [[nodiscard]] bool empty() const { return _filters.empty(); }

  /**
   * @brief Run every filter in insertion order, then strip any trailing extra columns.
   *
   * @param tbl     Decoded batch, consumed.
   * @param layout  Row provenance of @p tbl (see @ref batch_row_run).
   * @param stream  CUDA stream for GPU operations.
   * @param mr      Allocator for filter results.
   */
  [[nodiscard]] std::unique_ptr<cudf::table> apply(std::unique_ptr<cudf::table> tbl,
                                                   batch_layout layout,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr) const;

 private:
  std::vector<std::shared_ptr<iceberg_delete_filter>> _filters;
  size_t _extra_column_count = 0;
};

}  // namespace sirius::op::scan
