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
// batch_row_run — where a batch's rows came from
//===----------------------------------------------------------------------===//
/**
 * @brief One contiguous run of decoded rows, mapped back to its source file rows.
 *
 * Positional deletes and deletion vectors are keyed on
 * @c (data_file_path, row_position_within_that_file), so applying them requires knowing which
 * file row each decoded row is. A decoded batch is the concatenation of the selected row
 * groups of the split's files, in source order, so that mapping is a list of runs rather than
 * the single @c (path, first_row) pair the pre-removal hook assumed: one split can span
 * several files, and row-group pruning leaves gaps between the row groups of each file.
 *
 * A run is only valid when nothing dropped rows between decode and here — which is why the
 * iceberg path disables reader-side filter pushdown. Rows removed inside the reader would
 * shift every subsequent row's position with no way to recover the mapping.
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

//===----------------------------------------------------------------------===//
// Abstract delete filter interface
//===----------------------------------------------------------------------===//

/**
 * @brief Abstract interface for a single Iceberg delete filter stage.
 *
 * Concrete implementations handle V2 positional deletes (which V3 deletion vectors merge
 * into) and V2 equality deletes. Each filter is applied to one decoded batch at a time by
 * @ref iceberg_delete_pipeline, from the scan's materialize step.
 */
class iceberg_delete_filter {
 public:
  virtual ~iceberg_delete_filter() = default;

  /**
   * @brief Apply this filter to a decoded batch.
   *
   * Called on a GPU worker thread on the task-local stream.
   *
   * @param tbl     The GPU table to filter.
   * @param layout  Row provenance of @p tbl. Position-keyed filters use it; equality deletes
   *                match on key values and ignore it.
   * @param stream  CUDA stream for GPU operations.
   * @param mr      Allocator for any result the filter allocates. Passing the scan's memory
   *                space keeps the filtered table inside the engine's memory accounting.
   * @return Filtered table (may be the same pointer if nothing was deleted).
   */
  virtual std::unique_ptr<cudf::table> apply(std::unique_ptr<cudf::table> tbl,
                                             batch_layout layout,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr) = 0;
};

// Forward declaration — full definition in iceberg_metadata_reader.hpp.
struct IcebergDeleteData;

//===----------------------------------------------------------------------===//
// Positional delete filter
//===----------------------------------------------------------------------===//

/**
 * @brief Applies Iceberg V2 positional deletes to each data batch.
 *
 * Holds a per-data-file map of sorted row positions to delete.  For each
 * batch the hook does a binary search to find positions in range, builds
 * a boolean mask, and applies cudf::apply_boolean_mask.
 */
class positional_delete_filter : public iceberg_delete_filter {
 public:
  /**
   * @param delete_data  Shared ownership of the materialized delete data
   *                     (keeps positional delete map alive for query lifetime).
   */
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

//===----------------------------------------------------------------------===//
// Equality delete filter
//===----------------------------------------------------------------------===//

/**
 * @brief Applies Iceberg V2 equality deletes to each data batch.
 *
 * Holds a shared reference to the pre-materialized IcebergDeleteData
 * (which owns the GPU hash join and delete key table).  For each batch,
 * probes the hash join with the data chunk's key columns, builds a
 * boolean anti-join mask entirely on GPU via thrust::transform, and
 * applies cudf::apply_boolean_mask.
 *
 * No GPU-to-host data transfer is required.
 */
class equality_delete_filter : public iceberg_delete_filter {
 public:
  /**
   * @param delete_data      Shared ownership of the materialized delete data
   *                         (keeps GPU table + hash join alive for query lifetime).
   * @param data_key_indices Indices into the data-chunk columns that correspond
   *                         to the equality key columns.
   */
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

//===----------------------------------------------------------------------===//
// Delete pipeline — composes filters + strips extra columns
//===----------------------------------------------------------------------===//

/**
 * @brief Owns an ordered list of iceberg_delete_filters and applies them all to a batch,
 * then strips any force-projected extra columns from the result.
 *
 * The "extra columns" mechanism exists because equality-delete key columns
 * may not be in the user's query projection.  The scan is widened to include
 * them (appended at the end), and after all filters run, the pipeline strips
 * those trailing columns so downstream operators see only what was requested.
 *
 * Before iceberg was removed from the build this was a @c post_convert_fn_t handed to the
 * host parquet representation. That type went away with it, and its two extra arguments —
 * a single data file path and a single first-row offset — could not describe a split
 * spanning several files or pruned row groups anyway. @ref apply takes a @ref batch_layout
 * instead, and is called from the scan's materialize step.
 */
class iceberg_delete_pipeline {
 public:
  /// Add a filter stage.  Filters are applied in insertion order.
  void add_filter(std::shared_ptr<iceberg_delete_filter> filter);

  /// Set the number of extra columns appended to the scan for delete-key
  /// projection.  These will be stripped after all filters run.
  void set_extra_column_count(size_t n) { _extra_column_count = n; }

  /// Return the number of extra columns that were force-projected.
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
