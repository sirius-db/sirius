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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sirius::op::scan {

// Forward-declare the post_convert_fn_t so we don't pull in the full
// host_parquet_representation header in every translation unit.
// The actual definition lives in data/host_parquet_representation.hpp.
using post_convert_fn_t =
  std::function<std::unique_ptr<cudf::table>(std::unique_ptr<cudf::table>,
                                             std::string const& data_file_path,
                                             int64_t first_row_offset,
                                             rmm::cuda_stream_view)>;

//===----------------------------------------------------------------------===//
// Abstract delete filter interface
//===----------------------------------------------------------------------===//

/**
 * @brief Abstract interface for a single Iceberg delete filter stage.
 *
 * Concrete implementations handle V2 positional deletes, V2 equality deletes,
 * and (future) V3 deletion vectors.  Each filter is applied to one GPU batch
 * at a time by the post-convert hook.
 */
class iceberg_delete_filter {
 public:
  virtual ~iceberg_delete_filter() = default;

  /**
   * @brief Apply this filter to a data batch.
   *
   * Called on a GPU worker thread during parquet decompression.
   *
   * @param tbl            The GPU table to filter.
   * @param data_file_path Path of the data file this batch came from.
   * @param first_row      Absolute row offset of the first row in this batch.
   * @param stream         CUDA stream for GPU operations.
   * @return Filtered table (may be the same pointer if nothing was deleted).
   */
  virtual std::unique_ptr<cudf::table> apply(std::unique_ptr<cudf::table> tbl,
                                             std::string const& data_file_path,
                                             int64_t first_row,
                                             rmm::cuda_stream_view stream) = 0;
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
                                     std::string const& data_file_path,
                                     int64_t first_row,
                                     rmm::cuda_stream_view stream) override;

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
                                     std::string const& data_file_path,
                                     int64_t first_row,
                                     rmm::cuda_stream_view stream) override;

 private:
  std::shared_ptr<const IcebergDeleteData> _delete_data;
  size_t _group_index;
  std::vector<cudf::size_type> _data_key_indices;
};

//===----------------------------------------------------------------------===//
// Delete pipeline — composes filters + strips extra columns
//===----------------------------------------------------------------------===//

/**
 * @brief Owns an ordered list of iceberg_delete_filters and produces the
 * post_convert_fn_t that applies them all, then strips any force-projected
 * extra columns from the result.
 *
 * The "extra columns" mechanism exists because equality-delete key columns
 * may not be in the user's query projection.  The scan is widened to include
 * them (appended at the end), and after all filters run, the pipeline strips
 * those trailing columns so downstream operators see only what was requested.
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
   * @brief Build a single post_convert_fn_t that:
   *   1. Runs each filter in insertion order.
   *   2. Strips any trailing extra columns from the result.
   */
  [[nodiscard]] post_convert_fn_t build_hook() const;

 private:
  std::vector<std::shared_ptr<iceberg_delete_filter>> _filters;
  size_t _extra_column_count = 0;
};

}  // namespace sirius::op::scan
