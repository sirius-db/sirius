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

#include "op/sirius_physical_operator.hpp"

#include <cudf/io/parquet.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// row_group_range
//===----------------------------------------------------------------------===//
/**
 * @brief Represents a range of consecutive row groups within a single parquet file.
 *
 * Used as the unit of work for both the metadata scan (partitioning) and the
 * GPU scan (byte-range preloading).
 */
struct row_group_range {
  row_group_range(size_t file_idx,
                  size_t start_row_group,
                  size_t row_group_count,
                  size_t reserved_uncompressed_bytes,
                  size_t reserved_compressed_bytes)
    : file_idx(file_idx),
      start_row_group(start_row_group),
      row_group_count(row_group_count),
      reserved_uncompressed_bytes(reserved_uncompressed_bytes),
      reserved_compressed_bytes(reserved_compressed_bytes)
  {
  }

  size_t file_idx;
  size_t start_row_group;
  size_t row_group_count;
  size_t reserved_uncompressed_bytes;
  size_t reserved_compressed_bytes;
};

//===----------------------------------------------------------------------===//
// parquet_metadata_input
//===----------------------------------------------------------------------===//
/**
 * @brief Input to a parquet metadata scan task.
 *
 * Carries a batch of file paths (up to max_file_processed) along with the
 * target approximate batch size used when partitioning row groups.
 *
 * prepare_for_processing returns an empty handle vector because this type
 * holds no cucascade data_batch objects — it is pure metadata.
 */
class parquet_metadata_input : public op::operator_data {
 public:
  parquet_metadata_input(std::vector<std::string> file_paths, size_t approximate_batch_size)
    : file_paths(std::move(file_paths)), approximate_batch_size(approximate_batch_size)
  {
  }

  std::optional<std::vector<::cucascade::data_batch_processing_handle>> prepare_for_processing(
    const ::cucascade::memory::memory_space* /*requested_memory_space*/,
    rmm::cuda_stream_view /*stream*/) override
  {
    return std::vector<::cucascade::data_batch_processing_handle>{};
  }

  std::vector<std::string> file_paths;
  size_t approximate_batch_size;
};

//===----------------------------------------------------------------------===//
// partitioned_parquet_metadata
//===----------------------------------------------------------------------===//
/**
 * @brief Output of a parquet metadata scan task.
 *
 * Contains the parsed parquet file metadata and the row-group partitions
 * computed from it, ready for consumption by sirius_gpu_parquet_scan_operator.
 *
 * prepare_for_processing returns an empty handle vector because this type
 * holds no cucascade data_batch objects — it is pure metadata.
 */
class partitioned_parquet_metadata : public op::operator_data {
 public:
  partitioned_parquet_metadata() = default;

  std::optional<std::vector<::cucascade::data_batch_processing_handle>> prepare_for_processing(
    const ::cucascade::memory::memory_space* /*requested_memory_space*/,
    rmm::cuda_stream_view /*stream*/) override
  {
    return std::vector<::cucascade::data_batch_processing_handle>{};
  }

  /**
   * @brief Compute total uncompressed bytes across all row-group partitions.
   */
  [[nodiscard]] size_t compute_total_uncompressed_bytes() const
  {
    size_t total = 0;
    for (auto const& rg : row_group_partitions) {
      total += rg.reserved_uncompressed_bytes;
    }
    return total;
  }

  std::vector<std::string> file_paths;
  std::vector<cudf::io::parquet::FileMetaData> file_metadatas;
  cudf::io::parquet_reader_options reader_options;
  std::vector<row_group_range> row_group_partitions;
  std::vector<size_t> file_sizes;
  std::vector<size_t> footer_offsets;
  std::vector<size_t> metadata_byte_sizes;
  std::vector<size_t> selected_column_indices;
};

//===----------------------------------------------------------------------===//
// parquet_scan_data
//===----------------------------------------------------------------------===//
/**
 * @brief Input to a GPU parquet scan task.
 *
 * References the shared partitioned_parquet_metadata and carries the subset of
 * row_group_range objects that this particular task must read.  The scan ranges
 * are batched by sirius_gpu_parquet_scan_operator::get_next_task_input_data()
 * to meet the configured batch_data_size.
 *
 * prepare_for_processing returns an empty handle vector because this type
 * holds no cucascade data_batch objects.
 */
class parquet_scan_data : public op::operator_data {
 public:
  parquet_scan_data(partitioned_parquet_metadata meta, std::vector<row_group_range> scan_ranges)
    : metadata(std::move(meta)), scan_ranges(std::move(scan_ranges))
  {
  }

  std::optional<std::vector<::cucascade::data_batch_processing_handle>> prepare_for_processing(
    const ::cucascade::memory::memory_space* /*requested_memory_space*/,
    rmm::cuda_stream_view /*stream*/) override
  {
    return std::vector<::cucascade::data_batch_processing_handle>{};
  }

  partitioned_parquet_metadata metadata;
  std::vector<row_group_range> scan_ranges;
};

}  // namespace sirius::op::scan
