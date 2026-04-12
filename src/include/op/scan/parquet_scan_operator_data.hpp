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

// sirius
#include <expression_executor/gpu_expression_translator.hpp>
#include <op/sirius_physical_operator.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>

// standard library
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace sirius::op::scan {

using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

//===----------------------------------------------------------------------===//
// row_group_range
//===----------------------------------------------------------------------===//
/**
 * @brief Represents a set of row groups within a single parquet file.
 *
 * Used as the unit of work for both the metadata scan (partitioning) and the
 * GPU scan (byte-range preloading).
 */
struct row_group_range {
  row_group_range(std::size_t file_idx,
                  std::vector<cudf::size_type> row_group_indices,
                  std::size_t reserved_uncompressed_bytes,
                  std::size_t reserved_compressed_bytes)
    : file_idx(file_idx),
      row_group_indices(std::move(row_group_indices)),
      reserved_uncompressed_bytes(reserved_uncompressed_bytes),
      reserved_compressed_bytes(reserved_compressed_bytes)
  {
  }

  std::size_t file_idx;
  std::vector<cudf::size_type> row_group_indices;
  std::size_t reserved_uncompressed_bytes;
  std::size_t reserved_compressed_bytes;
};

//===----------------------------------------------------------------------===//
// parquet_metadata_input
//===----------------------------------------------------------------------===//
/**
 * @brief Input to a parquet metadata scan task.
 *
 * Carries a batch of file paths (up to max_file_processed) along with the
 * target approximate batch size used when partitioning row groups.
 */
class parquet_metadata_input : public op::operator_data {
 public:
  parquet_metadata_input(std::vector<std::string> file_paths, std::size_t approximate_batch_size)
    : file_paths(std::move(file_paths)), approximate_batch_size(approximate_batch_size)
  {
  }

  std::vector<std::string> file_paths;
  std::size_t approximate_batch_size;
};

//===----------------------------------------------------------------------===//
// partitioned_parquet_metadata
//===----------------------------------------------------------------------===//
/**
 * @brief Output of a parquet metadata scan task.
 *
 * Contains the parsed parquet file metadata and the row-group partitions
 * computed from it, ready for consumption by sirius_gpu_parquet_scan_operator.
 */
class partitioned_parquet_metadata : public op::operator_data {
 public:
  using translated_expression = gpu_expression_translator::translated_expression;

  partitioned_parquet_metadata() = default;

  std::vector<std::string> file_paths;
  std::vector<std::shared_ptr<cudf::io::datasource>> datasources;  ///< Parallel to file_paths.
  std::vector<row_group_range> row_group_partitions;

  std::shared_ptr<cudf::io::parquet_reader_options> reader_options;
  /// Either a) the translated filter expression for row-group pruning and filter pushdown, or
  ///        b) the coalesced duckdb expression if filter translation failed.
  /// Shared ownership of the translated filter expression is used so that
  /// the cuDF AST nodes referenced by reader_options remain alive.
  std::variant<std::shared_ptr<translated_expression>, std::shared_ptr<duckdb::Expression>>
    filter_expression;
  std::vector<std::size_t> post_filter_projection_ids;
};

//===----------------------------------------------------------------------===//
// parquet_scan_data
//===----------------------------------------------------------------------===//
/**
 * @brief Input to a GPU parquet scan task.
 *
 * Contains all per-partition data needed to read a single row_group_range from
 * a parquet file.  Fields are extracted from partitioned_parquet_metadata by
 * get_next_task_input_data() so that each task is self-contained.
 */
class parquet_scan_data : public op::operator_data {
 public:
  using translated_expression = gpu_expression_translator::translated_expression;
  parquet_scan_data(std::string file_path,
                    row_group_range rg_range,
                    std::shared_ptr<cudf::io::parquet_reader_options> reader_options,
                    std::variant<std::shared_ptr<translated_expression>,
                                 std::shared_ptr<duckdb::Expression>> filter_expression,
                    std::vector<std::size_t> post_filter_projection_ids,
                    std::shared_ptr<cudf::io::datasource> datasource)
    : file_path(std::move(file_path)),
      rg_range(std::move(rg_range)),
      reader_options(std::move(reader_options)),
      filter_expression(std::move(filter_expression)),
      post_filter_projection_ids(std::move(post_filter_projection_ids)),
      datasource(std::move(datasource))
  {
  }

  std::string file_path;
  row_group_range rg_range;
  std::shared_ptr<cudf::io::parquet_reader_options> reader_options;
  /// Either a) the translated filter expression for row-group pruning and filter pushdown, or
  ///        b) the coalesced duckdb expression if filter translation failed.
  /// Shared ownership of the translated filter expression is used so that
  /// the cuDF AST nodes referenced by reader_options remain alive.
  std::variant<std::shared_ptr<translated_expression>, std::shared_ptr<duckdb::Expression>>
    filter_expression;
  std::vector<std::size_t> post_filter_projection_ids;
  /// Datasource for the parquet file, shared with other partitions of the same file.
  std::shared_ptr<cudf::io::datasource> datasource;
};

}  // namespace sirius::op::scan
