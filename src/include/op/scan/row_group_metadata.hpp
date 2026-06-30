/*
 * Copyright 2026, Sirius Contributors.
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
#include <io/sirius_datasource.hpp>
// cudf

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet_schema.hpp>

// standard library
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace sirius::op::scan {

using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

//===----------------------------------------------------------------------===//
// row_group_slice
//===----------------------------------------------------------------------===//
/**
 * @brief Represents a set of row groups within a single parquet file.
 *
 * Multiple slices can be bundled together to form a single parquet partition corresponding
 * to a data batch. Used by @c parquet_gpu_ingestible (new path) and @c parquet_scan_task
 * (legacy gpu_processing path) — kept in a neutral header so both consumers share without
 * cross-depending on each other's headers.
 */
struct row_group_slice {
  row_group_slice(std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata,
                  std::string file_path,
                  std::vector<cudf::size_type> row_group_indices,
                  std::size_t reserved_uncompressed_bytes,
                  std::size_t reserved_compressed_bytes,
                  std::shared_ptr<io::sirius_datasource> datasource)
    : file_metadata(file_metadata),
      file_path(file_path),
      row_group_indices(std::move(row_group_indices)),
      reserved_uncompressed_bytes(reserved_uncompressed_bytes),
      reserved_compressed_bytes(reserved_compressed_bytes),
      datasource(std::move(datasource))
  {
  }
  std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
  std::string file_path;
  std::vector<cudf::size_type> row_group_indices;
  std::size_t reserved_uncompressed_bytes;
  std::size_t reserved_compressed_bytes;
  /// Pre-built datasource for this file. Created once by the split provider
  /// and reused by materialize_table. When null, materialize_table falls
  /// back to cudf::io::datasource::create(file_path).
  std::shared_ptr<io::sirius_datasource> datasource;
};

//===----------------------------------------------------------------------===//
// row_group_range
//===----------------------------------------------------------------------===//
/**
 * @brief Represents a set of row groups within a single parquet file.
 *
 * Used as the unit of work for the legacy parquet_scan_task path (byte-range preloading).
 *
 * @todo This needs to be deleted once Iceberg is integrated into scan manager framework.
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

}  // namespace sirius::op::scan
