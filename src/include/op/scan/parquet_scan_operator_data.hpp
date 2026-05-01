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
#include <expression_executor/gpu_expression_translator_internal.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/sirius_physical_operator.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
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
 * Multiple slices can be bundled together to form a single parquet partition corresponding to a
 * data batch.
 */
struct row_group_slice {
  row_group_slice(std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata,
                  std::string file_path,
                  std::vector<cudf::size_type> row_group_indices,
                  std::size_t reserved_uncompressed_bytes,
                  std::size_t reserved_compressed_bytes)
    : file_metadata(file_metadata),
      file_path(file_path),
      row_group_indices(std::move(row_group_indices)),
      reserved_uncompressed_bytes(reserved_uncompressed_bytes),
      reserved_compressed_bytes(reserved_compressed_bytes)
  {
  }
  std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
  std::string file_path;
  std::vector<cudf::size_type> row_group_indices;
  std::size_t reserved_uncompressed_bytes;
  std::size_t reserved_compressed_bytes;
};

//===----------------------------------------------------------------------===//
// row_group_range
//===----------------------------------------------------------------------===//
/**
 * @brief Represents a set of row groups within a single parquet file.
 *
 * Used as the unit of work for both the parquet split provider (partitioning) and
 * the GPU scan (byte-range preloading).
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

//===----------------------------------------------------------------------===//
// parquet_scan_data
//===----------------------------------------------------------------------===//
/**
 * @brief Input to a GPU parquet scan task.
 *
 * Contains all per-partition data needed to read a single row_group_range from
 * a parquet file. Each instance is constructed by parquet_split_provider::run_batch
 * and pushed into the gpu scan operator's split_connector; the operator pulls one
 * via get_next_task_input_data() per task so each task is self-contained.
 */
class parquet_scan_data : public op::operator_data {
 public:
  using translated_expression = gpu_expression_translator::translated_expression;
  parquet_scan_data(std::vector<row_group_slice> rg_slices,
                    std::shared_ptr<cudf::io::parquet_reader_options> reader_options,
                    std::shared_ptr<duckdb::Expression> filter_expression,
                    std::shared_ptr<scan_plan const> plan,
                    std::vector<std::string> partition_values)
    : rg_slices(std::move(rg_slices)),
      reader_options(std::move(reader_options)),
      filter_expression(std::move(filter_expression)),
      plan(std::move(plan)),
      partition_values(std::move(partition_values))
  {
  }

  /**
   * @brief Capture the task's reserved memory space so that
   *        sirius_gpu_parquet_scan_operator::execute() can tag its output batches with it.
   *
   * parquet_scan_data is the input to a source task: it owns no upstream data batches
   * that need locking or conversion, so the handle vector is always empty. Its only
   * preparation responsibility is to record @p requested_memory_space into
   * @ref gpu_memory_space, where sirius_gpu_parquet_scan_operator::execute() will read
   * it when wrapping the freshly-read cudf::table in a data_batch via make_data_batch().
   *
   * The recorded pointer is valid for the lifetime of the task's memory reservation —
   * from when gpu_pipeline_executor sets the reservation on the task local state
   * (see gpu_pipeline_executor::manager_loop) through the end of execute().
   *
   * @param requested_memory_space  Memory space associated with the task reservation;
   *                                stored into gpu_memory_space for use during execute().
   * @param stream                  Unused — no data movement occurs during preparation
   *                                for this source input.
   * @return  Always an empty handle vector; there are no batches to keep locked.
   */
  std::optional<std::vector<::cucascade::data_batch_processing_handle>> prepare_for_processing(
    const ::cucascade::memory::memory_space* requested_memory_space,
    rmm::cuda_stream_view /*stream*/) override
  {
    gpu_memory_space = const_cast<cucascade::memory::memory_space*>(requested_memory_space);
    return std::vector<::cucascade::data_batch_processing_handle>{};
  };

  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    return std::accumulate(
      rg_slices.begin(), rg_slices.end(), std::size_t{0}, [](auto acc, auto const& s) {
        return acc + s.reserved_uncompressed_bytes;
      });
  }

  /// The row group slices (potentially across multiple files)
  std::vector<row_group_slice> rg_slices;
  /// The reader options to apply in the parquet reader
  std::shared_ptr<cudf::io::parquet_reader_options> reader_options;
  /// The coalesced duckdb filter expression.
  std::shared_ptr<duckdb::Expression> filter_expression;
  /// Scan plan shared across all splits of this scan. Carries the D-order column name
  /// table used by execute()'s per-task AST translation, plus the post-read assembly layout.
  std::shared_ptr<scan_plan const> plan;
  /// Hive partition values shared by every file in @ref rg_slices, in scan_plan::partition_columns
  /// order. Empty when the table has no hive partitions. partition_inject_fn consumes this directly
  /// instead of re-parsing a file path at execute time.
  std::vector<std::string> partition_values;
  /// GPU memory space for allocating output tables produced by execute().
  cucascade::memory::memory_space* gpu_memory_space = nullptr;
};

}  // namespace sirius::op::scan
