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
#include "cucascade/data/gpu_data_representation.hpp"

#include <expression_executor/gpu_expression_translator_internal.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/sirius_physical_operator.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>

// standard library
#include <cstddef>
#include <functional>
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
 * Used as the unit of work for both the parquet split provider (partitioning) and
 * the GPU scan (byte-range preloading).
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
  parquet_scan_data(std::string file_path,
                    row_group_range rg_range,
                    std::shared_ptr<cudf::io::parquet_reader_options> reader_options,
                    std::shared_ptr<duckdb::Expression> filter_expression,
                    std::shared_ptr<cudf::io::datasource> datasource,
                    std::shared_ptr<scan_plan const> plan)
    : file_path(std::move(file_path)),
      rg_range(std::move(rg_range)),
      reader_options(std::move(reader_options)),
      filter_expression(std::move(filter_expression)),
      datasource(std::move(datasource)),
      plan(std::move(plan))
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
   * @return  Always an empty accessor vector; there are no batches to keep locked.
   *          The vector is typed on cucascade::mutable_data_batch (post-#117
   *          RAII) but is always empty for this source-input override.
   */
  std::optional<std::vector<::cucascade::mutable_data_batch>> prepare_for_processing(
    const ::cucascade::memory::memory_space* requested_memory_space,
    rmm::cuda_stream_view stream) override
  {
    gpu_memory_space = const_cast<cucascade::memory::memory_space*>(requested_memory_space);
    return std::vector<::cucascade::mutable_data_batch>{};
  };

  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    return rg_range.reserved_uncompressed_bytes;
  }

  std::string file_path;
  row_group_range rg_range;
  std::shared_ptr<cudf::io::parquet_reader_options> reader_options;
  /// The coalesced duckdb filter expression.
  std::shared_ptr<duckdb::Expression> filter_expression;
  /// Datasource for the parquet file, shared with other partitions of the same file.
  std::shared_ptr<cudf::io::datasource> datasource;
  /// Scan plan shared across all splits of this scan. Carries the D-order column name
  /// table used by execute()'s per-task AST translation, plus the post-read assembly layout.
  std::shared_ptr<scan_plan const> plan;
  /// GPU memory space for allocating output tables produced by execute().
  cucascade::memory::memory_space* gpu_memory_space = nullptr;
};

//===----------------------------------------------------------------------===//
// scan_cached_operator_data
//===----------------------------------------------------------------------===//
/**
 * @brief Input to a GPU parquet scan task served from a pinned (cached) entry.
 *
 * The cached_split_provider produces one of these per cached batch. It carries
 * a zero-copy data_batch (whose gpu_table_representation is a view over the
 * pinned data columns in scan_plan D-order) plus the filter expression and the
 * post-read assembly closure that the scan operator applies in execute(). The
 * gpu_memory_space lives on the wrapped data_batch.
 *
 * The cached path is gated upstream so it never sees hive partitions, so
 * @ref inject_fn — when non-null — only ever performs DATA-source permutation
 * and pure-filter-column pruning. It is null exactly when scan_plan's output
 * layout is identity over data_columns (no permute, no prune).
 */
class scan_cached_operator_data : public op::operator_data {
 public:
  scan_cached_operator_data(std::shared_ptr<cucascade::data_batch> batch,
                            std::shared_ptr<duckdb::Expression> filter_expression,
                            partition_inject_fn_t inject_fn)
    : batch(std::move(batch)),
      filter_expression(std::move(filter_expression)),
      inject_fn(std::move(inject_fn))
  {
  }

  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    if (!batch) { return 0; }
    // R2 read-only accessor scoped to the single size read on the cached batch.
    auto ro = batch->to_read_only();
    if (!ro.get_data()) { return 0; }
    return ro.get_data()->cast<cucascade::gpu_table_representation>().get_size_in_bytes();
  }

  /// Cached data batch viewed by the scan. Owning shared_ptr keeps the pinned
  /// columns alive for the lifetime of the task.
  std::shared_ptr<cucascade::data_batch> batch;
  /// Coalesced DuckDB filter expression, evaluated post-read in execute() against
  /// the cached batch. Null when no filter applies. The cached path does not carry
  /// an AST-translated filter — pushdown is a parquet-reader concern, and the
  /// cached batch is already materialized.
  std::shared_ptr<duckdb::Expression> filter_expression;
  /// Post-read assembly closure produced by scan_plan::build_inject_fn(). Null
  /// when the plan's output layout is identity over data_columns — execute()
  /// then forwards the cached batch (or filter result) without re-permuting.
  partition_inject_fn_t inject_fn;
};

}  // namespace sirius::op::scan
