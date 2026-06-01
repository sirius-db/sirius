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
#include <data/sirius_converter_registry.hpp>
#include <expression_executor/gpu_expression_translator_internal.hpp>
#include <io/io_context.hpp>
#include <io/types.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/sirius_physical_operator.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/common.hpp>

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
                  std::size_t reserved_compressed_bytes,
                  std::shared_ptr<sirius::io::sirius_ioctx> io_ctx        = nullptr,
                  std::shared_ptr<sirius::io::sirius_io_object> io_object = nullptr)
    : file_metadata(file_metadata),
      file_path(file_path),
      row_group_indices(std::move(row_group_indices)),
      reserved_uncompressed_bytes(reserved_uncompressed_bytes),
      reserved_compressed_bytes(reserved_compressed_bytes),
      io_ctx(std::move(io_ctx)),
      io_object(std::move(io_object))
  {
  }
  std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
  std::string file_path;
  std::vector<cudf::size_type> row_group_indices;
  std::size_t reserved_uncompressed_bytes;
  std::size_t reserved_compressed_bytes;
  /// Sirius IO context that minted @c io_object.  When non-null, the scan
  /// operator builds a @c sirius_datasource via @c io_ctx->make_datasource;
  /// when null, it falls back to @c cudf::io::datasource::create.
  std::shared_ptr<sirius::io::sirius_ioctx> io_ctx;
  /// Sirius io_object created by the split provider for this file.  Shared
  /// across every slice from the same parquet file.
  std::shared_ptr<sirius::io::sirius_io_object> io_object;
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
                    std::vector<std::string> partition_values,
                    bool disable_filter_pushdown = false)
    : rg_slices(std::move(rg_slices)),
      reader_options(std::move(reader_options)),
      filter_expression(std::move(filter_expression)),
      plan(std::move(plan)),
      partition_values(std::move(partition_values)),
      disable_filter_pushdown(disable_filter_pushdown)
  {
  }

  [[nodiscard]] op::operator_data_type get_type() const override
  {
    return op::operator_data_type::PARQUET_SCAN;
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
   */
  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              rmm::cuda_stream_view stream) override
  {
    gpu_memory_space = const_cast<cucascade::memory::memory_space*>(requested_memory_space);
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
  /// order. Empty when the table has no hive partitions. assemble_scan_output consumes this
  /// directly instead of re-parsing a file path at execute time.
  std::vector<std::string> partition_values;
  /// When true, the scan operator MUST NOT call set_filter on its reader options.
  /// Set by parquet_split_provider when any decimal column is stored as
  /// FIXED_LEN_BYTE_ARRAY / BYTE_ARRAY — cudf's row-group stats filter cannot
  /// compare a fixed_point_scalar AST literal against those physical types and
  /// throws "Invalid type and stats combination" (stats_filter_helpers.hpp:132).
  /// The filter still applies post-decode via gpu_expression_executor::select.
  bool disable_filter_pushdown = false;
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
 * pinned data columns in scan_plan D-order) plus the filter expression and a
 * shared scan_plan that execute() consults via @c needs_output_assembly /
 * @c assemble_scan_output. The gpu_memory_space lives on the wrapped data_batch.
 *
 * The cached path is gated upstream so it never sees hive partitions: when
 * assembly is needed, it only performs DATA-source permutation and pure-filter-
 * column pruning. When @c needs_output_assembly(*plan) is false, execute()
 * forwards the cached batch (or the filter result) without re-permuting.
 */
class scan_cached_operator_data : public op::operator_data {
 public:
  scan_cached_operator_data(std::shared_ptr<cucascade::data_batch> batch,
                            std::shared_ptr<duckdb::Expression> filter_expression,
                            std::shared_ptr<scan_plan const> plan)
    : batch(std::move(batch)),
      filter_expression(std::move(filter_expression)),
      plan(std::move(plan))
  {
  }

  [[nodiscard]] op::operator_data_type get_type() const override
  {
    return op::operator_data_type::SCAN_CACHED;
  }

  /**
   * @brief Move host-tier cached batches onto the requested GPU memory space.
   *
   * pin_table tier='host' wraps the cached chunks as host_data_representation; the
   * scan operator's execute() expects gpu_table_representation, so we convert here
   * (when the gpu_pipeline_executor has already reserved the target memory space).
   * GPU-resident batches need no work — execute() consumes them directly.
   */
  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              rmm::cuda_stream_view stream) override
  {
    if (!batch || !requested_memory_space) { return; }
    {
      auto ro = batch->to_read_only();
      if (!ro.get_data() || ro.get_current_tier() == ::cucascade::memory::Tier::GPU) { return; }
    }
    auto& registry = ::sirius::converter_registry::get();
    auto mut       = batch->to_mutable();
    mut.convert_to<cucascade::gpu_table_representation>(registry, requested_memory_space, stream);
  }

  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    // Use the generic representation accessor so this works for both GPU-resident
    // batches and host_data_representation batches that prepare_for_processing
    // will upgrade to GPU. Sizes are close enough between tiers to size the
    // task's memory reservation.
    auto ro          = batch->to_read_only();
    auto const* data = ro.get_data();
    if (!data) { return 0; }
    return data->get_size_in_bytes();
  }

  /// Cached data batch viewed by the scan. Owning shared_ptr keeps the pinned
  /// columns alive for the lifetime of the task.
  std::shared_ptr<cucascade::data_batch> batch;
  /// Coalesced DuckDB filter expression, evaluated post-read in execute() against
  /// the cached batch. Null when no filter applies. The cached path does not carry
  /// an AST-translated filter — pushdown is a parquet-reader concern, and the
  /// cached batch is already materialized.
  std::shared_ptr<duckdb::Expression> filter_expression;
  /// Shared scan plan describing the output layout. execute() queries
  /// @c needs_output_assembly to decide whether to call @c assemble_scan_output
  /// on the filter result (or, when no filter applies, the cached batch).
  std::shared_ptr<scan_plan const> plan;
};

}  // namespace sirius::op::scan
