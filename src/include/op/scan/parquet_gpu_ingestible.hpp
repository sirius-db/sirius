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
#include <helper/logical_type.hpp>
#include <io/gpu_ingestible.hpp>
#include <op/scan/row_group_metadata.hpp>  // row_group_slice + hybrid_scan_reader
#include <op/scan/scan_plan.hpp>
#include <sirius_config.hpp>

// duckdb
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/multi_file/multi_file_data.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/table_filter.hpp>

// cudf
#include <cudf/io/parquet.hpp>

// standard library
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::scan_manager {
class sirius_scan_manager;
}  // namespace sirius::scan_manager

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// parquet_ingestible_table_info
//===----------------------------------------------------------------------===//
/**
 * @brief Parquet bind-data carrier; factory for @c parquet_gpu_ingestible.
 *
 * Populated once by the pipeline converter from the DuckDB
 * @c parquet_scan binding, parked on the gpu scan operator until
 * @c sirius_scan_manager::prepare_for_query consumes it.
 */
class parquet_ingestible_table_info : public io::ingestible_table_info {
 public:
  duckdb::vector<sirius::logical_type> returned_types;
  std::vector<std::string> resolved_file_paths;
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<duckdb::idx_t> projection_ids;
  duckdb::vector<std::string> names;
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  duckdb::vector<duckdb::HivePartitioningIndex> partition_indices;
  std::size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE;
  std::size_t scan_output_arity      = 0;
  /// Maximum number of files handled by one metadata-scan task. One file per
  /// task gives the scan-side balancing_strategy the finest placement
  /// granularity: each file lands on a different GPU via round-robin, spreading
  /// I/O and decode work evenly across all GPUs. Coarser values (e.g. 8) would
  /// batch all files into a single task and prevent cross-GPU distribution.
  std::size_t max_file_processed = 1;

  parquet_ingestible_table_info() = default;

  std::shared_ptr<io::gpu_ingestible> make_ingestible(
    std::unique_ptr<io::ingestible_table_info> self,
    scan_manager::sirius_scan_manager const& mgr) override;

  [[nodiscard]] std::span<std::string const> file_paths() const override
  {
    return std::span<std::string const>(resolved_file_paths.data(), resolved_file_paths.size());
  }
};

//===----------------------------------------------------------------------===//
// parquet_split_info
//===----------------------------------------------------------------------===//
/**
 * @brief Per-split scan metadata for a parquet row-group batch.
 *
 * One emitted by @c parquet_gpu_ingestible::next_split_provider for each
 * row-group partition. Carries everything @c materialize_table needs to
 * issue the read: the byte-range slices (with their per-file ioctx +
 * io_object), the shared reader options (column projection / filter
 * pushdown), the canonical scan plan, and the per-batch pushdown safety
 * flag.
 */
class parquet_split_info : public io::scan_info {
 public:
  /// Row-group slices for this batch — possibly across multiple parquet
  /// files when the per-file row groups don't fill the byte budget.
  std::vector<row_group_slice> rg_slices;
  /// Shared parquet_reader_options (column projection, filter pushdown
  /// when AST translation succeeded). Shared across every split emitted
  /// by the same batch.
  std::shared_ptr<cudf::io::parquet_reader_options> reader_options;
  /// Canonical scan_plan for the table, shared across every split of
  /// this ingestible.
  std::shared_ptr<scan_plan const> plan;
  /// When true, @c materialize_table MUST NOT call @c set_filter on its
  /// reader options; the parquet file has a FLBA-decimal column whose
  /// row-group stats cudf cannot compare against an AST literal. The
  /// filter still applies post-decode via @c gpu_expression_executor.
  bool disable_filter_pushdown = false;
  /// Hive partition values for this split, in @c scan_plan::partition_columns
  /// order. Empty when the plan has no partition columns. Duplicated here
  /// (also lives on @c parquet_post_filter_and_projection_info) so
  /// @ref materialize_table can call @c assemble_scan_output inline on the
  /// reader-side pushdown path and emit @c filter_state::ROW_FILTERED_AND_PROJECTED.
  std::vector<std::string> partition_values;
  /// Whether the scan plan needs post-decode assembly (hive partition
  /// injection or column reordering). Mirrors @c needs_output_assembly(*plan).
  bool needs_assembly = false;

  [[nodiscard]] std::size_t estimated_bytes() const noexcept override
  {
    std::size_t total = 0;
    for (auto const& s : rg_slices) {
      total += s.reserved_uncompressed_bytes;
    }
    return total;
  }
};

//===----------------------------------------------------------------------===//
// parquet_post_filter_and_projection_info
//===----------------------------------------------------------------------===//
/**
 * @brief Per-split post-decode assembly description.
 *
 * Emitted only when @c needs_output_assembly(*plan) is true for the
 * batch — assembly is the only post-decode work parquet does (the filter
 * is fully handled inside @c materialize_table, either via pushdown or
 * via a post-decode @c gpu_expression_executor). @c partition_values is
 * shared across the whole batch because every file in the batch carries
 * identical hive values (enforced at emission).
 */
class parquet_post_filter_and_projection_info : public io::post_filter_and_projection_info {
 public:
  /// Hive partition values for the split, in @c scan_plan::partition_columns
  /// order. Empty when the plan has no partition columns.
  std::vector<std::string> partition_values;
};

//===----------------------------------------------------------------------===//
// parquet_gpu_ingestible
//===----------------------------------------------------------------------===//
/**
 * @brief Concrete @c io::gpu_ingestible for parquet sources.
 *
 * Owns the shared scan plan and coalesced filter expression; pre-decomposes
 * the file list into per-task batches in its constructor (one batch per
 * @c max_file_processed files). @ref next_split_provider atomically claims
 * the next batch index and returns a callable that runs the footer-read /
 * row-group-pruning / partition-by-bytes work — port of
 * @c parquet_split_provider::run_batch.
 *
 * @ref materialize_table is the per-task read + filter step (port of
 * @c sirius_gpu_parquet_scan_operator::read_table_from_metadata, minus
 * assembly). @ref post_filter_and_project does assembly only.
 */
class parquet_gpu_ingestible : public io::gpu_ingestible {
 public:
  /// Built by @c parquet_ingestible_table_info::make_ingestible. The base
  /// @c _table_info owns the parquet bind data; this constructor casts it
  /// back to @c parquet_ingestible_table_info for typed access.
  parquet_gpu_ingestible(std::unique_ptr<io::ingestible_table_info> info,
                         scan_manager::sirius_scan_manager const& mgr);

  ~parquet_gpu_ingestible() override;

  [[nodiscard]] bool has_more_splits() const override;
  std::function<std::vector<std::unique_ptr<op::operator_data>>()> next_split_provider() override;

  io::filtered_table materialize_table(io::scan_info const& info,
                                       ::cucascade::memory::memory_space const& mem_space,
                                       rmm::cuda_stream_view stream) override;

  std::unique_ptr<cudf::table> post_filter_and_project(
    std::unique_ptr<cudf::table> input,
    io::post_filter_and_projection_info const& info,
    ::cucascade::memory::memory_space const& mem_space,
    rmm::cuda_stream_view stream) override;

 private:
  /// One per-task batch of files. The footer-read loop in @ref run_batch
  /// walks these files sequentially, building up a single output vector
  /// of @c scan_operator_input splits.
  struct file_batch {
    std::vector<std::string> file_paths;
  };

  void run_batch(file_batch const& batch, std::vector<std::unique_ptr<op::operator_data>>& out);

  // Canonical scan plan — built once in the constructor, shared by every
  // emitted split via its parquet_split_info::plan member.
  std::shared_ptr<scan_plan const> _plan;
  // Coalesced DuckDB filter expression. Empty when no filters survived the
  // partition-column drop pass.
  std::shared_ptr<duckdb::Expression> _duckdb_filter_expression;
  std::vector<std::string> _file_paths;
  std::size_t _approximate_batch_size{};
  std::size_t _max_file_processed{};
  std::size_t _total_files{};
  scan_manager::sirius_scan_manager const* _scan_manager{nullptr};

  std::vector<file_batch> _batches;
  std::atomic<std::size_t> _next_batch_idx{0};
};

}  // namespace sirius::op::scan
