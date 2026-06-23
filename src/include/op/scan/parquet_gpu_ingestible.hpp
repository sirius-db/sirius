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
#include <op/scan/gpu_ingestible.hpp>
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
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// parquet_ingestible_table_info
//===----------------------------------------------------------------------===//
/**
 * @brief Parquet bind-data carrier; factory for @c parquet_gpu_ingestible.
 *
 * Populated once by the pipeline converter from the DuckDB
 * @c parquet_scan binding, parked on the gpu scan operator until
 */
class parquet_ingestible_table_info : public ingestible_table_info {
 public:
  duckdb::vector<sirius::logical_type> returned_types;
  std::vector<std::string> resolved_file_paths;
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<duckdb::idx_t> projection_ids;
  duckdb::vector<std::string> names;
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  duckdb::vector<duckdb::HivePartitioningIndex> partition_indices;
  /// Target uncompressed byte budget for one data-batch split. Consumed only by
  /// parquet_batch_coalecer when it bundles files / chunks row groups — the
  /// ingestible's metadata scan operates one file at a time and does no batching.
  std::size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE;
  std::size_t scan_output_arity      = 0;

  parquet_ingestible_table_info() = default;

  [[nodiscard]] std::span<std::string const> file_paths() const override
  {
    return std::span<std::string const>(resolved_file_paths.data(), resolved_file_paths.size());
  }

  [[nodiscard]] std::span<std::string const> column_names() const override { return names; }

  /// True (non-empty) iff @p other reads the same set of files as this scan and
  /// every column @p other requests is also read by this scan — i.e. this scan's
  /// (cached) data is a superset that can serve @p other. Files are matched as a
  /// set (order-independent); columns by primary (storage) index, as in
  /// @c duckdb_native_ingestible_table_info.
  ///
  /// Returns, for each column @p other requests (in @p other's @c column_ids
  /// order), the position of that column within THIS scan's @c column_ids — a
  /// gather index into this scan's (cached) materialized columns that reproduces
  /// @p other's requested layout. Returns empty when this scan cannot serve
  /// @p other (different scan type, different file set, or a missing column).
  [[nodiscard]] std::vector<std::size_t> can_serve_with_columns(
    const ingestible_table_info& other) const override
  {
    auto const* o = dynamic_cast<parquet_ingestible_table_info const*>(&other);
    if (o == nullptr) { return {}; }

    // Same set of files (order-independent equality).
    if (resolved_file_paths.size() != o->resolved_file_paths.size()) { return {}; }
    auto these_files = resolved_file_paths;
    auto those_files = o->resolved_file_paths;
    std::sort(these_files.begin(), these_files.end());
    std::sort(those_files.begin(), those_files.end());
    if (these_files != those_files) { return {}; }

    // Map this scan's columns by primary (storage) index → position, then build
    // the gather projection. Every column other requests must be present.
    std::unordered_map<duckdb::idx_t, std::size_t> this_pos;
    this_pos.reserve(column_ids.size());
    for (std::size_t i = 0; i < column_ids.size(); ++i) {
      this_pos.emplace(column_ids[i].GetPrimaryIndex(), i);
    }

    std::vector<std::size_t> projection;
    projection.reserve(o->column_ids.size());
    for (auto const& c : o->column_ids) {
      auto it = this_pos.find(c.GetPrimaryIndex());
      if (it == this_pos.end()) { return {}; }  // this scan lacks a requested column
      projection.push_back(it->second);
    }
    return projection;
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
class parquet_split_info : public scan_info {
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

  /// One fadvise_entry per row-group slice: the slice's datasource paired with
  /// the column-chunk byte ranges the read will fetch for that file's row groups
  /// (computed via @c hybrid_scan_reader::all_column_chunks_byte_ranges, honoring
  /// the reader_options column projection). Drives prefetch for the materialize
  /// read across every file in the batch.
  [[nodiscard]] std::vector<fadvise_entry> fadvise_entries() const override;
};

//===----------------------------------------------------------------------===//
// parquet_file_scan_info
//===----------------------------------------------------------------------===//
/**
 * @brief Per-file metadata unit emitted by @ref next_split_provider.
 *
 * Each metadata-scan task processes exactly one parquet file: it reads the
 * footer, prunes row groups against the filter, and records per-row-group byte
 * accounting. The result is one @c parquet_file_scan_info. File batching and
 * row-group chunking by byte budget are then performed downstream by
 * @c parquet_batch_coalecer, which coalesces these into @c parquet_split_info
 * data batches. The shared @c reader_options and @c scan_plan live on the
 * ingestible and are stamped onto the emitted splits by the coalescer.
 */
class parquet_file_scan_info : public scan_info {
 public:
  /// A single pruned row group with the byte accounting the coalescer chunks on.
  /// @c uncompressed_bytes excludes pure-filter columns (read for filtering but
  /// not emitted), matching the legacy @c rg_contribution accounting.
  struct row_group_entry {
    cudf::size_type index;
    std::size_t uncompressed_bytes;
    std::size_t compressed_bytes;
  };

  /// Parsed footer metadata for this file.
  std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
  /// File path (also the datasource cache key).
  std::string file_path;
  /// Pre-built datasource for this file, reused by @c materialize_table. May be
  /// null for local paths no sirius backend claims.
  std::shared_ptr<io::sirius_datasource> datasource;
  /// Pruned row groups for this file, in file order, with byte accounting.
  std::vector<row_group_entry> row_groups;
  /// Shared reader options (column projection), used to compute the column-chunk
  /// byte ranges for @ref fadvise_entries. Same options the coalescer stamps onto
  /// the emitted @c parquet_split_info.
  std::shared_ptr<cudf::io::parquet_reader_options> reader_options;
  /// Hive partition values for this file, in @c scan_plan::partition_columns
  /// order. Empty when the plan has no partition columns.
  std::vector<std::string> partition_values;
  /// When true, this file has an FLBA-decimal column whose row-group stats cudf
  /// cannot compare against an AST literal — reader-side pushdown must be
  /// disabled for any split that includes it.
  bool disable_filter_pushdown = false;

  [[nodiscard]] std::size_t estimated_bytes() const noexcept override
  {
    std::size_t total = 0;
    for (auto const& rg : row_groups) {
      total += rg.uncompressed_bytes;
    }
    return total;
  }

  /// A single fadvise_entry: this file's datasource paired with the column-chunk
  /// byte ranges the read will fetch for its row groups (via
  /// @c hybrid_scan_reader::all_column_chunks_byte_ranges, honoring the
  /// reader_options column projection).
  [[nodiscard]] std::vector<fadvise_entry> fadvise_entries() const override;
};

//===----------------------------------------------------------------------===//
// parquet_gpu_ingestible
//===----------------------------------------------------------------------===//
/**
 * @brief Concrete @c io::gpu_ingestible for parquet sources.
 *
 * Owns the shared scan plan, reader options, and coalesced filter expression.
 * @ref next_split_provider hands out one file at a time: each metadata-scan task
 * reads that file's footer, prunes its row groups, and emits a single
 * @c parquet_file_scan_info. File bundling and row-group chunking by byte budget
 * are handled downstream by @c parquet_batch_coalecer (@ref create_batch_coalecer),
 * which coalesces the per-file units into @c parquet_split_info data batches.
 *
 * @ref materialize_table is the per-split read + filter step; it also assembles
 * hive-partition output inline (it owns the per-split partition values).
 * @ref post_filter_and_project applies a pending filter and non-partition
 * projection.
 */
class parquet_gpu_ingestible : public gpu_ingestible {
 public:
  /// Built by @c parquet_ingestible_table_info::make_ingestible. The base
  /// @c _table_info owns the parquet bind data; this constructor casts it
  /// back to @c parquet_ingestible_table_info for typed access.
  explicit parquet_gpu_ingestible(std::unique_ptr<parquet_ingestible_table_info> info);

  ~parquet_gpu_ingestible() override;

  std::unique_ptr<batch_coalecer> create_batch_coalecer() const override;

  [[nodiscard]] bool has_processed_all_metadata() const override;

  metadata_scan_task_t next_split_provider(std::shared_ptr<io::sirius_ioctx> io_ctx) override;

  filtered_table materialize_metadata_to_table(scan_info const& info,
                                               const cucascade::memory::memory_space& mem_space,
                                               rmm::cuda_stream_view stream) override;

  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&& table,
    const cucascade::memory::memory_space& mem_space,
    rmm::cuda_stream_view stream) override;

  [[nodiscard]] const ingestible_table_info& table_info() const noexcept override { return *_info; }

 private:
  /// Read one file's footer, prune its row groups against the filter, and record
  /// per-row-group byte accounting. Returns a single @c parquet_file_scan_info.
  /// Runs on a scan-manager dispatcher thread (the task returned by
  /// @ref next_split_provider).
  std::unique_ptr<scan_info> build_file_scan_info(std::string const& file_path,
                                                  std::shared_ptr<io::sirius_ioctx> const& io_ctx);

  std::unique_ptr<parquet_ingestible_table_info> _info;

  // Canonical scan plan — built once in the constructor, shared by every
  // emitted split via its parquet_split_info::plan member.
  std::shared_ptr<scan_plan const> _plan;
  // Shared reader options (column projection only — never set_filter, which is
  // a per-split decision applied in materialize_table). Built once in the
  // constructor and stamped onto every emitted split by the coalescer.
  std::shared_ptr<cudf::io::parquet_reader_options> _reader_options;
  // Coalesced DuckDB filter expression. Empty when no filters survived the
  // partition-column drop pass.
  std::shared_ptr<duckdb::Expression> _duckdb_filter_expression;
  std::vector<std::string> _file_paths;

  // Per-file metadata-scan cursor. next_split_provider hands out one file index
  // per claim; the coalescer downstream batches files and chunks row groups.
  std::atomic<std::size_t> _next_file_idx{0};
};

std::shared_ptr<parquet_gpu_ingestible> make_ingestible(
  std::unique_ptr<parquet_ingestible_table_info> info);

}  // namespace sirius::op::scan
