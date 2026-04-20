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
#include <config.hpp>
#include <expression_executor/gpu_expression_translator.hpp>
#include <op/scan/sirius_gpu_parquet_scan_operator.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>
#include <sirius_config.hpp>

// duckdb
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

// standard library
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Operator that parses parquet file metadata and produces row-group partitions.
 *
 * Pipeline role:
 *   - Both source and sink of the metadata-scan pipeline (pipeline 1), which is a
 *     single-op pipeline containing only this operator.
 *   - get_next_task_input_data() returns parquet_metadata_input (up to
 *     max_file_processed files per task).
 *   - execute() parses parquet footers, builds partitioned_parquet_metadata.
 *   - sink() forwards each produced partitioned_parquet_metadata into the paired
 *     sirius_gpu_parquet_scan_operator via its accumulate_metadata() entry point.
 *     This is a direct handoff — no inter-pipeline port or data repository is
 *     involved; the metadata never flows through the generic pipeline data path.
 *   - finalize_operator() (invoked once after pipeline 1 completes) calls
 *     gpu_scan::finalize_partitions(), which freezes the partition index and
 *     unblocks the downstream scan pipeline (pipeline 2), where gpu_scan serves
 *     as the source.
 *
 * @pre The caller must validate before construction that:
 *   - The table function is NOT an in-out function (in_out_function == false).
 *   - There are no dynamic table filters (dynamic_filters == nullptr).
 *   - file_paths is non-empty.
 */
class sirius_parquet_metadata_scan_operator : public sirius_physical_operator {
 public:
  using translated_expression = gpu_expression_translator::translated_expression;

  /// The physical operator type for this operator.
  static constexpr SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::PARQUET_METADATA_SCAN;

  /// Default number of files processed per metadata-scan task.
  static constexpr size_t DEFAULT_MAX_FILE_PROCESSED = 8;

  //===----------Constructor----------===//
  /**
   * @brief Construct the metadata scan operator from the individual fields extracted from the
   *        physical parquet scan node (or equivalent source).
   *
   * @param gpu_scan                The downstream gpu scan operator into which to push partition
   *                                metadata. This is necessary in order to avoid the gpu scan
   *                                serving as sink operator in both the metadatascan pipeline and
   *                                in a subsequent standalone pipeline for which it is the source
   *                                operator.
   * @param types                   Output column types.
   * @param returned_types          The types of all columns in the source file.
   * @param estimated_cardinality   Estimated output row count.
   * @param file_paths              The list of parquet files to scan.
   * @param column_ids              Column ids exposed by the table function (used for column
   *                                selection; see detail::make_selected_column_indices).
   * @param projection_ids          Indices into column_ids that the planner has projected out
   *                                (empty = no projection, read all columns).
   * @param names                   All column names in schema order (used to build column-name
   *                                projections passed to the parquet reader).
   * @param table_filter_set        The table filter set for row-group pruning and filter pushdown
   *                                (optional; may be nullptr if no filters or filter translation
   *                                fails).
   * @param approximate_batch_size  Target uncompressed bytes per row-group partition.
   * @param max_file_processed      Maximum number of files handled by one metadata task.
   *
   * @throws if projection_ids is nonempty or filter_expression is non-nullptr but names is empty
   *         (column names are required for both projection and filter pushdown).
   */
  sirius_parquet_metadata_scan_operator(
    sirius_gpu_parquet_scan_operator* gpu_scan,
    duckdb::vector<sirius::logical_type> types,
    duckdb::vector<sirius::logical_type> const& returned_types,
    duckdb::idx_t estimated_cardinality,
    std::vector<std::string> const& file_paths,
    duckdb::vector<duckdb::ColumnIndex> const& column_ids,
    duckdb::vector<duckdb::idx_t> const& projection_ids,
    duckdb::vector<std::string> const& names,
    duckdb::unique_ptr<duckdb::TableFilterSet> table_filter_set = nullptr,
    std::size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE,
    std::size_t max_file_processed     = DEFAULT_MAX_FILE_PROCESSED);

  //===----------Source interface----------===//
  bool is_source() const override { return true; }

  //===----------Scheduling interface----------===//
  /**
   * @return READY (pointing to itself) while there are unprocessed files,
   *         or nullopt when all files have been dispatched.
   */
  std::optional<task_creation_hint> get_next_task_hint() override;

  /**
   * @brief Returns true once all files have been dispatched to metadata tasks.
   *
   * @return true iff _next_file_idx >= _total_files (all files dispatched).
   * @note Overrides the default port-based check since this is a source operator with
   *       no input ports.
   */
  [[nodiscard]] bool all_ports_empty() override;

  /**
   * @brief Creates a parquet_metadata_input for the next batch of unprocessed files.
   *
   * Atomically advances the file-index counter and returns up to max_file_processed
   * file paths. Returns nullptr when all files have been consumed.
   */
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //===----------Execution----------===//
  /**
   * @brief Parse parquet metadata for the files in @p input_data and produce
   *        a partitioned_parquet_metadata.
   *
   * If a filter expression was provided at construction, this method attempts to translate it
   * into a cuDF AST for filter pushdown and row-group pruning. If column names were not provided
   * to the constructor, or if AST translation fails, the original DuckDB expression is stored on
   * the result for post-scan filtering in the GPU scan operator.
   *
   * @param input_data  Must be a parquet_metadata_input instance.
   * @param stream      CUDA stream used for AST filter translation and row-group pruning.
   * @return            A partitioned_parquet_metadata containing the parsed FileMetaData
   *                    objects, reader options, and row-group partitions.
   * @throws std::runtime_error if @p input_data is not a parquet_metadata_input.
   */
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //===----------Sink interface (forwarding to gpu_scan)----------===//
  bool is_sink() const override { return true; }

  /**
   * @brief Forward a partitioned_parquet_metadata produced by execute() into the paired
   *        sirius_gpu_parquet_scan_operator.
   *
   * Invoked by the pipeline framework once per completed metadata task, from worker threads.
   * Delegates to gpu_scan::accumulate_metadata(), which is thread-safe; no ordering between
   * concurrent sink() calls is required or observed.
   *
   * @param input_data  Must dynamic_cast to partitioned_parquet_metadata.
   * @param stream      Unused; metadata accumulation is CPU-only.
   * @throws std::runtime_error if @p input_data is not a partitioned_parquet_metadata.
   */
  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

  /**
   * @brief Finalize the paired gpu_scan's partition index, unblocking the downstream pipeline.
   *
   * Invoked by the pipeline framework exactly once, after every sink() call for this pipeline
   * has returned. Delegates to gpu_scan::finalize_partitions(), which freezes the accumulated
   * metadata into a flat partition index and publishes it with release semantics. Until this
   * call completes, gpu_scan's source methods report "not ready" and the downstream scan
   * pipeline cannot begin dispatching tasks.
   */
  void finalize_operator() override;

  //===----------Accessors----------===//
  [[nodiscard]] std::size_t get_total_files() const { return _total_files; }
  [[nodiscard]] std::size_t get_max_file_processed() const { return _max_file_processed; }
  [[nodiscard]] std::size_t get_approximate_batch_size() const { return _approximate_batch_size; }

 private:
  /// The list of parquet files to scan.
  std::vector<std::string> _file_paths;
  /// Column indices to read (after projection), indices into parquet schema.
  std::vector<std::size_t> _selected_column_indices;
  /// Whether projection is applied.
  bool _is_projected;
  /// Column names for the projected columns, in column_ids order.
  std::vector<std::string> _projected_column_names;
  /// Whether there is a filter expression (AST translation is deferred to execute()).
  bool _has_filter;
  /// The coalesced DuckDB filter expression (AST translation attempted in execute()).
  std::shared_ptr<duckdb::Expression> _duckdb_filter_expression;
  /// Pre-computed column name lookup for AST translation (ref_index -> column name).
  std::vector<std::string> _column_name_by_ref;
  /// The projection ids corresponding to columns that remain after pruning pure filter columns.
  /// These are passed forward to the GPU scan operator to apply as a post-filter projection after
  /// filter pushdown.
  std::vector<std::size_t> _post_filter_projection_ids;
  /// The set of column indices corresponding to columns that will be pruned after filtering.
  /// For the metadata scan operator, this is used to prune bytes from the accumulated uncompressed
  /// byte count for partitioning purposes.
  std::unordered_set<std::size_t> _pure_filter_column_indices;

  std::size_t _approximate_batch_size;
  std::size_t _max_file_processed;
  std::size_t _total_files;

  /// Atomic file-batch counter; incremented by get_next_task_input_data().
  std::atomic<std::size_t> _next_file_idx{0};

  /// Paired GPU parquet scan operator — the source of the downstream pipeline. sink() forwards
  /// accumulated metadata into it; finalize_operator() freezes its partition index. Set at
  /// construction; never null.
  sirius_gpu_parquet_scan_operator* _gpu_scan;
};

}  // namespace sirius::op::scan
