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
#include "op/sirius_physical_operator_type.hpp"

// duckdb
#include "duckdb/common/column_index.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Operator that parses parquet file metadata and produces row-group partitions.
 *
 * This operator replaces the metadata-parsing work that was previously performed
 * in parquet_scan_task_global_state's constructor, moving it into the execute()
 * method as required by the design principle that no compute should be done in
 * global or local task state.
 *
 * Pipeline role:
 *   - Source of the metadata-scan pipeline (pipeline 1).
 *   - get_next_task_input_data() returns parquet_metadata_input (up to
 *     max_file_processed files per task).
 *   - execute() parses parquet footers, builds partitioned_parquet_metadata.
 *   - The sink of pipeline 1 is sirius_gpu_parquet_scan_operator, which
 *     accumulates the produced partitioned_parquet_metadata objects.
 *
 * This operator intentionally has no dependency on sirius_physical_parquet_scan so
 * that it can be placed before that operator in the physical plan.  The caller
 * is responsible for extracting all required fields from the scan operator (or
 * other plan nodes) and passing them here directly.
 *
 * @pre The caller must validate before construction that:
 *   - The table function is NOT an in-out function (in_out_function == false).
 *   - There are no dynamic table filters (dynamic_filters == nullptr).
 *   - file_paths is non-empty.
 */
class sirius_parquet_metadata_scan_operator : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::PARQUET_METADATA_SCAN;

  /// Default number of files processed per metadata-scan task.
  static constexpr size_t DEFAULT_MAX_FILE_PROCESSED = 8;

  //===----------Constructor----------===//
  /**
   * @brief Construct the metadata scan operator from the individual fields extracted from the
   *        physical parquet scan node (or equivalent source).
   *
   * @param types                Output column types.
   * @param estimated_cardinality  Estimated output row count.
   * @param file_paths           All parquet file paths to scan.
   * @param column_ids           Column ids exposed by the table function (used for column
   *                             selection; see detail::make_selected_column_indices).
   * @param projection_ids       Indices into column_ids that the planner has projected out
   *                             (empty = no projection, read all columns).
   * @param names                All column names in schema order (used to build column-name
   *                             projections passed to the parquet reader).
   * @param approximate_batch_size  Target uncompressed bytes per row-group partition.
   * @param max_file_processed   Maximum number of files handled by one metadata task.
   */
  sirius_parquet_metadata_scan_operator(duckdb::vector<duckdb::LogicalType> types,
                                        duckdb::idx_t estimated_cardinality,
                                        std::vector<std::string> file_paths,
                                        duckdb::vector<duckdb::ColumnIndex> const& column_ids,
                                        duckdb::vector<duckdb::idx_t> const& projection_ids,
                                        duckdb::vector<std::string> const& names,
                                        size_t approximate_batch_size,
                                        size_t max_file_processed = DEFAULT_MAX_FILE_PROCESSED);

  //===----------Source interface----------===//
  bool is_source() const override { return true; }

  /**
   * @brief Returns READY (pointing to itself) while there are unprocessed files,
   *        or nullopt when all files have been dispatched.
   */
  std::optional<task_creation_hint> get_next_task_hint() override;

  /**
   * @brief Returns true once all files have been dispatched to metadata tasks.
   *
   * @pre  Called only from pipeline scheduling logic; no external ordering required.
   * @post Returns true iff _next_file_idx >= _total_files (all files dispatched).
   * @note Overrides the default port-based check since this is a source operator with
   *       no input ports.
   */
  [[nodiscard]] bool all_ports_empty() override;

  /**
   * @brief Creates a parquet_metadata_input for the next batch of unprocessed files.
   *
   * Atomically advances the file-index counter and returns up to max_file_processed
   * file paths.  Returns nullptr when all files have been consumed.
   */
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //===----------Execution----------===//
  /**
   * @brief Parse parquet metadata for the files in @p input_data and produce
   *        a partitioned_parquet_metadata.
   *
   * @param input_data  Must be a parquet_metadata_input instance.
   * @param stream      CUDA stream (unused; metadata parsing is CPU-only).
   * @return            A partitioned_metadata_operator_data wrapping a shared_ptr to the
   *                    partitioned_parquet_metadata containing the parsed FileMetaData
   *                    objects, reader options, and row-group partitions.
   */
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //===----------Accessors----------===//
  [[nodiscard]] size_t get_total_files() const { return _total_files; }
  [[nodiscard]] size_t get_max_file_processed() const { return _max_file_processed; }
  [[nodiscard]] size_t get_approximate_batch_size() const { return _approximate_batch_size; }

 private:
  /// All parquet file paths for this scan.
  std::vector<std::string> _file_paths;
  /// Column indices to read (after projection), indices into parquet schema.
  std::vector<size_t> _selected_column_indices;
  /// Whether projection is applied.
  bool _is_projected;
  /// Column names used when setting column-name projection on reader options.
  std::vector<std::string> _projected_column_names;

  size_t _approximate_batch_size;
  size_t _max_file_processed;
  size_t _total_files;

  /// Atomic file-batch counter; incremented by get_next_task_input_data().
  std::atomic<size_t> _next_file_idx{0};
};

}  // namespace sirius::op::scan
