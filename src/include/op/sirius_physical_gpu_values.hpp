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

#include "duckdb/common/optionally_owned_ptr.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "op/sirius_physical_operator.hpp"

#include <atomic>

namespace sirius::op {

/**
 * @brief Task input descriptor for sirius_physical_gpu_values.
 *
 * Source operator data: owns no upstream batches, so prepare_for_processing
 * only captures the task reservation's memory space so
 * sirius_physical_gpu_values::execute knows where to materialize the output
 * table (mirrors scan_operator_input). Deliberately NOT a
 * pipelineable_operator_data: the task creator treats empty-batch
 * pipelineable input as "no data" and would skip task creation.
 */
class gpu_values_input : public operator_data {
 public:
  explicit gpu_values_input(std::size_t estimated_bytes) : _estimated_bytes(estimated_bytes) {}

  [[nodiscard]] operator_data_type get_type() const override
  {
    return operator_data_type::GPU_VALUES;
  }

  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              rmm::cuda_stream_view /*stream*/) override
  {
    _gpu_memory_space = const_cast<::cucascade::memory::memory_space*>(requested_memory_space);
  }

  /// Feeds the reservation system's input_basis (see
  /// gpu_pipeline_task_local_state::get_task_consumption_basis).
  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    return _estimated_bytes;
  }

  /// Memory space captured by prepare_for_processing where execute()
  /// materializes the output table.
  [[nodiscard]] ::cucascade::memory::memory_space* get_gpu_memory_space() const
  {
    return _gpu_memory_space;
  }

 private:
  ::cucascade::memory::memory_space* _gpu_memory_space = nullptr;
  std::size_t _estimated_bytes;
};

/**
 * @brief GPU source operator for plan-materialized CPU-side data.
 *
 * Replaces DuckDB source operators whose data already exists at plan time —
 * COLUMN_DATA_SCAN (VALUES clauses, materialized subqueries, statistics
 * propagation results), DUMMY_SCAN (constant-only queries like SELECT 42),
 * and EMPTY_RESULT (WHERE false) — with a first-class pipeline source that
 * is driven by the normal task creator loop and executes as a
 * gpu_pipeline_task. execute() converts the data into a cudf table in the
 * task reservation's GPU memory space.
 *
 * Single-shot: the source's entire content fits one task, so exactly one
 * task input is ever handed out. Unlike GPU_SCAN there is no split
 * connector, scan manager involvement, or ingestible — the data is small
 * and already in memory.
 */
class sirius_physical_gpu_values : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::GPU_VALUES;

  /// Construct from a COLUMN_DATA_SCAN operator that owns a ColumnDataCollection.
  sirius_physical_gpu_values(duckdb::vector<sirius::logical_type> types,
                             duckdb::idx_t estimated_cardinality,
                             duckdb::optionally_owned_ptr<duckdb::ColumnDataCollection> collection);

  /// Construct for DUMMY_SCAN (produce_single_row = true) or EMPTY_RESULT (false).
  sirius_physical_gpu_values(duckdb::vector<sirius::logical_type> types,
                             duckdb::idx_t estimated_cardinality,
                             bool produce_single_row);

  // -----------------------------
  // Source interface
  // -----------------------------
  bool is_source() const override { return true; }

  std::optional<task_creation_hint> get_next_task_hint() override;
  [[nodiscard]] bool all_ports_empty() override;
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  // -----------------------------
  // Execution
  // -----------------------------
  /**
   * @brief Materialize the source data as a cudf table on the GPU.
   *
   * COLUMN_DATA_SCAN: converts the ColumnDataCollection chunk-by-chunk into
   * device columns. DUMMY_SCAN: produces a one-row all-null table; when the
   * operator has zero output columns, a one-row all-null TINYINT sentinel
   * column is emitted instead because cudf derives table row count from
   * columns (a 0-column table has 0 rows) — downstream constant projections
   * read num_rows and never the sentinel values. EMPTY_RESULT: produces a
   * 0-row table with the declared schema.
   */
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  /**
   * @brief Viability gate: throw for output types the host→device conversion
   *        cannot represent faithfully, triggering DuckDB CPU fallback.
   *
   * Called by the plan generator's replace_with_gpu_values BEFORE the operator
   * is built and before the source's ColumnDataCollection is moved out. HUGEINT/UHUGEINT
   * are 16 bytes in DuckDB but map to 8-byte cuDF types (silent value
   * corruption); sub-int32 decimals and nested types have no cuDF-compatible
   * layout (get_cudf_type would throw inside the row-copy loop, failing the
   * query mid-flight instead of falling back).
   */
  static void throw_if_unsupported_types(const duckdb::vector<sirius::logical_type>& types);

  /**
   * @brief Reject a collection that is too large for the single GPU_VALUES task.
   *
   * GPU_VALUES intentionally materializes its source as one cuDF table. The
   * caller must therefore cap the collection before moving it into this
   * operator; oversized collections fall back to DuckDB's streaming CPU path
   * instead of entering an unsplittable GPU OOM retry loop.
   */
  static void throw_if_collection_too_large(const duckdb::ColumnDataCollection& collection,
                                            std::size_t max_source_bytes);

 private:
  [[nodiscard]] std::size_t estimated_source_bytes() const;

  //! The ColumnDataCollection to convert (nullptr for DUMMY_SCAN / EMPTY_RESULT)
  duckdb::optionally_owned_ptr<duckdb::ColumnDataCollection> _collection;
  //! Whether to produce a single all-null row (DUMMY_SCAN behavior)
  bool _produce_single_row{false};
  //! Latched by the first successful get_next_task_hint. The source is
  //! single-shot: one task converts everything, so concurrent scheduling
  //! requests from the task creator must not both see READY.
  std::atomic<bool> _task_scheduled{false};
  //! Set once get_next_task_input_data hands out the single task input;
  //! drives all_ports_empty so the task creation loop makes exactly one task.
  std::atomic<bool> _input_handed_out{false};
};

}  // namespace sirius::op
