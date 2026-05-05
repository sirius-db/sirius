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
#include <op/scan/hive_partition.hpp>
#include <op/scan/parquet_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <memory>
#include <optional>

namespace sirius::scan_manager {
class split_connector;
class sirius_scan_manager;
}  // namespace sirius::scan_manager

namespace sirius::op::scan {

struct scan_info;

//===----------------------------------------------------------------------===//
// Parquet scan operator
//===----------------------------------------------------------------------===//
/**
 * @brief Operator that reads parquet byte ranges for a batch of row groups and produces
 *        gpu_table_representation data batches for downstream GPU operators.
 *
 * The operator carries a scan_info (populated by the pipeline converter as a concrete
 * subclass — parquet_scan_info today). The scan_manager reads it during prepare_for_query,
 * checks for a pinned-cache match against the scan_info's common fields, and otherwise
 * dispatches through scan_info::make_provider() to build the format's split_provider.
 * Splits — one parquet_scan_data per row-group partition — are pushed into the operator's
 * bound split_connector by the provider on the scan_manager thread pool. The operator
 * pulls splits via get_next_task_input_data(), which blocks inside
 * split_connector::get_next_split until a split arrives or the connector is closed.
 */
class sirius_gpu_parquet_scan_operator : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::GPU_PARQUET_SCAN;

  //===----------Constructor----------===//
  /**
   * @param types                  Output column types (forwarded from the parquet scan operator).
   * @param estimated_cardinality  Estimated row count.
   * @param scan_info              Bind-data extracted by the pipeline converter, consumed
   *                               by the scan_manager during prepare_for_query.
   */
  sirius_gpu_parquet_scan_operator(duckdb::vector<sirius::logical_type> types,
                                   duckdb::idx_t estimated_cardinality,
                                   std::unique_ptr<scan_info> scan_info);

  /// Test-only delegating constructor. The pipeline converter always builds the
  /// 3-arg form; unit tests that don't exercise the scan_manager path use this
  /// to keep their old @c (types, estimated_cardinality) call sites unchanged.
  sirius_gpu_parquet_scan_operator(duckdb::vector<sirius::logical_type> types,
                                   duckdb::idx_t estimated_cardinality)
    : sirius_gpu_parquet_scan_operator(std::move(types), estimated_cardinality, nullptr)
  {
  }

  ~sirius_gpu_parquet_scan_operator() override;

  //===----------Source interface----------===//
  bool is_source() const override { return true; }

  //===----------Scheduling interface----------===//
  /**
   * @return nullopt once the bound split_connector is closed and drained;
   *         READY pointing at this operator otherwise.
   */
  std::optional<task_creation_hint> get_next_task_hint() override;

  /**
   * @return true once the bound split_connector is closed and drained.
   */
  [[nodiscard]] bool all_ports_empty() override;

  /**
   * @brief Pull the next parquet_scan_data from the bound split_connector.
   *
   * Blocks inside split_connector::get_next_split until a split is available or
   * the connector is closed.
   *
   * @return the next split, or nullptr when the connector is closed and drained.
   */
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //===----------Execution----------===//
  /**
   * @brief Produce a gpu_table_representation data batch from @p input_data.
   *
   * Two input shapes are supported:
   *   - parquet_scan_data: the byte ranges described by the split are read from disk via
   *     read_table_from_metadata(), with optional filter pushdown / post-read filtering
   *     and (when scan_plan needs assembly) a final reshape to the plan's output layout.
   *   - scan_cached_operator_data: the table is already pinned in GPU memory; the filter
   *     expression on the split is applied, then the plan's output assembly is applied
   *     when needed. When neither is needed the cached batch is forwarded unchanged.
   *
   * @param input_data  Must be either parquet_scan_data or scan_cached_operator_data.
   * @param stream      CUDA stream.
   * @return gpu_table_representation data batch wrapped as pipelineable_operator_data.
   * @throws std::runtime_error if @p input_data is of an unsupported type, or if a
   *         parquet_scan_data does not have an associated gpu memory space.
   */
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  /**
   * @brief Estimate peak GPU memory for this operator when no execution history is available.
   *
   * @param stats  Batch count and total input bytes for the task about to run.
   * @return Estimated peak GPU bytes this operator will allocate.
   */
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

 private:
  /// Read the parquet byte ranges described by @p scan_data and apply the post-read
  /// filter (when not pushed down) and the scan_plan's output assembly. Used by
  /// execute() when the input split is a parquet_scan_data.
  std::unique_ptr<cudf::table> read_table_from_metadata(const parquet_scan_data& scan_data,
                                                        rmm::cuda_stream_view stream);

  // The scan_manager owns the wiring between this operator and its split_provider.
  // No other code should reach into scan_info / connector, so those entry points
  // are private and exposed only through this friend.
  friend class scan_manager::sirius_scan_manager;

  /// \brief Take ownership of the bind-data so the scan_manager's factory can build a
  ///        split provider. Called once per query; subsequent calls return nullptr.
  std::unique_ptr<scan_info> take_scan_info();

  /// \brief Install the split connector that the scan_manager will feed.
  void set_split_connector(std::unique_ptr<scan_manager::split_connector> connector);

  /// \brief Connector accessor for the scan_manager driver.
  scan_manager::split_connector* get_split_connector() noexcept { return _split_connector.get(); }

  //===----------Fields----------===//
  std::unique_ptr<scan_manager::split_connector> _split_connector;
  std::unique_ptr<scan_info> _scan_info;
};

}  // namespace sirius::op::scan
