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

struct parquet_scan_info;

//===----------------------------------------------------------------------===//
// Parquet scan operator
//===----------------------------------------------------------------------===//
/**
 * @brief Operator that reads parquet byte ranges for a batch of row groups and produces
 *        gpu_table_representation data batches for downstream GPU operators.
 *
 * The operator carries a parquet_scan_info populated by the pipeline converter; the
 * scan_manager reads it during prepare_for_query to construct a parquet_split_provider
 * for this operator. Splits — one parquet_scan_data per row-group partition — are
 * pushed into the operator's bound split_connector by the provider on the scan_manager
 * thread pool. The operator pulls splits via get_next_task_input_data(), which blocks
 * inside split_connector::get_next_split until a split arrives or the connector is
 * closed.
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
                                   std::unique_ptr<parquet_scan_info> scan_info);

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
   * @brief Read the byte ranges described by @p input_data from disk and produce a
   *        gpu_table_representation data batch.
   *
   * @param input_data  Must be a parquet_scan_data instance.
   * @param stream      CUDA stream.
   * @return gpu_table_representation data batch wrapped as pipelineable_operator_data
   * @throws std::runtime_error if the input_data is not parquet_scan_data, or the parquet_scan_data
   *         does not have an associated gpu memory space
   */
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

 private:
  // The scan_manager owns the wiring between this operator and its split_provider.
  // No other code should reach into scan_info / connector / hive-inject installation,
  // so those entry points are private and exposed only through this friend.
  friend class scan_manager::sirius_scan_manager;

  /// \brief Take ownership of the bind-data so the scan_manager's factory can build a
  ///        split provider. Called once per query; subsequent calls return nullptr.
  std::unique_ptr<parquet_scan_info> take_scan_info();

  /// \brief Install the split connector that the scan_manager will feed.
  void set_split_connector(std::unique_ptr<scan_manager::split_connector> connector);

  /// \brief Connector accessor for the scan_manager driver.
  scan_manager::split_connector* get_split_connector() noexcept { return _split_connector.get(); }

  /// \brief Install the closure that reshapes the reader's output to the scan_plan's
  ///        D-order layout: reorders data columns, drops pure-filter columns, and
  ///        injects hive-partition columns synthesized from the file path. No-op
  ///        when the scan is a trivial identity (no partitions, 1:1 data layout).
  void set_partition_inject_fn(partition_inject_fn_t fn) { _partition_inject_fn = std::move(fn); }

  //===----------Fields----------===//
  partition_inject_fn_t _partition_inject_fn;
  std::unique_ptr<scan_manager::split_connector> _split_connector;
  std::unique_ptr<parquet_scan_info> _scan_info;
};

}  // namespace sirius::op::scan
