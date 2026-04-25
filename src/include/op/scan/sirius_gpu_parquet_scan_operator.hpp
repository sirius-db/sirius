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
#include <scan_manager/split_connector.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <memory>
#include <optional>

namespace sirius::scan_manager {
class split_provider;
}  // namespace sirius::scan_manager

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Parquet scan operator
//===----------------------------------------------------------------------===//
/**
 * @brief Operator that reads parquet byte ranges for a batch of row groups and produces
 *        gpu_table_representation data batches for downstream GPU operators.
 *
 * This operator is the source of the GPU parquet scan pipeline. Its splits — one
 * parquet_scan_data per row-group partition — are produced by a parquet_split_provider
 * driven by the scan_manager on its own thread pool, and pushed into the operator's
 * bound split_connector. The operator pulls splits from the connector via
 * get_next_task_input_data(); get_next_task_hint() reports READY (self) until the
 * connector is closed and drained, then nullopt.
 */
class sirius_gpu_parquet_scan_operator : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::GPU_PARQUET_SCAN;

  //===----------Constructor----------===//
  /**
   * @param types                  Output column types (forwarded from the parquet scan operator).
   * @param estimated_cardinality  Estimated row count.
   */
  sirius_gpu_parquet_scan_operator(duckdb::vector<sirius::logical_type> types,
                                   duckdb::idx_t estimated_cardinality);

  ~sirius_gpu_parquet_scan_operator() override;

  //===----------Split-connector binding----------===//
  /**
   * @brief Replace the bound split_connector. Intended for the scan_manager to install
   *        its own connector during query preparation; before this is called, the
   *        operator uses an internally-allocated default connector.
   */
  void set_split_connector(std::unique_ptr<scan_manager::split_connector> connector);

  /**
   * @brief Get the bound split_connector for split production. Never null — the
   *        operator default-allocates a connector at construction.
   */
  scan_manager::split_connector* get_split_connector() noexcept
  {
    return _split_connector.get();
  }

  //===----------Split-provider handoff to scan_manager----------===//
  /**
   * @brief Attach the split provider constructed during plan generation.
   *
   * The pipeline converter extracts bind_data and builds a parquet_split_provider
   * here so the scan_manager can take ownership during prepare_for_query and
   * start it on the scan-manager thread pool.
   */
  void attach_split_provider(std::unique_ptr<scan_manager::split_provider> provider);

  /**
   * @brief Take ownership of the attached split provider. Returns nullptr if
   *        none was attached.
   */
  std::unique_ptr<scan_manager::split_provider> take_split_provider();

  /**
   * @brief Install a hive-partition injection function.
   *
   * Called once by the paired split provider during plan generation when the scan
   * involves hive partition columns. The closure, built by build_partition_inject_fn(),
   * is invoked by execute() after the data columns have been read from the parquet file,
   * to interleave partition-column values parsed from the file path into the output table.
   *
   * No-op (leaves _hive_partition_inject_fn null) for non-partitioned scans.
   */
  void set_hive_partition_inject_fn(partition_inject_fn_t fn)
  {
    _hive_partition_inject_fn = std::move(fn);
  }

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
   * @return the next split, or nullptr when no split is currently available
   *         (either the connector is empty-but-open, or it has been closed and
   *         drained). Returning nullptr does not imply the scan is finished —
   *         the caller must consult get_next_task_hint() / all_ports_empty()
   *         for that.
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
  partition_inject_fn_t _hive_partition_inject_fn;
  std::unique_ptr<scan_manager::split_connector> _split_connector;
  std::unique_ptr<scan_manager::split_provider> _split_provider;
};

}  // namespace sirius::op::scan
