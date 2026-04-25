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

namespace sirius::op::scan {
class sirius_parquet_metadata_scan_operator;
}  // namespace sirius::op::scan

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Parquet scan operator
//===----------------------------------------------------------------------===//
/**
 * @brief Operator that reads parquet byte ranges for a batch of row groups and produces
 *        gpu_table_representation data batches for downstream GPU operators.
 *
 * This operator is the source of the GPU parquet scan pipeline. It is paired with a
 * sirius_parquet_metadata_scan_operator, which runs as a separate upstream pipeline and
 * streams parsed metadata into this operator — not through the data-batch / port
 * channel, since partitioned_parquet_metadata is not a data_batch.
 *
 * Lifecycle (streaming — the scan pipeline runs concurrently with the metadata pipeline):
 *
 *   1. Metadata accumulation (upstream pipeline, CPU-only):
 *      - sirius_parquet_metadata_scan_operator::execute() parses parquet footers and
 *        produces partitioned_parquet_metadata.
 *      - Its sink() forwards each result here via accumulate_metadata(), which copies
 *        the metadata into a shared_ptr and appends one partition_entry per
 *        row_group_range to _partition_index under _metadata_mutex. Partitions become
 *        claimable by the scan pipeline the moment they are appended.
 *      - When the upstream pipeline finishes, its finalize_operator() calls
 *        finalize_partitions() on this operator, which sets _finalized to signal that
 *        no further partitions will arrive.
 *
 *   2. Scan (this pipeline, GPU):
 *      - get_next_task_hint() returns READY as soon as _partition_index contains an
 *        unclaimed entry, regardless of _finalized. If no entry is currently claimable
 *        and _finalized is false, it surfaces the upstream metadata scan as
 *        WAITING_FOR_INPUT_DATA so task_creator can schedule it.
 *      - get_next_task_input_data() claims one partition from the index under
 *        _metadata_mutex and returns it as parquet_scan_data. Each partition maps 1:1
 *        to a row_group_range — the metadata scan operator is responsible for sizing
 *        partitions to the target batch size.
 *      - execute(parquet_scan_data) reads the byte ranges, optionally applies a
 *        fallback filter expression, and emits a gpu_table_representation data batch.
 *
 * Scheduling coupling:
 *   The upstream → downstream pipeline edge is expressed via a null-repo "handoff"
 *   port on this operator (MemoryBarrierType::PARTIAL). setup_pipeline_parents() uses
 *   that port to discover the dependency so the metadata pipeline is registered as a
 *   parent of this pipeline. No data flows through the port — the handoff is via
 *   accumulate_metadata() / finalize_partitions().
 *
 * Thread-safety:
 *   - accumulate_metadata() is called from upstream worker threads; _metadata_mutex
 *     serializes its appends to _partition_index against concurrent claims by
 *     get_next_task_input_data() and size reads by get_next_task_hint() /
 *     all_ports_empty().
 *   - finalize_partitions() must be called exactly once, after ALL accumulate_metadata()
 *     calls have returned. It sets _finalized with release semantics; get_next_task_hint()
 *     reads it with acquire semantics to decide whether to return nullopt or to defer
 *     to the upstream pipeline.
 *   - get_next_task_input_data() / get_next_task_hint() / all_ports_empty() are safe to
 *     call from multiple worker threads and serve partitions as soon as they are
 *     appended; returning "no work right now" does not imply the scan is finished.
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

  //===----------Metadata-scan-op handoff to scan_manager----------===//
  /**
   * @brief Attach the metadata scan operator constructed during plan generation.
   *
   * The pipeline converter still extracts bind_data and constructs the metadata
   * scan operator. Instead of placing it in its own pipeline, it parks the
   * operator here so the scan_manager can take ownership during prepare_for_query
   * and drive its execute() on the scan-manager thread pool.
   */
  void attach_metadata_scan_op(std::unique_ptr<sirius_parquet_metadata_scan_operator> op);

  /**
   * @brief Take ownership of the attached metadata scan operator. Returns nullptr
   *        if none was attached.
   */
  std::unique_ptr<sirius_parquet_metadata_scan_operator> take_metadata_scan_op();

  //===----------Metadata handoff (called by metadata_scan)----------===//
  /**
   * @brief Build a parquet_scan_data per partition in @p metadata and push them into
   *        the bound split_connector. Invoked from metadata_scan.sink() on upstream
   *        worker threads.
   *
   * @param metadata The metadata scan output.
   */
  void accumulate_metadata(const partitioned_parquet_metadata& metadata);

  /**
   * @brief Close the bound split_connector, signaling that no more splits will arrive.
   *
   * Must be called exactly once, after all accumulate_metadata() calls have returned.
   * Invoked from metadata_scan.finalize_operator().
   */
  void finalize_partitions();

  /**
   * @brief Install a hive-partition injection function.
   *
   * Called once by the paired metadata scan operator at construction time when the scan
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
  std::unique_ptr<sirius_parquet_metadata_scan_operator> _metadata_scan_op;
};

}  // namespace sirius::op::scan
