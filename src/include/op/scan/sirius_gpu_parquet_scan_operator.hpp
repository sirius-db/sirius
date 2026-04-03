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

#include "op/scan/parquet_scan_operator_data.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Operator that reads parquet byte ranges for a batch of row groups and produces
 *        host_parquet_representation data batches for downstream GPU operators.
 *
 * This operator plays two roles in the two-pipeline parquet execution model:
 *
 *  Pipeline 1 (metadata scan):
 *    - Acts as the sink.  sirius_parquet_metadata_scan_operator::execute() produces
 *      partitioned_parquet_metadata objects (wrapped in partitioned_metadata_operator_data)
 *      that are delivered here via sink().
 *    - The operator accumulates the shared_ptr<partitioned_parquet_metadata> objects.
 *
 *  Pipeline 2 (GPU parquet scan):
 *    - Acts as the source.
 *    - Once mark_pipeline1_finished() has been called, build_partition_list() freezes
 *      the accumulated metadata into a pre-built list of parquet_scan_data batches.
 *    - get_next_task_input_data() atomically claims the next pre-built batch.
 *    - execute(parquet_scan_data) reads the actual byte ranges from disk.
 *
 * Thread-safety:
 *   - sink() is called from pipeline 1 worker threads and protects _accumulated_metadata
 *     with _metadata_mutex.
 *   - mark_pipeline1_finished() must be called after ALL sink() calls have completed
 *     (i.e., after pipeline 1 has fully finished).
 *   - build_partition_list() is guarded by _pipeline1_finished and uses a double-checked
 *     lock on _partitions_built (std::atomic<bool>) for safe one-time initialization.
 *   - get_next_task_input_data() / get_next_task_hint() / all_ports_empty() are called
 *     from pipeline 2 worker threads and only proceed after the partition list is frozen.
 */
class sirius_gpu_parquet_scan_operator : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::GPU_PARQUET_SCAN;

  /// Default target uncompressed bytes per GPU scan task.
  static constexpr size_t DEFAULT_BATCH_DATA_SIZE = 512ULL * 1024 * 1024;  // 512 MB

  //===----------Constructor----------===//
  /**
   * @param types          Output column types (forwarded from the parquet scan operator).
   * @param estimated_cardinality  Estimated row count.
   * @param batch_data_size  Target uncompressed bytes per scan task.
   */
  sirius_gpu_parquet_scan_operator(duckdb::vector<duckdb::LogicalType> types,
                                   duckdb::idx_t estimated_cardinality,
                                   size_t batch_data_size = DEFAULT_BATCH_DATA_SIZE);

  //===----------Sink interface (pipeline 1)----------===//
  bool is_sink() const override { return true; }

  /**
   * @brief Receive partitioned_parquet_metadata produced by pipeline 1 and accumulate it.
   *
   * Expects @p input_data to be a partitioned_metadata_operator_data instance.
   * Non-matching types are logged as a warning and silently ignored.
   *
   * @param input_data  Should be a partitioned_metadata_operator_data instance.
   * @param stream      Unused; metadata handling is CPU-only.
   */
  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

  //===----------Pipeline 1 completion signal----------===//
  /**
   * @brief Signal that pipeline 1 has fully finished and no more sink() calls will arrive.
   *
   * @pre  All sink() calls for pipeline 1 must have completed before this is called.
   * @post build_partition_list() will proceed on the next call to get_next_task_hint(),
   *       all_ports_empty(), or get_next_task_input_data().
   */
  void mark_pipeline1_finished();

  //===----------Source interface (pipeline 2)----------===//
  bool is_source() const override { return true; }

  /**
   * @brief Returns READY (pointing to itself) while there are unconsumed pre-built batches,
   *        or nullopt when all batches have been dispatched or pipeline 1 has not yet finished.
   *
   * @note Calling this method has the side effect of invoking build_partition_list() once
   *       _pipeline1_finished is set, which permanently freezes the internal partition list.
   */
  std::optional<task_creation_hint> get_next_task_hint() override;

  /**
   * @brief Returns true once all pre-built batches have been dispatched.
   *
   * @pre  Pipeline 1 must have finished (mark_pipeline1_finished() called) before this
   *       can return true; otherwise returns false (no batches available yet).
   * @post Returns true iff all row-group batches have been dispatched via
   *       get_next_task_input_data().
   * @note Calling this method has the side effect of invoking build_partition_list() once
   *       _pipeline1_finished is set, which permanently freezes the internal partition list.
   */
  [[nodiscard]] bool all_ports_empty() override;

  /**
   * @brief Claims and returns the next pre-built parquet_scan_data batch.
   *
   * Returns nullptr when all batches have been consumed or when the partition list is
   * not yet built (pipeline 1 not finished).
   *
   * @note NOT YET IMPLEMENTED end-to-end: execute() always throws until
   *       parquet_gpu_scan_task integration is complete.
   */
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //===----------Execution (pipeline 2)----------===//
  /**
   * @brief Read the byte ranges described by @p input_data from disk and produce a
   *        host_parquet_representation data batch.
   *
   * @param input_data  Must be a parquet_scan_data instance.
   * @param stream      CUDA stream (passed through for consistency; I/O is CPU-side).
   *
   * @note NOT YET IMPLEMENTED. This method always throws std::runtime_error until the
   *       parquet_gpu_scan_task integration is complete. Do not wire this operator into
   *       a live pipeline until the TODO in the .cpp is resolved.
   */
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //===----------Accessors----------===//
  [[nodiscard]] size_t get_batch_data_size() const { return _batch_data_size; }

  /**
   * @brief Return the total number of pre-built scan batches (tasks) that will be created.
   *
   * Returns an estimate from accumulated partition counts before the partition list is built,
   * and the exact batch count after mark_pipeline1_finished() + build_partition_list().
   */
  [[nodiscard]] size_t get_total_partitions() const;

 private:
  size_t _batch_data_size;

  // -------------------------------------------------------------------
  // Members protected by _metadata_mutex:
  //   _accumulated_metadata  — written by sink(), read by build_partition_list()
  // -------------------------------------------------------------------
  mutable std::mutex _metadata_mutex;
  std::vector<partitioned_parquet_metadata> _accumulated_metadata;

  // -------------------------------------------------------------------
  // Partition list — written once by build_partition_list(), then read-only.
  //
  //   _pipeline1_finished  — set by mark_pipeline1_finished() after all sink() calls
  //                          complete. Guards build_partition_list() to prevent premature
  //                          freezing before all metadata has arrived.
  //   _partitions_built    — std::atomic<bool> used for double-checked locking inside
  //                          build_partition_list(). Use acquire/release semantics so
  //                          that all writes to _pre_built_batches are visible to any
  //                          thread that observes _partitions_built == true.
  //   _pre_built_batches   — frozen list of scan batches; one entry per GPU task.
  //   _next_batch_idx      — atomic index into _pre_built_batches; each fetch_add(1)
  //                          claims exactly one batch for one task.
  // -------------------------------------------------------------------
  std::atomic<bool> _pipeline1_finished{false};
  std::atomic<bool> _partitions_built{false};

  struct pre_built_batch {
    const partitioned_parquet_metadata* metadata;
    std::vector<row_group_range> scan_ranges;
  };
  std::vector<pre_built_batch> _pre_built_batches;
  std::atomic<size_t> _next_batch_idx{0};

  /// Build _pre_built_batches from accumulated metadata (called once after pipeline 1 finishes).
  void build_partition_list();
};

}  // namespace sirius::op::scan
