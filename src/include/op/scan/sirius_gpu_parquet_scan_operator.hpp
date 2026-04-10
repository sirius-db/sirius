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
#include <op/scan/parquet_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// standard library
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
 *      partitioned_parquet_metadata objects that are delivered here via sink().
 *    - The operator accumulates the partitioned_parquet_metadata objects.
 *
 *  Pipeline 2 (GPU parquet scan):
 *    - Acts as the source.
 *    - Once finalize_metadata() has been called (after all sink() calls complete),
 *      a flat partition index is built and get_next_task_input_data() can atomically
 *      claim partitions.  Each partition maps 1:1 to a row_group_range — the metadata
 *      scan operator is responsible for sizing partitions to the target batch size.
 *    - execute(parquet_scan_data) reads the actual byte ranges from disk.
 *
 * Thread-safety:
 *   - sink() is called from pipeline 1 worker threads and protects _accumulated_metadata
 *     with _metadata_mutex.
 *   - finalize_metadata() must be called exactly once, after ALL sink() calls have
 *     completed (i.e., after pipeline 1 has fully finished).  It builds the partition
 *     index and sets _finalized, after which all source-side methods become usable.
 *   - get_next_task_input_data() / get_next_task_hint() / all_ports_empty() are called
 *     from pipeline 2 worker threads and only proceed after _finalized is set.
 */
class sirius_gpu_parquet_scan_operator : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::GPU_PARQUET_SCAN;

  //===----------Constructor----------===//
  /**
   * @param types                  Output column types (forwarded from the parquet scan operator).
   * @param estimated_cardinality  Estimated row count.
   */
  sirius_gpu_parquet_scan_operator(duckdb::vector<duckdb::LogicalType> types,
                                   duckdb::idx_t estimated_cardinality);

  //===----------Sink interface (pipeline 1)----------===//
  bool is_sink() const override { return true; }

  /**
   * @brief Receive partitioned_parquet_metadata produced by pipeline 1 and accumulate it.
   *
   * @param input_data  Should be a partitioned_parquet_metadata instance.
   * @param stream      Unused; metadata handling is CPU-only.
   */
  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

  //===----------Pipeline 1 → Pipeline 2 transition----------===//
  /**
   * @brief Signal that pipeline 1 has fully finished, freeze metadata, and build partition index.
   *
   * Must be called exactly once after ALL sink() calls have completed.  After this call,
   * the source-side methods (get_next_task_hint, all_ports_empty, get_next_task_input_data)
   * become usable.
   */
  void finalize_metadata();

  //===----------Source interface (pipeline 2)----------===//
  bool is_source() const override { return true; }

  /**
   * @brief Returns READY while there are unconsumed partitions, or nullopt when all
   *        partitions have been dispatched or metadata has not yet been finalized.
   */
  std::optional<task_creation_hint> get_next_task_hint() override;

  /**
   * @brief Returns true once all partitions have been dispatched.
   *
   * Returns false if metadata has not yet been finalized.
   */
  [[nodiscard]] bool all_ports_empty() override;

  /**
   * @brief Claims and returns the next parquet_scan_data for a single row_group_range.
   *
   * Returns nullptr when all partitions have been consumed or metadata is not yet finalized.
   */
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //===----------Execution (pipeline 2)----------===//
  /**
   * @brief Read the byte ranges described by @p input_data from disk and produce a
   *        host_parquet_representation data batch.
   *
   * @param input_data  Must be a parquet_scan_data instance.
   * @param stream      CUDA stream (passed through for consistency; I/O is CPU-side).
   */
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //===----------Accessors----------===//
  /**
   * @brief Return the total number of partitions (tasks) that will be created.
   *
   * Returns an estimate from accumulated partition counts before finalization,
   * and the exact count after finalize_metadata().
   */
  [[nodiscard]] std::size_t get_total_partitions() const;

 private:
  // ===----------------------------------------------------------------------===//
  // Accumulation phase (pipeline 1).
  //   _metadata_mutex protects _accumulated_metadata during concurrent sink() calls.
  // ===----------------------------------------------------------------------===//
  mutable std::mutex _metadata_mutex;
  std::vector<partitioned_parquet_metadata> _accumulated_metadata;

  // ===----------------------------------------------------------------------===//
  // Partition index — built once by finalize_metadata(), then read-only.
  //
  //   _finalized          — set by finalize_metadata() with release semantics after
  //                          the partition index is fully written.  Source-side methods
  //                          check this with acquire semantics before accessing the index.
  //   _partition_index    — flat list pairing each partition with its parent metadata.
  //   _next_partition_idx — atomic counter; each fetch_add(1) claims one partition.
  // ===----------------------------------------------------------------------===//
  std::atomic<bool> _finalized{false};

  struct partition_entry {
    partitioned_parquet_metadata const* metadata;
    std::size_t partition_idx;
  };
  std::vector<partition_entry> _partition_index;
  std::atomic<std::size_t> _next_partition_idx{0};
};

}  // namespace sirius::op::scan
