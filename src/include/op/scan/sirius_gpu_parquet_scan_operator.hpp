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
 *        gpu_table_representation data batches for downstream GPU operators.
 *
 * This operator is the source of the GPU parquet scan pipeline. It is paired with a
 * sirius_parquet_metadata_scan_operator, which runs as a separate upstream pipeline and
 * pushes parsed metadata directly into this operator — not through the data-batch / port
 * channel, since partitioned_parquet_metadata is not a data_batch.
 *
 * Lifecycle:
 *
 *   1. Metadata accumulation (upstream pipeline, CPU-only):
 *      - sirius_parquet_metadata_scan_operator::execute() parses parquet footers and
 *        produces partitioned_parquet_metadata.
 *      - Its sink() forwards each result here via accumulate_metadata().
 *      - When the upstream pipeline finishes, its finalize_operator() calls
 *        finalize_partitions() on this operator, which freezes the accumulated metadata
 *        into a flat partition index and releases _finalized.
 *
 *   2. Scan (this pipeline, GPU):
 *      - get_next_task_hint() / all_ports_empty() block on _finalized so this pipeline
 *        cannot begin until accumulation is complete.
 *      - get_next_task_input_data() atomically claims one partition from the index and
 *        returns it as parquet_scan_data. Each partition maps 1:1 to a row_group_range —
 *        the metadata scan operator is responsible for sizing partitions to the target
 *        batch size.
 *      - execute(parquet_scan_data) reads the byte ranges, optionally applies a
 *        fallback filter expression, and emits a gpu_table_representation data batch.
 *
 * Scheduling coupling:
 *   The upstream → downstream pipeline edge is expressed via a null-repo
 *   "dependency" port on this operator. setup_pipeline_parents() uses that port
 *   to discover the dependency so this pipeline is not scheduled until the metadata
 *   pipeline completes. No data flows through the port.
 *
 * Thread-safety:
 *   - accumulate_metadata() is called from upstream worker threads; _metadata_mutex
 *     protects _accumulated_metadata.
 *   - finalize_partitions() must be called exactly once, after ALL accumulate_metadata()
 *     calls have completed. It builds the partition index and sets _finalized with
 *     release semantics; source-side methods read _finalized with acquire semantics
 *     before accessing the index.
 *   - get_next_task_input_data() / get_next_task_hint() / all_ports_empty() are safe to
 *     call from multiple worker threads and return no-op results until _finalized.
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

  //===----------Metadata handoff (called by metadata_scan)----------===//
  /**
   * @brief Accumulate one partitioned_parquet_metadata produced by the metadata scan pipeline.
   *
   * Invoked from metadata_scan.sink().
   *
   * @param metadata The metadata scan output.
   */
  void accumulate_metadata(const partitioned_parquet_metadata& metadata);

  /**
   * @brief Freeze metadata and build the partition index.
   *
   * Must be called exactly once after ALL accumulate_metadata() calls complete. Invoked from
   * metadata_scan.finalize_operator().
   */
  void finalize_partitions();

  //===----------Source interface----------===//
  bool is_source() const override { return true; }

  /**
   * @return READY while there are unconsumed partitions; nullopt when all
   *         partitions have been dispatched or metadata has not yet been finalized.
   */
  std::optional<task_creation_hint> get_next_task_hint() override;

  /**
   * @return true once all partitions have been dispatched;
   *         false if metadata has not yet been finalized.
   */
  [[nodiscard]] bool all_ports_empty() override;

  /**
   * @brief Claims and returns the next parquet_scan_data for a single row_group_range.
   *
   * @return the claimed parquet_scan_data or nullptr when all partitions have been consumed or
   *         metadata is not yet finalized.
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

  //===----------Accessors----------===//
  /**
   * @brief Return the total number of partitions (i.e., tasks) that will be created.
   *
   * @pre finalize_partitions() has been called.
   * @return The total number of row group partitions (i.e., tasks).
   * @throws std::runtime_error if called before finalize_partitions().
   */
  [[nodiscard]] std::size_t get_total_partitions() const;

 private:
  // ===----------------------------------------------------------------------===//
  // Accumulation phase
  //   _metadata_mutex protects _accumulated_metadata during concurrent accumulate_metadata() calls.
  // ===----------------------------------------------------------------------===//
  std::mutex _metadata_mutex;
  std::vector<partitioned_parquet_metadata> _accumulated_metadata;

  // ===----------------------------------------------------------------------===//
  // Partition index — built once by finalize_partitions(), then read-only.
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
