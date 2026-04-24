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
#include <mutex>
#include <optional>
#include <vector>

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

  //===----------Metadata handoff (called by metadata_scan)----------===//
  /**
   * @brief Append partitions for one partitioned_parquet_metadata to the partition index.
   *
   * Invoked from metadata_scan.sink() on upstream worker threads. Copies @p metadata
   * into a shared_ptr so its lifetime is tied to the partition entries, then expands
   * each element of row_group_partitions into an individual partition_entry under
   * _metadata_mutex. Partitions become claimable by get_next_task_input_data() as
   * soon as they are appended.
   *
   * @param metadata The metadata scan output.
   */
  void accumulate_metadata(const partitioned_parquet_metadata& metadata);

  /**
   * @brief Signal that no more metadata will arrive.
   *
   * Must be called exactly once, after all accumulate_metadata() calls have returned.
   * Invoked from metadata_scan.finalize_operator(). Sets _finalized under
   * _metadata_mutex so get_next_task_hint() can return std::nullopt once the partition
   * index is fully drained instead of continuing to defer to the upstream pipeline.
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
   * @return READY pointing at this operator while _partition_index contains an
   *         unclaimed entry (regardless of whether accumulation is still in
   *         progress);
   *         WAITING_FOR_INPUT_DATA pointing at the upstream metadata scan when no
   *         entry is currently claimable and the metadata pipeline has not yet
   *         been finalized (surfaces the upstream dependency to
   *         task_creator::get_operator_for_next_task, which otherwise cannot
   *         discover it — the metadata handoff is a side channel, not a data
   *         repo);
   *         nullopt once all partitions have been claimed AND
   *         finalize_partitions() has been called.
   */
  std::optional<task_creation_hint> get_next_task_hint() override;

  /**
   * @return true when no partition is currently claimable (either all have been
   *         claimed or none have been accumulated yet); false while
   *         _partition_index contains an unclaimed entry. Callers combine this
   *         with is_source_pipeline_finished() to decide whether the scan
   *         pipeline is truly done.
   */
  [[nodiscard]] bool all_ports_empty() override;

  /**
   * @brief Claims and returns the next parquet_scan_data for a single row_group_range.
   *
   * @return the claimed parquet_scan_data, or nullptr when no partition is currently
   *         claimable (either all partitions have been claimed, or none have been
   *         accumulated yet). Returning nullptr does not imply the scan is finished —
   *         the caller must consult is_source_pipeline_finished() for that.
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
  // ===----------------------------------------------------------------------===//
  // Streaming partition index
  //   _metadata_mutex    — serializes accumulate_metadata() appends against
  //                         get_next_task_input_data() claims. Also covers size
  //                         reads by get_next_task_hint() / all_ports_empty() so
  //                         the observed size is consistent with ongoing appends.
  //   _partition_index   — grown incrementally by accumulate_metadata(); each entry
  //                         holds a shared_ptr to the metadata object it indexes
  //                         into (so the copy outlives the upstream sink's
  //                         operator_data) plus the partition offset within that
  //                         metadata's row_group_partitions.
  //   _next_partition_idx— counter of the next unclaimed entry; advanced under
  //                         _metadata_mutex by get_next_task_input_data().
  //   _finalized         — set once by finalize_partitions() under _metadata_mutex
  //                         to signal that no more accumulate_metadata() calls will
  //                         arrive. Read by get_next_task_hint() under _metadata_mutex
  //                         to decide nullopt vs. WAITING.
  // ===----------------------------------------------------------------------===//
  std::mutex _metadata_mutex;
  bool _finalized{false};
  partition_inject_fn_t _hive_partition_inject_fn;

  struct partition_entry {
    std::shared_ptr<partitioned_parquet_metadata> metadata;
    std::size_t partition_idx;  ///< Index into the associated metadata's partition list
  };
  std::vector<partition_entry> _partition_index;
  std::size_t _next_partition_idx{0};
};

}  // namespace sirius::op::scan
