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

#include "op/cross_rank_sort_boundaries.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"
#include "sirius_config.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

namespace sirius {
namespace op {

/**
 * @brief What SORT_SAMPLE derives the number of range partitions from, besides the data size
 *
 * Set by the plan generator; the defaults describe an operator built without an engine context.
 */
struct sort_partition_policy {
  //! false: the previous rule, data bytes over max_sort_partition_memory_fraction of the GPU
  //! memory free when the boundaries are computed (SET sort_partition_auto_sizing = false)
  bool auto_sizing = true;
  //! Merge tasks one GPU runs at once: the executor threads per GPU (0 = unknown)
  std::size_t merge_concurrency = 0;
  //! GPUs the query runs on
  std::size_t num_gpus = 1;
  //! Every partition takes the one-shot merge (sort_merge_mode = 'one_shot'), which holds a whole
  //! partition, its merge temporaries and its output on one GPU
  bool one_shot_only = false;
  //! Every partition takes the windowed merge (sort_merge_mode = 'windowed')
  bool windowed_only = false;
  //! In 'auto' mode, the largest partition (bytes) that takes the one-shot merge; larger ones are
  //! merged in windows (0 = unknown). Set when neither flag above is.
  std::size_t auto_one_shot_partition_bytes = 0;
};

//! Inputs of compute_sort_partition_sizing()
struct sort_partition_sizing_input {
  std::size_t data_bytes          = 0;  ///< estimated bytes of the whole sort input
  std::size_t data_rows           = 0;  ///< estimated rows of the whole sort input (0 = unknown)
  std::size_t avg_batch_bytes     = 0;  ///< average bytes of one sorted input batch (0 = unknown)
  std::size_t gpu_capacity_bytes  = 0;  ///< memory limit of one GPU
  std::size_t gpu_available_bytes = 0;  ///< GPU memory free right now (previous rule only)
  std::size_t max_partition_bytes_override = 0;  ///< max_sort_partition_bytes (0 = automatic)
  double memory_fraction                   = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION;
  sort_partition_policy policy;
};

//! Result of compute_sort_partition_sizing()
struct sort_partition_sizing {
  std::size_t num_partitions      = 1;
  std::size_t max_partition_bytes = 0;         ///< the byte bound applied (0 = none)
  const char* rule                = "single";  ///< the term that decided the count, for the log
};

//! Share of a GPU's memory limit per partition when the sort does not fit the GPUs (the data and
//! its sorted copy exceed their limits, so runs are spilled while the sort is still producing
//! them). Smaller partitions then keep more of each merge on the GPU: with the limit at half the
//! data, a tenth of the limit per partition merged faster than a third or a fifth.
inline constexpr double kSortPartitionSpillFraction = 0.10;

//! How far above the one-shot threshold of 'auto' mode partitions are kept when they are meant to
//! be merged in windows: range partitions are not equal in size and the data size is an estimate,
//! so partitions sized just above the threshold would partly fall under it.
inline constexpr double kSortPartitionWindowedRouteMargin = 1.2;

//! Smallest partition the concurrency term creates: it never cuts the data into partitions
//! smaller than this just to give every merge thread one.
inline constexpr std::size_t kSortPartitionMinConcurrencyBytes = std::size_t{2} << 30;

//! Smallest piece (input batch bytes / partitions) the concurrency term accepts: below it every
//! batch is cut into pieces whose per-piece handling costs more than the merge gains.
inline constexpr std::size_t kSortPartitionMinPieceBytes = std::size_t{4} << 20;

//! Most partitions the concurrency term asks for: more than one per executor thread of a GPU
//! (4 by default) did not merge faster on 2 or 4 GPUs.
inline constexpr std::size_t kSortPartitionMaxConcurrency = 4;

//! Rows per partition the automatic sizing aims below: three quarters of the cuDF column limit
//! (2^31 - 1 rows), leaving room for partitions that come out larger than the sample predicts.
inline constexpr std::size_t kSortPartitionMaxRows = (std::size_t{1} << 31) / 4 * 3;

//! One-shot merges the automatic sizing leaves room for on one GPU at the same time
inline constexpr std::size_t kSortPartitionOneShotConcurrentMerges = 2;

/**
 * @brief Number of range partitions for a sort of the given size
 *
 * max_sort_partition_bytes, when set, decides alone: ceil(data / override). Otherwise, with
 * policy.auto_sizing off, the previous rule applies: ceil(data / (fraction x free GPU memory)).
 * With auto sizing the count is the largest of
 *  - memory: ceil(data / (fraction x GPU memory limit)) while the data and its sorted copy fit the
 *    GPUs (2 x data <= limit x policy.num_gpus); beyond that, ceil(data /
 *    (kSortPartitionSpillFraction x limit)). Both use the limit, not the free memory of one
 *    moment, so the count does not change from run to run;
 *  - one-shot (policy.one_shot_only): a one-shot merge needs 3 x partition bytes + 8 bytes per row
 *    on the GPU, so partitions are at most limit / ((3 + 8 / row width) x 2 concurrent merges);
 *  - rows, where partitions take the one-shot merge (policy.one_shot_only, or 'auto' mode with
 *    partitions at or below policy.auto_one_shot_partition_bytes): ceil(rows /
 *    kSortPartitionMaxRows), because a one-shot merge makes one cuDF table of the partition;
 *  - concurrency (policy.windowed_only on more than one GPU): one partition per merge task a GPU
 *    runs at once (min(policy.merge_concurrency, kSortPartitionMaxConcurrency)), but no partition
 *    smaller than kSortPartitionMinConcurrencyBytes and no piece of an input batch smaller than
 *    kSortPartitionMinPieceBytes. On one GPU more partitions than the memory term gives do not
 *    merge faster.
 *
 * In 'auto' mode the count is then lowered, if needed, so that the partitions do not all fall
 * under the one-shot threshold when their one-shot merges cannot run at once: several partitions
 * at or below policy.auto_one_shot_partition_bytes are only kept when
 * min(partitions, policy.merge_concurrency) one-shot merges fit the GPU limit together; otherwise
 * the count becomes the largest one whose partitions are at least
 * kSortPartitionWindowedRouteMargin times the threshold, and so are merged in windows.
 */
[[nodiscard]] sort_partition_sizing compute_sort_partition_sizing(
  const sort_partition_sizing_input& input);

class sirius_physical_sort_sample : public sirius_physical_operator {
 public:
  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }

  enum class BoundaryState : uint8_t { NOT_DONE, SCHEDULED, DONE };

  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::SORT_SAMPLE;

  sirius_physical_sort_sample(
    sirius_physical_order* order_by,
    uint64_t sort_sample_bytes           = config::DEFAULT_SORT_SAMPLE_BYTES,
    uint64_t max_partition_bytes         = 0,
    double max_partition_memory_fraction = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION);

  sirius_physical_sort_sample(
    duckdb::vector<sirius::logical_type> types,
    duckdb::vector<duckdb::BoundOrderByNode> orders,
    std::size_t estimated_cardinality,
    uint64_t sort_sample_bytes           = config::DEFAULT_SORT_SAMPLE_BYTES,
    uint64_t max_partition_bytes         = 0,
    double max_partition_memory_fraction = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION);

  ~sirius_physical_sort_sample() override;

  //! Order specification (copied from ORDER_BY) — determines which columns to sample
  duckdb::vector<duckdb::BoundOrderByNode> orders;

  //! How the partition count is chosen; set by the plan generator before execution
  void set_partition_policy(const sort_partition_policy& policy) { _partition_policy = policy; }
  [[nodiscard]] const sort_partition_policy& partition_policy() const { return _partition_policy; }

  //! Target bytes to accumulate for the boundary sample.
  [[nodiscard]] uint64_t sort_sample_bytes() const noexcept { return _sort_sample_bytes; }
  //! Max partition bytes, or 0 to size partitions from the GPU memory fraction.
  [[nodiscard]] size_t max_partition_bytes_override() const noexcept
  {
    return _max_partition_bytes_override;
  }
  //! Fraction of a GPU's memory limit per partition.
  [[nodiscard]] double max_partition_memory_fraction() const noexcept
  {
    return _max_partition_memory_fraction;
  }

 public:
  bool is_source() const override { return true; }
  // SORT_SAMPLE is not a pipeline sink — it runs as an intermediate operator in the
  // SORT_PARTITION pipeline, so the tree-based build matches the legacy 3-pipeline shape.
  bool is_sink() const override { return false; }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  //! Override to wait for enough sample bytes before returning READY
  std::optional<task_creation_hint> get_next_task_hint() override;

  //! Override to pull sample bytes before boundary computation, then one batch at a time
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  /**
   * @brief The partition boundaries (P-1 rows, sort key columns only) as a table on the caller's
   * GPU
   *
   * The boundaries are computed once, on the GPU that ran the boundary task (from this operator's
   * sample, or by the cross-rank hook from every process's sample), and kept in host memory. Each
   * GPU that partitions batches gets its own device copy, uploaded on first use and complete (its
   * stream synchronised) before any caller sees it, so the returned view can be read by kernels on
   * @p space 's device from any stream. The view stays valid until clear_partition_boundaries().
   *
   * Must be called with @p space 's device current, after boundaries_computed() returned true and
   * only when get_num_partitions() > 1.
   *
   * @param space  GPU memory space of the batch being partitioned; the copy is allocated from it
   * @param stream Stream used for the one-time upload
   */
  cudf::table_view get_partition_boundaries(cucascade::memory::memory_space& space,
                                            ::cuda::stream_ref stream);

  //! Number of boundary rows (get_num_partitions() - 1, or 0 for a single partition)
  size_t num_boundary_rows() const { return _host_boundaries ? _host_boundaries->num_rows : 0; }

  //! Get the computed number of partitions
  size_t get_num_partitions() const { return _num_partitions; }

  //! Whether boundaries have been computed
  bool boundaries_computed() const
  {
    return _boundary_state.load(std::memory_order_acquire) == BoundaryState::DONE;
  }

  //! Release the host copy of the partition boundaries and every per-GPU copy
  void clear_partition_boundaries();

  //! Boundaries come from @p hook instead of this operator's sample alone. Set while planning.
  //! @throws std::logic_error once the boundary task has been claimed.
  void set_cross_rank_boundaries(std::shared_ptr<cross_rank_sort_boundaries> hook)
  {
    if (boundary_state() != BoundaryState::NOT_DONE) {
      throw std::logic_error(
        "[sirius_physical_sort_sample::set_cross_rank_boundaries] the plan cannot change during "
        "execution");
    }
    _cross_rank_boundaries = std::move(hook);
  }
  //! The hook set by set_cross_rank_boundaries(), or null when this operator's sample stands alone.
  [[nodiscard]] cross_rank_sort_boundaries* cross_rank_boundaries() const noexcept
  {
    return _cross_rank_boundaries.get();
  }

  //! With a hook the boundary task is still owed, input or not, until it has been claimed.
  [[nodiscard]] bool all_ports_empty() override;
  //! With a hook the boundary task runs with no input, since the other processes wait for it.
  //! get_next_task_input_data() has already claimed it (SCHEDULED) when the task creator asks.
  //! Bytes of sorted runs waiting to be partitioned at which the local sort producing them is
  //! given no further task (`input_backlogged()`); the range partitioning is then also scheduled
  //! ahead of that sort, so a run is cut as soon as it exists instead of waiting behind the sorts
  //! still queued. 0 (the default): plan order, and the producer is never held back.
  void set_input_backlog_bytes(std::size_t bytes) noexcept { _input_backlog_bytes = bytes; }
  [[nodiscard]] std::size_t input_backlog_bytes() const noexcept { return _input_backlog_bytes; }
  [[nodiscard]] bool runs_ahead_of_producer() const noexcept override
  {
    // Not where the boundaries come from every rank: there the runs wait for the other ranks.
    return _input_backlog_bytes != 0 && !_cross_rank_boundaries;
  }

  //! Whether `input_backlog_bytes()` of runs are waiting while the producer is still running:
  //! the parked runs plus, at their average size, the producer's unfinished tasks. Never before
  //! the boundaries exist, when the runs wait for the sample and not for a task, and never with
  //! cross-rank boundaries.
  [[nodiscard]] bool input_backlogged(std::string_view port_id) override;

  [[nodiscard]] bool runs_without_input() const override
  {
    return _cross_rank_boundaries != nullptr && boundary_state() != BoundaryState::DONE;
  }

  //! A device INT32 column holding @p indices.
  static std::unique_ptr<cudf::column> index_column(std::vector<int32_t> const& indices,
                                                    ::cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr);

  //! Rows i * rows / parts of @p sorted for i in [1, parts), offset by @p first: parts - 1
  //! evenly spaced boundaries.
  static std::unique_ptr<cudf::table> evenly_spaced(cudf::table_view sorted,
                                                    cudf::size_type first,
                                                    cudf::size_type rows,
                                                    std::size_t parts,
                                                    ::cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr);

 private:
  //! Evenly spaced boundaries for @p num_parts partitions from this operator's own sorted
  //! @p sample_keys, as a table on @p space 's GPU.
  sort_boundaries local_boundaries(cudf::table_view sample_keys,
                                   std::size_t num_parts,
                                   cucascade::memory::memory_space* space,
                                   ::cuda::stream_ref stream);

  /**
   * @brief Keep @p made as this operator's boundaries: a host copy, uploaded per GPU on demand
   *
   * @p made 's table (from local_boundaries() or the cross-rank hook) lives on the GPU of the
   * boundary task. It is copied to host memory; the copy is complete when this returns, so the
   * caller may release the table and publish DONE. Must be called with that GPU current.
   *
   * @param mr Device memory for the packed copy the download goes through
   */
  void store_boundaries(const sort_boundaries& made,
                        ::cuda::stream_ref stream,
                        rmm::device_async_resource_ref mr);

  [[nodiscard]] BoundaryState boundary_state() const noexcept
  {
    return _boundary_state.load(std::memory_order_acquire);
  }

  //! Host copy of the boundary rows in cudf::pack layout, written once before DONE is published
  struct host_boundaries {
    std::vector<uint8_t> metadata;
    std::vector<uint8_t> data;
    size_t num_rows = 0;
  };

  //! One GPU's copy of the boundary rows; `view` points into `data`
  struct device_boundaries {
    rmm::device_buffer data;
    cudf::table_view view;
  };

  //! Partition boundary rows (P-1 rows containing sort key column values), immutable once
  //! _boundary_state is DONE and until clear_partition_boundaries()
  std::unique_ptr<host_boundaries> _host_boundaries;

  //! Per-GPU copies keyed by device id, created lazily under _device_boundaries_mutex
  std::mutex _device_boundaries_mutex;
  std::map<int, std::unique_ptr<device_boundaries>> _device_boundaries;

  //! Number of partitions computed from the sample
  size_t _num_partitions = 1;

  //! Boundary computation lifecycle: NOT_DONE → SCHEDULED (input claimed) → DONE.
  //! get_next_task_input_data() transitions to SCHEDULED under the task-creation lock;
  //! execute() completes the transition to DONE. On OOM the state stays SCHEDULED so the
  //! rescheduled task retries while task_creator does not spawn a duplicate boundary task.
  std::atomic<BoundaryState> _boundary_state{BoundaryState::NOT_DONE};

  //! Target bytes to accumulate for the boundary sample (from sirius_config operator_params)
  uint64_t _sort_sample_bytes = config::DEFAULT_SORT_SAMPLE_BYTES;

  //! Override for max partition bytes (0 = use GPU memory fraction)
  size_t _max_partition_bytes_override = 0;

  //! Fraction of a GPU's memory limit per partition (from sirius_config operator_params)
  double _max_partition_memory_fraction = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION;

  //! Concurrency, merge route and rule selection for the partition count
  sort_partition_policy _partition_policy;

  //! See set_cross_rank_boundaries().
  std::shared_ptr<cross_rank_sort_boundaries> _cross_rank_boundaries;
  std::size_t _input_backlog_bytes = 0;

  //! With a backlog limit, asks the task creator for this operator's next task: the request
  //! that restarts a producer held back by `input_backlogged()`.
  void schedule_next_task();

  //! What the hook returned, until the host copy is made: the hook takes part only once, so a
  //! boundary task retried after that (an OOM while copying) must not ask it again. Only one
  //! boundary task runs at a time.
  std::optional<sort_boundaries> _pending_cross_rank_boundaries;
};

}  // namespace op
}  // namespace sirius
