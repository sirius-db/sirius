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

#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"

#include <atomic>
#include <cstdint>
#include <optional>
#include <string_view>

namespace sirius::op {

/**
 * @brief Carries dense-count input batches split by normalized join side
 *
 * `pipelineable_operator_data` owns one flattened batch sequence. This subclass appends all
 * preserved-side batches followed by all counted-side batches, so the two side counts describe
 * every position: batch `i` belongs to the preserved input exactly when `i < preserved_count()`.
 *
 * The batches are one partition of each side. A partition index is carried only when the operator
 * produced more than one partition: an indexed task is pinned to `_active_gpu_ids[idx % size]` so
 * every task of a partition shares a device, while a single-partition task has no such constraint
 * and is better placed by data affinity.
 */
class dense_count_join_input : public partitioned_operator_data {
 public:
  /**
   * @brief Combine the two normalized inputs into one batch container
   *
   * Either side may be empty. Every supplied batch pointer must be non-null.
   *
   * @param preserved_batches Batches from the outer-join-preserved input
   * @param counted_batches Batches from the input whose matches contribute to COUNT
   */
  dense_count_join_input(std::vector<std::shared_ptr<::cucascade::data_batch>> preserved_batches,
                         std::vector<std::shared_ptr<::cucascade::data_batch>> counted_batches);

  /**
   * @brief As above, tagging the batches with the partition they were drained from
   *
   * @param preserved_batches Batches from the outer-join-preserved input
   * @param counted_batches Batches from the input whose matches contribute to COUNT
   * @param partition_idx Partition these batches came from; pins the task to one GPU
   */
  dense_count_join_input(std::vector<std::shared_ptr<::cucascade::data_batch>> preserved_batches,
                         std::vector<std::shared_ptr<::cucascade::data_batch>> counted_batches,
                         std::size_t partition_idx);

  /** @brief Number of leading batches that belong to the preserved input. */
  [[nodiscard]] std::size_t preserved_count() const noexcept { return _preserved_count; }

  /** @brief Number of trailing batches that belong to the counted input. */
  [[nodiscard]] std::size_t counted_count() const noexcept { return _counted_count; }

 private:
  std::size_t _preserved_count;
  std::size_t _counted_count;
};

/**
 * @brief Execute a grouped COUNT over an outer equi-join without materializing joined rows
 *
 * The planner normalizes LEFT and RIGHT joins so `children[0]` is the preserved, grouped input and
 * `children[1]` is the counted input. Counted-side NULL keys never match, and preserved-side NULL
 * keys form one group; both strategies derive the emitted COUNT value from the rule stated on
 * `sirius::op::dense_count_semantics`.
 *
 * Both inputs use FULL barriers and are hash-partitioned on the join key, and one task drains one
 * partition of each side. That partitioning co-locates equal keys, including NULL keys, so a task
 * sees every row of every key it sees and the per-task outputs concatenate without a merge step.
 *
 * Runtime uses direct-address histograms when the observed key domain satisfies its layout,
 * density, and memory gates; otherwise it uses exact sparse aggregation. Both paths emit
 * `[key, BIGINT count]`.
 */
class sirius_physical_dense_count_join : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::DENSE_COUNT_JOIN;

  static constexpr std::string_view PRESERVED_PORT = "preserved";
  static constexpr std::string_view COUNTED_PORT   = "counted";

  /**
   * @brief Configure the normalized join inputs and runtime histogram budget
   *
   * @param types Output schema `[group key, BIGINT count]`
   * @param estimated_cardinality Estimated number of output groups
   * @param preserved_key_idx Join-key column within the preserved child's output
   * @param counted_key_idx Join-key column within the counted child's output
   * @param counted_value_idx COUNT(col) argument within the counted child's output, or
   *        `std::nullopt` for COUNT(*)
   * @param max_bins_bytes Budget for one partition task's direct-address histograms; the runtime
   *        density and input-size gates may still select sparse aggregation below it
   * @param hash_partition_bytes Configured per-partition byte target; get_partition_strategy
   *        doubles it because one task of this operator holds a partition of both inputs
   */
  sirius_physical_dense_count_join(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::size_t preserved_key_idx,
    std::size_t counted_key_idx,
    std::optional<std::size_t> counted_value_idx,
    uint64_t max_bins_bytes,
    uint64_t hash_partition_bytes = config::DEFAULT_HASH_PARTITION_BYTES);

  std::string params_to_string() const override;

  /**
   * @brief Map a direct child to its normalized preserved or counted input port
   *
   * @throw sirius::internal_exception If @p producer is not one of this operator's children
   *
   * @param producer Direct child whose destination port is requested
   * @return `PRESERVED_PORT` for child zero or `COUNTED_PORT` for child one
   */
  [[nodiscard]] std::string_view input_port_for(
    sirius_physical_operator const& producer) const override;

  /**
   * @brief Require the producer to finish before its batches are consumed
   *
   * @param producer Direct preserved or counted child
   * @return `MemoryBarrierType::FULL`
   */
  [[nodiscard]] MemoryBarrierType input_barrier_for(
    sirius_physical_operator const& producer) const override;

  /**
   * @brief Aggregate one partition of each input into one `[key, BIGINT count]` batch
   *
   * Selects the dense or sparse strategy from the materialized inputs and records it in
   * `last_strategy()`.
   *
   * @param input_data A prepared `dense_count_join_input`
   * @param stream CUDA stream used for aggregation and output materialization
   * @return Pipelineable data containing one output batch
   */
  std::unique_ptr<operator_data> execute(operator_data const& input_data,
                                         rmm::cuda_stream_view stream) override;

  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }

  /**
   * @brief Build one FULL-barrier producer pipeline per input
   *
   * This operator is a sink parent: each direct child reports `is_sink()` and terminates its own
   * pipeline through the generic per-operator protocol, so any subtree that plans to a
   * repository-fed root (scans, streaming chains, joins, aggregates, sorts) can feed either input.
   * The counted input is built first.
   *
   * @param current Pipeline that consumes the fused result
   * @param meta_pipeline Meta-pipeline that hosts this operator's pipeline and both producers
   */
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  /**
   * @brief Wait for both producers to finish before scheduling any fused task
   *
   * Unlike the base implementation, this permits either input port to be empty so an empty counted
   * side can still produce the preserved groups.
   *
   * @return A wait hint while a producer is active, a ready hint when queued input can be drained,
   *         or `std::nullopt` when no task remains
   */
  std::optional<task_creation_hint> get_next_task_hint() override;

  /**
   * @brief Drain the next partition of both inputs into one preserved-then-counted task input
   *
   * Partitions holding no batches on either side are skipped rather than ending task creation.
   *
   * @return The partition's batches as `dense_count_join_input`, or `nullptr` once every
   *         partition has been handed out
   */
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  /**
   * @brief Decide how many partitions the upstream PARTITIONs should produce
   *
   * Sizes from both inputs combined, since one task holds a partition of each side, and uses twice
   * the configured `hash_partition_bytes` as both the per-partition target and the small-table
   * threshold. Never broadcasts and never drives BUILD_PROBE.
   *
   * @param in What the sizing partition measured, including both siblings' byte totals
   * @return The partition count to apply to both input PARTITIONs
   */
  [[nodiscard]] partition_strategy get_partition_strategy(
    const partition_sizing_input& in) override;

  /**
   * @brief Estimate a conservative first-run peak across all execution strategies
   *
   * Arithmetic saturates rather than wrapping when the input bounds are too large to represent.
   *
   * @param stats Number and total logical size of the input batches
   * @return Maximum estimated peak for extrema reduction, dense histograms, and sparse aggregation
   */
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  [[nodiscard]] std::size_t preserved_key_idx() const noexcept { return _preserved_key_idx; }
  [[nodiscard]] std::size_t counted_key_idx() const noexcept { return _counted_key_idx; }
  [[nodiscard]] std::optional<std::size_t> counted_value_idx() const noexcept
  {
    return _counted_value_idx;
  }
  [[nodiscard]] uint64_t max_bins_bytes() const noexcept { return _max_bins_bytes; }

  /** @brief Runtime implementation selected by an `execute()` call. */
  enum class strategy : uint8_t {
    NOT_RUN,  ///< No strategy has yet been selected
    DENSE,    ///< Direct-address path or the no-non-NULL-key shortcut
    SPARSE    ///< Exact groupby-and-join path
  };

  /**
   * @brief Return the strategy chosen by whichever `execute()` call completed most recently
   *
   * Diagnostic only. Partition tasks run concurrently and each picks its own strategy, so this
   * reports one arbitrary task's choice whenever more than one partition exists.
   */
  [[nodiscard]] strategy last_strategy() const noexcept
  {
    return _last_strategy.load(std::memory_order_relaxed);
  }

 private:
  /// Number of partitions decided by get_partition_strategy, mirrored from the input repositories.
  /// One until sizing runs.
  [[nodiscard]] std::size_t partition_count() const;

  std::size_t _preserved_key_idx;
  std::size_t _counted_key_idx;
  std::optional<std::size_t> _counted_value_idx;
  uint64_t _max_bins_bytes;
  /// Written by every task, read by nothing the operator decides on; see last_strategy().
  std::atomic<strategy> _last_strategy{strategy::NOT_RUN};
  /// Next partition to hand to a task; guarded by the base `lock`.
  std::size_t _current_partition_index = 0;
  /// Partition of the task that carried NULL preserved keys, once one has; a *different* partition
  /// carrying them means the co-location contract above broke. The inner optional is empty in the
  /// single-partition case, where the input is deliberately left untagged. Retrying the same
  /// partition after an OOM compares equal and is allowed. Guarded by the base `lock`.
  std::optional<std::optional<std::size_t>> _null_group_partition;
};

}  // namespace sirius::op
