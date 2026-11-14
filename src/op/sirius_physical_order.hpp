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

#include "duckdb/planner/bound_query_node.hpp"
#include "op/sirius_physical_operator.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

namespace sirius {
namespace op {

// Helper to deep copy BoundOrderByNode vector (contains unique_ptr<Expression>)
inline duckdb::vector<duckdb::BoundOrderByNode> copy_orders(
  const duckdb::vector<duckdb::BoundOrderByNode>& src)
{
  duckdb::vector<duckdb::BoundOrderByNode> result;
  result.reserve(src.size());
  for (const auto& order : src) {
    result.push_back(order.Copy());
  }
  return result;
}

/**
 * @brief Limits of one sort unit (one sorted run) of a coalescing ORDER_BY.
 *
 * A zero limit means "no limit of that kind".
 */
struct sort_run_limits {
  /// Decoded bytes of the batches sorted as one run. Planning closes a group before it would
  /// exceed this, so a run is at most this large unless a single batch already is.
  std::size_t max_run_bytes = 0;
  /// Rows of one run. cuDF addresses rows with a 32-bit signed index.
  std::size_t max_run_rows = static_cast<std::size_t>(std::numeric_limits<int32_t>::max());
  /// Character bytes of any one string column of a run. A cuDF string column with 32-bit offsets
  /// holds less than 2^31 bytes; concatenating past that would switch the column to 64-bit
  /// offsets, which the operators downstream of the sort are not written for.
  std::size_t max_string_column_bytes = static_cast<std::size_t>(std::numeric_limits<int32_t>::max());
};

/**
 * @brief What the local sorts of ORDER_BY operators did, for tests and the stats log line.
 */
struct order_by_counters {
  uint64_t tasks              = 0;  ///< execute() calls
  uint64_t input_batches      = 0;  ///< Input batches sorted
  uint64_t runs               = 0;  ///< Sorted runs produced
  uint64_t concatenated_bytes = 0;  ///< Bytes of the runs built by concatenating several batches
};

/**
 * @brief Local sort: sorts its input into runs for SORT_PARTITION and MERGE_SORT.
 *
 * Two modes:
 *
 * - **Per batch** (default, and the only mode of an operator built outside the planner). The
 *   operator is the sink of its producer's pipeline and sorts every input batch of a task on its
 *   own, so every input batch becomes one sorted run.
 * - **Coalescing** (`enable_run_coalescing()`, set by the planner when `sort_coalesce_runs` is on
 *   and the schema and the sort keys allow it, see `order_by_run_limits()` in the plan generator). The planner puts a PASSTHROUGH_SINK below the operator, so the
 *   operator starts its own pipeline and reads the producer's batches from an input port. A task
 *   then claims a group of batches of up to `sort_run_limits::max_run_bytes`, concatenates them
 *   and sorts the concatenation as ONE run. A range partition therefore receives one run per
 *   group instead of one per input batch, which is what decides how many passes MERGE_SORT needs
 *   (see docs/super-sirius/operators.md).
 *
 * Coalescing changes which rows share a run, never the rows or how a unit is sorted: every unit
 * still goes through `gpu_order_impl::local_order_by`.
 */
class sirius_physical_order : public sirius_physical_operator {
 public:
  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    // Local sorting does not establish a sequence across independently completed batches.
    return sirius::OrderPreservationType::NO_ORDER;
  }

  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::ORDER_BY;

  sirius_physical_order(duckdb::vector<sirius::logical_type> types,
                        duckdb::vector<duckdb::BoundOrderByNode> orders,
                        duckdb::vector<std::size_t> projections_p,
                        std::size_t estimated_cardinality,
                        bool is_index_sort_p = false);

  //! Input data
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  duckdb::vector<std::size_t> projections;
  bool is_index_sort;

  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  //! Switch the operator to coalescing mode. Call before the pipelines are built: the operator
  //! must have its own input port (a PASSTHROUGH_SINK child). @p limits.max_run_bytes must be
  //! positive.
  void enable_run_coalescing(sort_run_limits limits);

  //! Whether tasks sort groups of input batches as one run each.
  [[nodiscard]] bool run_coalescing_enabled() const noexcept { return _coalesce; }

  //! The limits of one run; meaningful only when `run_coalescing_enabled()`.
  [[nodiscard]] const sort_run_limits& run_limits() const noexcept { return _limits; }

  //! PARTIAL for the PASSTHROUGH_SINK feeding a coalescing operator, so groups are sorted while
  //! the producer is still running; the base rule otherwise.
  [[nodiscard]] MemoryBarrierType input_barrier_for(
    sirius_physical_operator const& producer) const override;

  //! Coalescing mode: READY when a group is complete (or the producer finished and batches
  //! remain), otherwise WAITING on the producer. Per-batch mode: the base rule.
  std::optional<task_creation_hint> get_next_task_hint() override;

  //! Coalescing mode: claims the next complete group under the same policy as the hint.
  //! Per-batch mode: the base rule (one batch per port).
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //! A coalesced task holds its inputs, their concatenation, the sort's scratch and the output.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(const input_stats& stats) const override;

  /**
   * @brief One parked input batch as group planning sees it.
   */
  struct parked_batch {
    uint64_t id = 0;                  ///< Batch id in the input repository
    std::size_t bytes = 0;            ///< Decoded bytes
    std::optional<std::size_t> rows;  ///< Row count, empty when the representation has none
    /// GPU device holding the batch, or -1 when it is not on a GPU (spilled, or being moved).
    int device = -1;
  };

  /**
   * @brief Chooses the batches of the next run from the parked batches, in repository order.
   *
   * Batches are grouped per GPU device so that a run is built from batches that are already on
   * the GPU that will sort them; batches that are not on a GPU form one more group. A group is
   * complete when the next batch of its device would take it past `max_run_bytes` or
   * `max_run_rows` (that batch then starts the next group), or when it reaches `max_run_bytes`.
   * A batch that is larger than the limit by itself is a group of one. When the schema has a
   * string column a group is also kept below `max_string_column_bytes` of decoded bytes, a safe
   * bound for every string column in it.
   *
   * When @p producer_finished is true nothing more will arrive: if no group is complete, the
   * remaining batches are grouped in repository order regardless of device, under the same
   * limits, so the tail costs as few runs as possible.
   *
   * Pure function of its arguments; exposed for tests.
   *
   * @param parked Parked batches in repository order.
   * @param limits Limits of one run.
   * @param has_string_column Whether any column of the schema is a string.
   * @param producer_finished Whether the producer pipeline has finished.
   * @return Indices into @p parked of the next group, empty when no group is ready.
   */
  [[nodiscard]] static std::vector<std::size_t> plan_next_run(
    const std::vector<parked_batch>& parked,
    const sort_run_limits& limits,
    bool has_string_column,
    bool producer_finished);

  //! This operator's counters.
  [[nodiscard]] order_by_counters counters() const noexcept;

  //! Counters summed over every ORDER_BY of the process since the last reset; for tests.
  [[nodiscard]] static order_by_counters process_counters() noexcept;
  static void reset_process_counters() noexcept;

 protected:
  //! Logs one `[order_by][stats]` line at INFO when the operator's pipeline finishes.
  void on_finalize_operator() override;

 private:
  //! Snapshot of the input port's parked batches. The caller holds `lock`.
  [[nodiscard]] std::vector<parked_batch> snapshot_parked(
    ::cucascade::shared_data_repository& repo);

  bool _coalesce = false;
  sort_run_limits _limits{};
  bool _has_string_column = false;
  //! Rows and decoded bytes of parked batches by batch id: neither changes when a batch moves
  //! between tiers, so each batch is measured once. Guarded by `lock`.
  struct cached_stats {
    std::size_t bytes = 0;
    std::optional<std::size_t> rows;
  };
  std::unordered_map<uint64_t, cached_stats> _stats_cache;

  std::atomic<uint64_t> _tasks{0};
  std::atomic<uint64_t> _input_batches{0};
  std::atomic<uint64_t> _runs{0};
  std::atomic<uint64_t> _concatenated_bytes{0};
};

}  // namespace op
}  // namespace sirius
