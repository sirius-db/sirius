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
#include "op/merge/merge_config.hpp"
#include "op/merge/windowed_merge.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"

#include <cudf/types.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sirius {
namespace op {

namespace merge {
class windowed_merge_input;
}  // namespace merge

/**
 * @brief Counts of the merge routes taken and of the work the windowed tasks did.
 */
struct merge_sort_counters {
  std::size_t windowed_partitions{0};  ///< partitions armed on the windowed route
  std::size_t one_shot_partitions{0};  ///< partitions armed on the one-shot route
  std::size_t windowed_tasks{0};       ///< windowed tasks that committed
  std::size_t intermediate_tasks{0};   ///< of those, the tasks that wrote an intermediate run
  std::size_t mergers_built{0};        ///< windowed mergers constructed
  merge::merge_round_stats rounds;     ///< merge counters summed over the committed tasks
};

/**
 * @brief Merges the sorted runs of each sort partition into the query's output order.
 *
 * A partition takes one of two routes, chosen when it is first armed:
 *  - one-shot: every batch of the partition is popped into one task, which merges them whole;
 *  - windowed: the runs stay in the input repository and a chain of tasks, one at a time per
 *    partition, reads them through CPU-side cursors. Each task publishes one output chunk (or,
 *    when the partition has more runs than the fan-in cap, writes an intermediate run back into
 *    the partition) and schedules the operator again. The tasks of a chain share one merger,
 *    kept in the partition's state between them, so rows a task read into a window but did not
 *    emit are not read again by the next one.
 */
class sirius_physical_merge_sort : public sirius_physical_operator {
 public:
  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }

  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::MERGE_SORT;

 public:
  sirius_physical_merge_sort(sirius_physical_order* order_by);

  sirius_physical_merge_sort(duckdb::vector<sirius::logical_type> types,
                             duckdb::vector<duckdb::BoundOrderByNode> orders,
                             duckdb::vector<std::size_t> projections_p,
                             std::size_t estimated_cardinality,
                             bool is_index_sort_p = false);

  //! Input data
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  duckdb::vector<std::size_t> projections;
  bool is_index_sort;

  sirius_physical_operator* child_op;
  sirius_physical_operator* get_child_op() const { return child_op; }

 public:
  // Source interface
  bool is_source() const override { return true; }

 public:
  // Sink interface
  bool is_sink() const override { return true; }

 public:
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  /// READY only while some partition can be armed: one with runs and no task in flight.
  std::optional<task_creation_hint> get_next_task_hint() override;

  /// Runs stay on whatever tier they are in and their repository is spilled first, unless the
  /// merge mode is one_shot.
  [[nodiscard]] port_residency_policy input_residency_policy(
    std::string_view port_id) const override;

  /// A windowed task needs its window working set, which is already what its input reports.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  //! Set the final output projection (applied after merge, to remove sort-key-only columns)
  void set_final_projections(duckdb::vector<std::size_t> proj,
                             duckdb::vector<sirius::logical_type> output_types)
  {
    _final_projections = std::move(proj);
    types              = std::move(output_types);
  }

  /// Install the query's sort_merge_* settings. Call before the first task is created. An
  /// operator that never receives them merges every partition one-shot at `position = partition`.
  void set_merge_settings(merge::merge_settings settings) noexcept { _merge_settings = settings; }

  [[nodiscard]] const merge::merge_settings& get_merge_settings() const noexcept
  {
    return _merge_settings;
  }

  /// Returns the budget a partition's windows are sized from. The argument is the GPU memory
  /// space holding most of the partition's bytes, or null when none of its runs is on a GPU.
  using device_budget_provider =
    std::function<merge::device_budget(const ::cucascade::memory::memory_space*)>;

  /// Supplies the GPU memory and task concurrency the windows are sized from. Without a
  /// provider the operator uses the GPU memory space holding most of the partition's bytes, and
  /// the default GPU executor width as the number of concurrent tasks.
  void set_device_budget_provider(device_budget_provider provider)
  {
    _device_budget_provider = std::move(provider);
  }

  [[nodiscard]] bool has_device_budget_provider() const noexcept
  {
    return static_cast<bool>(_device_budget_provider);
  }

  /// A provider that reports the memory of the space it is given, or of the smallest GPU space
  /// of @p memory_manager when it is given none, with @p concurrent_tasks tasks per device.
  /// @p memory_manager must outlive the provider.
  [[nodiscard]] static device_budget_provider make_device_budget_provider(
    const ::cucascade::memory::memory_reservation_manager& memory_manager,
    std::size_t concurrent_tasks);

  /// This operator's counters so far.
  [[nodiscard]] merge_sort_counters counters() const;

  /// The counters summed over every merge sort operator of the process since the last reset.
  [[nodiscard]] static merge_sort_counters process_counters();
  static void reset_process_counters();

 protected:
  /// Whether late materialization is installed on this operator; such a sort merges one-shot.
  [[nodiscard]] virtual bool has_late_materialization() const noexcept;

  /// Records the size of a run as it arrives, so arming a partition later does not need the
  /// lock of a run that is being moved between tiers by then.
  void on_input_batch_pushed(std::string_view port_id, ::cucascade::data_batch& batch) override;

 private:
  /// One sorted run of a partition and where the merge has read up to.
  struct run_state {
    /// Identifies the run within its partition for as long as the partition is merged.
    std::uint64_t id{0};
    /// The run's unread batches, front first: batches read to the end are dropped from it.
    merge::merge_input_cursor cursor;
    /// Decoded bytes of each batch of cursor.run.
    std::vector<std::size_t> batch_bytes;
    /// Selected by the intermediate pass in progress.
    bool in_pass{false};
  };

  /// Rows and decoded bytes of a batch.
  struct batch_size {
    std::optional<std::size_t> rows;
    std::size_t bytes{0};
  };

  /// The merger a partition's tasks share. It is in the partition's state between two tasks and
  /// owned by the task while one runs.
  struct partition_merger {
    /// run_state::id of the run each cursor reads.
    std::vector<std::uint64_t> run_ids;
    /// The merger's cursors: it reads them and writes them back. Never resized, so the span the
    /// merger holds stays valid.
    std::vector<merge::merge_input_cursor> cursors;
    /// What the merger was built for; a task with anything else builds a new one.
    ::cucascade::memory::memory_space* space{nullptr};
    merge::window_config config{};
    bool final_pass{true};
    /// Declared last: destroyed before the cursors it refers to.
    std::optional<merge::windowed_merger> merger;
  };

  struct partition_merge_state {
    std::vector<run_state> runs;
    std::uint64_t next_run_id{0};
    std::size_t next_chunk{0};
    /// At most one task per partition: each task starts from the previous one's cursors.
    bool in_flight{false};
    bool armed{false};
    /// Arming found a run locked by a tier move; a committing task makes it armable again.
    bool arm_deferred{false};
    merge::merge_route route{merge::merge_route::one_shot};
    /// Sizes of the partition's windowed tasks.
    merge::derived_merge_config config{};
    /// Types of the input columns, read from the partition's batches.
    std::vector<cudf::data_type> column_types;
    /// Estimated bytes per row of each input column.
    std::vector<std::size_t> column_widths;
    /// run_state::id of the runs the task in flight reads, in the order of its cursors.
    std::vector<std::uint64_t> task_runs;
    /// The intermediate run being written: its batches so far, in key order.
    run_state pending_run;
    /// The merger of the partition's task chain while no task holds it; null before the first
    /// task, and whenever the next task must build a new one.
    std::unique_ptr<partition_merger> merger;
  };

  /// First arm of a partition: sizes its batches, chooses the route and, on the windowed route,
  /// builds one cursor per batch. Returns the task input on the one-shot route, else null. Takes
  /// no blocking batch lock: when a batch it must read is locked by a tier move, the partition
  /// is left unarmed and @p busy is set to that batch.
  std::unique_ptr<operator_data> arm_partition(std::size_t partition,
                                               partition_merge_state& state,
                                               ::cucascade::shared_data_repository& repo,
                                               std::shared_ptr<::cucascade::data_batch>& busy);

  /// Pops every batch of @p partition into one merge task.
  std::unique_ptr<operator_data> make_one_shot_input(std::size_t partition,
                                                     ::cucascade::shared_data_repository& repo);

  /// Plans the next windowed task of an armed partition and marks it in flight.
  std::unique_ptr<operator_data> make_windowed_input(std::size_t partition,
                                                     partition_merge_state& state,
                                                     ::cucascade::shared_data_repository& repo);

  std::unique_ptr<operator_data> execute_one_shot(const pipelineable_operator_data& input,
                                                  ::cuda::stream_ref stream);

  std::unique_ptr<operator_data> execute_windowed(const merge::windowed_merge_input& input,
                                                  ::cuda::stream_ref stream);

  /// Whether a call to get_next_task_input_data() would produce a task. Needs _drain_mutex.
  [[nodiscard]] bool has_armable_partition(::cucascade::shared_data_repository& repo);

  /// The key column indices of `orders`, with their cudf order and null precedence.
  [[nodiscard]] merge::merge_key_spec input_key_spec() const;

  /// The input columns a final-pass task emits, in output order.
  [[nodiscard]] std::vector<std::size_t> output_columns(std::size_t num_input_columns) const;

  //! Final projection to apply after merge (empty = no extra projection)
  duckdb::vector<std::size_t> _final_projections;
  merge::merge_settings _merge_settings{.mode = sort_merge_mode::one_shot};
  device_budget_provider _device_budget_provider;
  //! Guards _partitions, _gpu_space_hint and _counters, in get_next_task_hint() too. Never held
  //! while a merge runs or while waiting for a batch lock.
  mutable std::mutex _drain_mutex;
  std::map<std::size_t, partition_merge_state> _partitions;
  merge_sort_counters _counters;
  //! Guards _pushed_sizes; taken after _drain_mutex when both are held.
  std::mutex _pushed_mutex;
  /// Sizes of the batches of partitions not armed yet, by batch id.
  std::unordered_map<std::uint64_t, batch_size> _pushed_sizes;
  /// The GPU memory space last seen holding runs; sizes partitions with no GPU-resident run.
  ::cucascade::memory::memory_space* _gpu_space_hint{nullptr};
};

}  // namespace op
}  // namespace sirius
