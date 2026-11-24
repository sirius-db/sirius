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
  std::size_t shrinks{0};  ///< windowed inputs that shrank their window for an OOM retry
  /// Windowed inputs whose OOM retry reserved window buffers for the runs they had planned to
  /// slice in place, at unchanged sizes. Not part of the statistics line.
  std::size_t recharges{0};
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
 *    emit are not read again by the next one. The window buffers of a kept merger stay
 *    allocated between two tasks, outside any reservation; the next task's estimate leaves them
 *    out, and on_finalize_operator() frees them. The window budget assumes one such merger per
 *    concurrent task of a device, so a partition without a merger gets its next task only while
 *    fewer windowed partitions than that are open (see at_open_partition_cap()). That number is
 *    the executor width, or the number of partitions the operator has when that is lower: the
 *    budget is then shared among the partitions that can really be open.
 *
 * In automatic mode a partition whose only run is on the GPU has nothing to merge and is
 * forwarded on the one-shot route, whatever its size; in windowed mode it is still read by a
 * windowed task. The merge_rule values (sort_merge_disabled_rules) switch this and the other
 * sizing and planning rules off one by one.
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

  /// READY only while some partition can be armed: one with runs and no task in flight, which
  /// either holds its merger or may open under the cap on open windowed partitions.
  std::optional<task_creation_hint> get_next_task_hint() override;

  /// Runs stay on whatever tier they are in and their repository is spilled first, unless the
  /// merge mode is one_shot.
  [[nodiscard]] port_residency_policy input_residency_policy(
    std::string_view port_id) const override;

  /// A windowed task needs its window working set, which is already what its input reports.
  /// With history, gpu_pipeline_task keeps the reservation at or above that working set. A
  /// one-shot merge of several batches peaks at 2.5 x its input on top of the materialised
  /// input (the pairwise merge's intermediates, its 8-byte row index and the output); with the
  /// one_shot_estimate rule, which is off by default, that is its cold-start estimate. A
  /// single batch is forwarded and keeps the default.
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
  /// of @p memory_manager when it is given none, with @p concurrent_tasks tasks per device (the
  /// executor width; arming lowers it to the operator's partition count).
  /// Its available bytes include the bytes reserved by the tasks in flight.
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

  /// Logs the statistics line if no task commit has: an operator with no input, or one whose
  /// pipeline finished before its last task committed (a LIMIT downstream), still logs one line.
  /// Frees the mergers the partitions keep between tasks, and with them their window buffers; a
  /// task that commits afterwards keeps none.
  void on_finalize_operator() override;

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
    /// Per cursor, as of the last commit: the merger holds window buffers for the batch the run
    /// is at. The run's last task was planned to read that batch through a window buffer and
    /// did not finish it. Taken from the plan, not from the merger: a batch the task found
    /// sliceable after all (converted whole to the GPU) is still counted.
    std::vector<bool> holds_buffer;
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
    /// A thread is parsing the header of one of the partition's compressed batches, without
    /// _drain_mutex, to learn the schema; it arms the partition when it has it.
    bool header_parsing{false};
    /// The header was parsed; header_schema is what it gave (null if it could not be parsed).
    bool header_parsed{false};
    std::optional<std::vector<cudf::data_type>> header_schema;
    merge::merge_route route{merge::merge_route::one_shot};
    /// Bytes and rows of the partition's runs when it was armed.
    merge::run_summary input_size{};
    /// Sizes of the partition's windowed tasks: derived_config, less what a task of the pass in
    /// progress shrank after an OOM.
    merge::derived_merge_config config{};
    /// The sizes derived at arming; every pass starts from them.
    merge::derived_merge_config derived_config{};
    /// A task of the final pass has been planned.
    bool final_pass_started{false};
    /// device_budget::concurrent_tasks those sizes were derived with: the mergers the window
    /// budget allows on one device at a time. With the partition_budget rule it is the executor
    /// width lowered to the number of partitions the operator had when this one was armed.
    std::size_t concurrent_tasks{1};
    /// The GPU memory space the task in flight runs in; null until it starts.
    ::cucascade::memory::memory_space* task_space{nullptr};
    /// Types of the input columns, read from the partition's batches.
    std::vector<cudf::data_type> column_types;
    /// Estimated bytes per row of each input column.
    std::vector<std::size_t> column_widths;
    /// run_state::id of the runs the task in flight reads, in the order of its cursors.
    std::vector<std::uint64_t> task_runs;
    /// Per run of task_runs: the task in flight was planned to read the batch the run is at
    /// through a window buffer.
    std::vector<bool> task_buffered;
    /// The intermediate run being written: its batches so far, in key order.
    run_state pending_run;
    /// The merger of the partition's task chain while no task holds it; null before the first
    /// task, and whenever the next task must build a new one.
    std::unique_ptr<partition_merger> merger;
  };

  /// The next windowed task of a partition, as planned under _drain_mutex. Sizing it reads the
  /// batches (and, once per compressed batch, its header), which is done without the mutex.
  struct windowed_task_plan {
    std::size_t partition{0};
    bool final_pass{true};
    std::size_t first_chunk{0};
    merge::window_config config{};
    /// One cursor per run the task reads.
    std::vector<merge::merge_input_cursor> cursors;
    /// Per cursor: the batches of cursor.run and their decoded bytes.
    std::vector<std::vector<std::shared_ptr<::cucascade::data_batch>>> batches;
    std::vector<std::vector<std::size_t>> batch_bytes;
    /// Per cursor: partition_merger::holds_buffer of the run in a kept merger this task
    /// continues; false without one.
    std::vector<bool> kept_buffer;
    /// Device of the partition's kept merger, if it has one.
    std::optional<int> kept_device;
    /// Input column indices the task fetches, in fetch order.
    std::vector<std::size_t> fetch_cols;
    std::size_t row_width{0};       ///< estimated bytes per row of the fetched columns
    std::size_t key_row_width{0};   ///< estimated bytes per row of the key columns
    std::size_t string_columns{0};  ///< how many of the fetched columns are STRING
    /// Estimated bytes per row of the STRING fetched columns together; 0 without the
    /// string_slot_estimate rule.
    std::size_t string_row_width{0};
    /// config.round_rows was sized from the output target (the round_rows rule).
    bool round_follows_output{false};
  };

  /// What a task's commit leaves to do once _drain_mutex is released.
  struct commit_followup {
    bool reschedule{false};
    std::optional<merge_sort_counters> final_counters;
  };

  /// First arm of a partition: sizes its batches, chooses the route and, on the windowed route,
  /// builds one cursor per batch. Returns the task input on the one-shot route, else null. Takes
  /// no blocking batch lock and reads no batch header. When a batch it must read is locked by a
  /// tier move, the partition is left unarmed and @p busy is set to that batch. When only the
  /// header of a compressed batch names the schema and it is not parsed yet, the partition is
  /// left unarmed and @p header_source is set to that batch's read lock: the caller parses it
  /// without _drain_mutex, stores the result in the partition's state and arms again.
  std::unique_ptr<operator_data> arm_partition(
    std::size_t partition,
    partition_merge_state& state,
    ::cucascade::shared_data_repository& repo,
    std::shared_ptr<::cucascade::data_batch>& busy,
    std::optional<::cucascade::read_only_data_batch>& header_source);

  /// Pops every batch of @p partition into one merge task.
  std::unique_ptr<operator_data> make_one_shot_input(std::size_t partition,
                                                     ::cucascade::shared_data_repository& repo);

  /// Plans the next windowed task of an armed partition and marks it in flight. Needs
  /// _drain_mutex.
  windowed_task_plan plan_windowed_task(std::size_t partition,
                                        partition_merge_state& state,
                                        ::cucascade::shared_data_repository& repo);

  /// Sizes a planned task and builds its input. Call without _drain_mutex: it reads the batches
  /// and may read a compressed batch's header from disk. If it throws, the partition is no
  /// longer in flight and the task can be planned again.
  std::unique_ptr<operator_data> make_windowed_input(windowed_task_plan plan);

  /// Whether the row-range reader of @p data can read every column of @p columns. True for a
  /// batch that is not spill-compressed. For a compressed one the answer comes from its header,
  /// parsed once per batch id (the plan of a batch does not depend on its tier).
  [[nodiscard]] bool reads_by_row_range(std::uint64_t batch_id,
                                        const ::cucascade::idata_representation& data,
                                        const std::vector<std::size_t>& columns);

  /// The part every task commit ends with: makes the partitions whose arming was deferred
  /// armable again, decides whether the operator has to be scheduled again and claims the
  /// statistics line when no work is left. Reschedules when a partition is armable and either
  /// @p always_reschedule or some partition was deferred. Needs _drain_mutex.
  [[nodiscard]] commit_followup finish_commit(::cucascade::shared_data_repository& repo,
                                              bool always_reschedule);

  /// Logs the statistics line and schedules the operator, as @p followup says. Call without
  /// _drain_mutex.
  void after_commit(const commit_followup& followup);

  std::unique_ptr<operator_data> execute_one_shot(const pipelineable_operator_data& input,
                                                  ::cuda::stream_ref stream);

  std::unique_ptr<operator_data> execute_windowed(const merge::windowed_merge_input& input,
                                                  ::cuda::stream_ref stream);

  /// The counters to log in the statistics line, for the one caller that gets to log it; null
  /// once the line is taken. Needs _drain_mutex.
  [[nodiscard]] std::optional<merge_sort_counters> claim_stats_line();

  /// Whether as many windowed partitions are open as the window budget of @p candidate assumes,
  /// so that @p candidate, which holds no merger, has to wait for one of them to finish. A
  /// partition is open while it has a task in flight or keeps its merger. The budget is per
  /// device and so is the count: an open partition counts on the device of its merger or of its
  /// running task, and one whose first task has not started, like @p candidate itself, may land
  /// on any device and counts on each. One-shot partitions hold no merger and are not counted.
  /// Needs _drain_mutex.
  [[nodiscard]] bool at_open_partition_cap(const partition_merge_state& candidate) const;

  /// Whether a call to get_next_task_input_data() would produce a task. Needs _drain_mutex.
  [[nodiscard]] bool has_armable_partition(::cucascade::shared_data_repository& repo);

  /// The key column indices of `orders`, with their cudf order and null precedence.
  [[nodiscard]] merge::merge_key_spec input_key_spec() const;

  /// The input columns a final-pass task emits, in output order.
  [[nodiscard]] std::vector<std::size_t> output_columns(std::size_t num_input_columns) const;

  /// Partitions of @p repo that hold batches or have been armed: the most windowed partitions
  /// that can be open. Needs _drain_mutex.
  [[nodiscard]] std::size_t count_partitions(::cucascade::shared_data_repository& repo) const;

  /// Bytes and rows of the runs of every partition of the sort: as recorded at arming for an
  /// armed partition, from the batches in @p repo for the others. A batch that was never sized
  /// and is locked by a tier move is left out. Never waits for a batch lock. Needs _drain_mutex.
  [[nodiscard]] std::vector<merge::run_summary> sort_partition_sizes(
    ::cucascade::shared_data_repository& repo);

  //! Final projection to apply after merge (empty = no extra projection)
  duckdb::vector<std::size_t> _final_projections;
  merge::merge_settings _merge_settings{.mode = sort_merge_mode::one_shot};
  device_budget_provider _device_budget_provider;
  //! Guards _partitions, _gpu_space_hint, _counters and _stats_logged, in get_next_task_hint()
  //! too. Taken before the process-wide counter mutex when both are held. Never held while a
  //! merge runs or while waiting for a batch lock.
  mutable std::mutex _drain_mutex;
  std::map<std::size_t, partition_merge_state> _partitions;
  merge_sort_counters _counters;
  //! The statistics line was logged, or is about to be by the thread that set this. Guarded by
  //! _drain_mutex, so the last commit and on_finalize_operator() log it once between them.
  bool _stats_logged{false};
  //! Guards _pushed_sizes; taken after _drain_mutex when both are held.
  std::mutex _pushed_mutex;
  /// Sizes of the batches of partitions not armed yet, by batch id.
  std::unordered_map<std::uint64_t, batch_size> _pushed_sizes;
  //! Guards _row_range_columns; taken after _drain_mutex when both are held.
  std::mutex _row_range_mutex;
  /// By id of a batch seen spill-compressed: per source column, whether its plan can be decoded
  /// by row range (empty when the header could not be read). Erased when the batch is consumed.
  std::unordered_map<std::uint64_t, std::vector<bool>> _row_range_columns;
  /// The GPU memory space last seen holding runs; sizes partitions with no GPU-resident run.
  ::cucascade::memory::memory_space* _gpu_space_hint{nullptr};
};

}  // namespace op
}  // namespace sirius
