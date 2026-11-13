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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

namespace sirius {
namespace op {

namespace merge {
class windowed_merge_input;
}  // namespace merge

/**
 * @brief Merges the sorted runs of each sort partition into the query's output order.
 *
 * A partition takes one of two routes, chosen when it is first armed:
 *  - one-shot: every batch of the partition is popped into one task, which merges them whole;
 *  - windowed: the runs stay in the input repository and a chain of tasks, one at a time per
 *    partition, reads them through CPU-side cursors. Each task publishes one output chunk (or,
 *    when the partition has more runs than the fan-in cap, writes an intermediate run back into
 *    the partition) and schedules the operator again.
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

  /// Supplies the GPU memory and task concurrency the windows are sized from. Without a
  /// provider the operator uses the GPU memory space holding most of the partition's bytes, and
  /// the default GPU executor width as the number of concurrent tasks.
  void set_device_budget_provider(std::function<merge::device_budget()> provider)
  {
    _device_budget_provider = std::move(provider);
  }

 protected:
  /// Whether late materialization is installed on this operator; such a sort merges one-shot.
  [[nodiscard]] virtual bool has_late_materialization() const noexcept;

 private:
  /// One sorted run of a partition and where the merge has read up to.
  struct run_state {
    /// The run's unread batches, front first: batches read to the end are dropped from it.
    merge::merge_input_cursor cursor;
    /// Decoded bytes of each batch of cursor.run.
    std::vector<std::size_t> batch_bytes;
    /// Selected by the intermediate pass in progress.
    bool in_pass{false};
  };

  struct partition_merge_state {
    std::vector<run_state> runs;
    std::size_t next_chunk{0};
    /// At most one task per partition: each task starts from the previous one's cursors.
    bool in_flight{false};
    bool armed{false};
    merge::merge_route route{merge::merge_route::one_shot};
    /// Sizes of the partition's windowed tasks.
    merge::derived_merge_config config{};
    /// Types of the input columns, read from the partition's batches.
    std::vector<cudf::data_type> column_types;
    /// Estimated bytes per row of each input column.
    std::vector<std::size_t> column_widths;
    /// Indices into `runs` of the runs the task in flight reads.
    std::vector<std::size_t> task_runs;
    /// The intermediate run being written: its batches so far, in key order.
    run_state pending_run;
  };

  /// First arm of a partition: sizes its batches, chooses the route and, on the windowed route,
  /// builds one cursor per batch. Returns the task input on the one-shot route, else null.
  std::unique_ptr<operator_data> arm_partition(std::size_t partition,
                                               partition_merge_state& state,
                                               ::cucascade::shared_data_repository& repo);

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
  std::function<merge::device_budget()> _device_budget_provider;
  //! Guards _partitions and _gpu_space_hint, in get_next_task_hint() too.
  std::mutex _drain_mutex;
  std::map<std::size_t, partition_merge_state> _partitions;
  /// The GPU memory space last seen holding runs; sizes partitions with no GPU-resident run.
  ::cucascade::memory::memory_space* _gpu_space_hint{nullptr};
};

}  // namespace op
}  // namespace sirius
