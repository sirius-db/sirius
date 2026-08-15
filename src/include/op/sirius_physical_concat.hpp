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

#include "duckdb/execution/physical_operator.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "sirius_config.hpp"

#include <unordered_map>

namespace sirius {
namespace op {

class sirius_physical_concat : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::CONCAT;

  //! `downstream_join` is the HJ/NLJ this CONCAT feeds — not the tree parent (that is
  //! `_parent_op`, stamped by `set_parent_ops`). Its join type picks `_concat_all`; the
  //! pointer is retained for the legacy converter's destination lookup.
  explicit sirius_physical_concat(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    sirius_physical_operator* downstream_join,
    bool is_build,
    uint64_t concat_batch_bytes = sirius::config::DEFAULT_CONCAT_BATCH_BYTES);

  std::string get_name() const override;

  bool is_source() const override;

  bool is_sink() const override;

  bool is_build_concat() const;

  //! The downstream HJ/NLJ this CONCAT feeds; distinct from `get_parent_op()` (tree parent).
  [[nodiscard]] sirius_physical_operator* get_downstream_join() const noexcept
  {
    return _downstream_join;
  }

  //! Answers READY/WAITING from the per-partition push-time totals (`_input_totals`), so the task
  //! creator's manager thread never touches a buffered `data_batch` lock. Once the source pipeline
  //! finishes, READY is decided from repository emptiness alone, independent of the totals.
  std::optional<task_creation_hint> get_next_task_hint() override;

  //! Forms the next input group with the greedy size-threshold walk over partitions. A group of
  //! exactly one batch is not wrapped in a task when sink wiring exists: since execute is an
  //! identity for one batch, the batch is pushed directly to the downstream consumers via their
  //! `push_data_batch_partitioned` (the same publication path `sink` uses) and the walk continues.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //! Measures the batch's size and records it in the per-partition totals before delegating to the
  //! base implementation, which publishes the batch into the input repository. This override is
  //! the accounting entry point: `get_next_task_hint` answers from the totals, so producers must
  //! push through it (the upstream PARTITION sink dispatches here through the virtual).
  void push_data_batch_partitioned(std::string_view port_id,
                                   std::shared_ptr<::cucascade::data_batch> batch,
                                   std::size_t partition_idx) override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void sink(const operator_data& output_data, rmm::cuda_stream_view stream) override;

  //! Used when PARTITION + `get_partition_strategy` selects BUILD_PROBE: merge all build batches
  //! before the join so the hash join sees a single build batch.
  void set_concat_all(bool concat_all);

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

 private:
  //! Running byte/batch totals over one input partition's currently buffered batches.
  struct partition_totals {
    uint64_t bytes    = 0;
    std::size_t count = 0;
  };

  //! Pop `batch_id` from `repo` and subtract its ledgered push-time size from the partition's
  //! totals. Requires `lock` to be held. Batches that entered the repository without going through
  //! `push_data_batch_partitioned` have no ledger entry and leave the totals untouched.
  std::shared_ptr<::cucascade::data_batch> pop_and_account(
    ::cucascade::shared_data_repository& repo, uint64_t batch_id, std::size_t partition_idx);

  bool _is_build;
  bool _concat_all;
  uint64_t _concat_batch_bytes;
  //! Non-owning. Captured at construction from the `downstream_join` ctor argument.
  sirius_physical_operator* _downstream_join = nullptr;
  //! Per-partition totals, indexed by partition and grown on push. Guarded by `lock`. Sizes are
  //! snapshotted at push time, so they can drift from live sizes while a batch idles (downgrade
  //! converts batches in place); the drift is bounded to the currently buffered batches and
  //! self-heals on pop — acceptable for a batching heuristic.
  std::vector<partition_totals> _input_totals;
  //! Push-time size of every currently buffered batch, keyed by batch id, so pops subtract exactly
  //! what the push added even when a batch's live size changed while idle. Guarded by `lock`.
  std::unordered_map<uint64_t, uint64_t> _pushed_batch_bytes;
};

}  // namespace op
}  // namespace sirius
