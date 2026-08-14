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

namespace sirius {
namespace op {

//! Buffers the per-partition data batches produced upstream and releases them in groups sized to
//! `_concat_batch_bytes` for the downstream hash or nested-loop join, so the join consumes a few
//! large inputs instead of many small ones. Group formation follows a single policy implemented in
//! `plan_pull_for_partition`: `get_next_task_hint` reports READY exactly when that policy would
//! release a group, and `get_next_task_input_data` pops exactly the batches the policy selects, so
//! task scheduling and data pulls cannot disagree.
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

  std::optional<task_creation_hint> get_next_task_hint() override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void sink(const operator_data& output_data, rmm::cuda_stream_view stream) override;

  //! Used when PARTITION + `get_partition_strategy` selects BUILD_PROBE: merge all build batches
  //! before the join so the hash join sees a single build batch.
  void set_concat_all(bool concat_all);

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

 private:
  //! Decides which batches a pull from partition `partition_idx` of `repo` would take, without
  //! removing anything. The policy: batches accumulate in arrival order until their cumulative size
  //! first exceeds `_concat_batch_bytes`; the accumulated group is then released and the
  //! overflowing batch stays behind to seed the next group. A batch that exceeds the threshold on
  //! its own is released as a single-batch group once another batch sits behind it or
  //! `pipeline_finished` is true. A group that never reaches the threshold is released only when
  //! `pipeline_finished` is true (tail flush). When `_concat_all` is set the whole partition forms
  //! one group, released only when `pipeline_finished` is true.
  //!
  //! Must be called with the operator mutex `lock` held. The returned plan stays valid while the
  //! lock is held: only `get_next_task_input_data` removes batches (also under `lock`), and
  //! concurrent producers can only append.
  //!
  //! @param repo The single input port's data repository
  //! @param partition_idx The partition to plan a pull for
  //! @param pipeline_finished Whether the source pipeline can produce no more batches
  //! @return The batch ids to pop, in pull order, or std::nullopt if no group should form yet
  [[nodiscard]] std::optional<std::vector<uint64_t>> plan_pull_for_partition(
    ::cucascade::shared_data_repository& repo,
    std::size_t partition_idx,
    bool pipeline_finished) const;

  bool _is_build;
  bool _concat_all;
  uint64_t _concat_batch_bytes;
  //! Non-owning. Captured at construction from the `downstream_join` ctor argument.
  sirius_physical_operator* _downstream_join = nullptr;
};

}  // namespace op
}  // namespace sirius
