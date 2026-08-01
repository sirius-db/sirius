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

#include <atomic>

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

  std::optional<task_creation_hint> get_next_task_hint() override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void sink(const operator_data& output_data, rmm::cuda_stream_view stream) override;

  //! Used when PARTITION + `get_partition_strategy` selects BUILD_PROBE: merge all build batches
  //! before the join so the hash join sees a single build batch.
  void set_concat_all(bool concat_all);

  //! Probe-side counterpart of `set_concat_all`, applied when `get_partition_strategy` selects
  //! BUILD_PROBE with a single build partition. The build is folded into one hash table, but probe
  //! batches are joined against that whole table independently, so instead of coalescing the probe
  //! into one `_concat_batch_bytes` batch (one join task, one thread) this CONCAT emits up to
  //! `parts` batches. Only ever set on a probe-side CONCAT (`!is_build_concat()`); `parts <= 1`
  //! restores the default coalescing. `min_batch_bytes` floors the resulting batch size so a tiny
  //! probe is not shredded into tasks that cost more to schedule than to run.
  void set_probe_split_parts(int parts,
                             uint64_t min_batch_bytes = config::MIN_PROBE_SPLIT_BATCH_BYTES);

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

 private:
  //! Per-output-batch byte budget for the current pull. Normally `_concat_batch_bytes`; when the
  //! probe split is enabled AND the source pipeline has finished (so the total probe size is
  //! known), a smaller budget that yields ~`_probe_split_parts` batches. Caller must hold `lock`.
  //! The result is cached in `_split_budget_bytes`: recomputing it per pull would shrink the
  //! budget geometrically as the repository drains.
  [[nodiscard]] uint64_t effective_batch_bytes();

  bool _is_build;
  bool _concat_all;
  uint64_t _concat_batch_bytes;
  //! Target probe-batch count for a single-partition BUILD_PROBE join; 1 = disabled. Set once at
  //! partition-sizing time, read by `execute` off the lock, hence atomic.
  std::atomic<int> _probe_split_parts{1};
  //! Floor on a split probe batch (see `set_probe_split_parts`).
  uint64_t _min_probe_split_bytes = config::MIN_PROBE_SPLIT_BATCH_BYTES;
  //! Cached probe-split byte budget (0 = not yet fixed). See `effective_batch_bytes`.
  std::atomic<uint64_t> _split_budget_bytes{0};
  //! Non-owning. Captured at construction from the `downstream_join` ctor argument.
  sirius_physical_operator* _downstream_join = nullptr;
};

}  // namespace op
}  // namespace sirius
