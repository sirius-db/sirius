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

#include "op/aggregate/probe_accumulate_context.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cstddef>
#include <memory>
#include <optional>

namespace sirius::op {

/**
 * @brief Emits the result of a probe-accumulate hash join once its probe side is exhausted.
 *
 * A hash join in accumulate mode (see `sirius_physical_hash_join::accumulate`) folds each probe
 * batch into per-build-row accumulators and outputs nothing but empty batches. This operator
 * sits directly above it. It waits for the join's pipeline to finish, then runs one task that
 * reads the accumulators from the shared `probe_accumulate_context` and produces one batch with
 * the fused aggregate's final schema: `[group key, aggregate...]`, one row per build row that
 * matched at least once.
 *
 * The task runs exactly once whether or not the join delivered any batch, so an empty build or an
 * empty probe still yields the (empty) result.
 */
class sirius_physical_probe_accumulate_emit : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::PROBE_ACCUMULATE_EMIT;

  /**
   * @param types Output schema `[group key, aggregate...]`
   * @param estimated_cardinality Estimated number of output groups
   * @param context State shared with the child join
   */
  sirius_physical_probe_accumulate_emit(duckdb::vector<sirius::logical_type> types,
                                        std::size_t estimated_cardinality,
                                        std::shared_ptr<probe_accumulate_context> context);

  [[nodiscard]] std::string params_to_string() const override;

  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }

  sirius::OrderPreservationType source_order() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }
  bool sink_order_dependent() const override { return false; }

  /**
   * @brief Wait for the child join's pipeline to finish, then offer exactly one task
   *
   * The base implementation gates on queued input bytes, which the join's empty batches never
   * add up to, so it would never fire.
   */
  std::optional<task_creation_hint> get_next_task_hint() override;

  /// The (empty) batches the join delivered, once; `nullptr` after that.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  /**
   * @brief Peak device memory of the emit task
   *
   * The task's input is a few empty batches, so the base estimate (a multiple of the input bytes)
   * would reserve nothing for a result that has one row per matched build row. This sizes it from
   * the build table the join recorded: the compaction and output columns, counted twice because the
   * result is cached and a copy becomes the output batch.
   */
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

 private:
  std::shared_ptr<probe_accumulate_context> _context;
  /// Whether the single task was handed out; guarded by the base `lock`.
  bool _emitted = false;
};

}  // namespace sirius::op
