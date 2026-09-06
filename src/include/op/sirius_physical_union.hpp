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

#include "op/sirius_physical_operator.hpp"

#include <string>
#include <vector>

namespace sirius {
namespace op {

//! Physical `UNION ALL`: an N-ary, non-materializing fan-in. Bag union computes nothing, so this
//! operator only routes batches from every arm into one downstream stream and `execute` is the
//! identity. Distinct `UNION`, `EXCEPT` and `INTERSECT` never reach it: the plan builder accepts
//! only `setop_all == true`.
//!
//! `wrap_union` wraps each arm `child -> PASSTHROUGH_SINK`, and each sink feeds a *distinct* input
//! port, `port_label(i)` for `children[i]`. The distinct names are a correctness requirement:
//! `add_port` is last-writer-wins on the name, the repository manager keys by
//! `(operator_id, port_id)`, and the pipeline-finish gate reads `is_source_pipeline_finished()` /
//! `all_ports_empty()` across the same `ports` map. A shared name would therefore orphan an arm's
//! repository, and because the orphan is no longer in that map, let the pipeline finish while that
//! arm still had rows.
class sirius_physical_union : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::UNION;

  //! @param types  Output schema, carried through unchanged. The binder has already cast every
  //!               arm to the common super-type, so Sirius does no type reconciliation itself.
  explicit sirius_physical_union(duckdb::vector<sirius::logical_type> types,
                                 std::size_t estimated_cardinality);

  //! The input port name for arm `i`; the single definition of the `"union_{i}"` convention.
  //!
  //! Returns by value, so it is safe for `get_port` (which copies) but NOT as the return of
  //! `input_port_for`, whose `string_view` is retained by the wiring descriptor and
  //! `next_port_info`. That view must point at the sink's own `_union_port_label` instead.
  [[nodiscard]] static std::string port_label(std::size_t arm_index)
  {
    return "union_" + std::to_string(arm_index);
  }

  std::string get_name() const override;

  bool is_source() const override;

  //! `NO_ORDER`: `order_preservation_recursive` stops at the first `is_source()` operator, so this
  //! answer decides the whole plan's, and a bag union provides no ordering.
  sirius::OrderPreservationType source_order() const override;

  //! The base throws for a non-sink operator with more than one child, so a multi-input operator
  //! must supply its own.
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  //! The base recurses into `children[0]` and throws when there is more than one child; collect
  //! every arm's sources instead.
  duckdb::vector<duckdb::const_reference<sirius_physical_operator>> get_sources() const override;

  //! Identity: forward the batch that `get_next_task_input_data` popped, unchanged.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //! Each arm feeds its own `"union_{i}"` port, labelled on that arm's PASSTHROUGH_SINK by
  //! `wrap_union`. Distinct names are required, not cosmetic: `add_port` is last-writer-wins and
  //! the repository manager keys by (operator_id, port_id), so a shared name would orphan an arm's
  //! repository. The returned view is backed by the producer's own member, which outlives every
  //! consumer of it.
  [[nodiscard]] std::string_view input_port_for(
    sirius_physical_operator const& producer) const override;

  //! A UNION arm streams: bag union keeps no cross-batch state and makes no ordering guarantee, so
  //! it never needs a complete side. Same answer as the join's inbound edges, so a fan-in consumer
  //! absorbed into its producers' branches is the established shape rather than a novelty. UNION
  //! overrides its own hint and never reads `port::type`, but the value still matters:
  //! `query_index` cuts a branch only on a FULL edge, so PARTIAL lets each arm's branch walk absorb
  //! the UNION's pipeline rather than give it one of its own, which feeds scheduling priorities. It
  //! is also what would keep the base hint from buffering every arm to completion, the day someone
  //! simplifies UNION back onto it.
  [[nodiscard]] MemoryBarrierType input_barrier_for(
    sirius_physical_operator const& producer) const override;

  //! READY when every *live* arm has a batch, popping one batch from one arm. The base is lockstep
  //! — READY only when every port has data, then one batch from every port — which strands a long
  //! arm's remaining batches once a short arm drains and its pipeline finishes. Excusing a finished
  //! arm is the whole difference; arms still producing rendezvous. A starved arm (live producer,
  //! empty port) outranks a ready one, because nothing else will start it. One arm per task also
  //! keeps each batch on the GPU that produced it, since a task runs on one device.
  std::optional<task_creation_hint> get_next_task_hint() override;
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //! Pure forwarder: no device allocation beyond the batches already resident.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& /*stats*/) const override
  {
    return 0;
  }

 private:
  //! Arm ports in arm order, resolved once on first use. Callers must hold `lock`.
  const std::vector<port*>& arm_ports();

  //! Arm to start the next pop at, so draining rotates across arms rather than emptying arm 0
  //! first. Guarded by `lock`.
  std::size_t _arm_cursor = 0;

  //! Arm to start the next producer nomination at, so successive `WAITING_FOR_INPUT_DATA` hints
  //! do not keep naming the same arm. Separate from `_arm_cursor`. Guarded by `lock`.
  std::size_t _wait_cursor = 0;

  //! Cache behind `arm_ports()`; empty until the wiring has been materialised.
  std::vector<port*> _arm_ports;
};

}  // namespace op
}  // namespace sirius
