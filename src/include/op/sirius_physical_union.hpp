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

//! Physical `UNION ALL`: an N-ary, non-materializing fan-in. Bag (multiset) union computes
//! nothing — no de-duplication, no key, no comparison, no ordering guarantee — so this operator
//! only routes batches from all of its arms into one downstream stream. `execute` is the identity.
//!
//! Each arm is wrapped `child -> PASSTHROUGH_SINK` by `wrap_union`, and each sink feeds a
//! *distinct* input port, `port_label(i)` for `children[i]`. Distinct ports are mandatory rather
//! than tidy: `add_port` is last-writer-wins on the name, the repository manager keys by
//! `(operator_id, port_id)`, and the pipeline-finish gate reads `is_source_pipeline_finished()` /
//! `all_ports_empty()` across the `ports` map — so a shared name would orphan an arm's repository
//! and let the UNION pipeline finish while that arm still had rows.
//!
//! N-ary, not binary: DuckDB binds `a UNION ALL b UNION ALL c` to one `LogicalSetOperation` with
//! three children, and `UNION BY NAME` / macro expansion can produce N-ary nodes regardless. A
//! "handle exactly 2" cut would silently fall back to the CPU on the common 3-way chain.
//!
//! Distinct `UNION`, `EXCEPT` and `INTERSECT` never reach this operator: the plan builder accepts
//! only `setop_all == true`, so those still fall back to the CPU.
class sirius_physical_union : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::UNION;

  //! @param types  Output schema. The binder has already cast every arm to the per-column common
  //!               super-type and guaranteed matching arity and order, so this is carried through
  //!               unchanged and Sirius does no type reconciliation of its own.
  explicit sirius_physical_union(duckdb::vector<sirius::logical_type> types,
                                 std::size_t estimated_cardinality);

  //! The input port name for arm `i` — the single definition of the `"union_{i}"` convention,
  //! shared by `wrap_union` (which labels each arm's sink) and the task-driver methods below.
  //!
  //! Returns by value. Safe to feed straight to `get_port`, which copies into a `std::string` to
  //! index `ports`. NOT safe as the return of `input_port_for`, which hands back a `string_view`
  //! that both the wiring descriptor and `next_port_info` retain — that view must point at the
  //! passthrough sink's own `_union_port_label`, which this helper initialises.
  [[nodiscard]] static std::string port_label(std::size_t arm_index)
  {
    return "union_" + std::to_string(arm_index);
  }

  std::string get_name() const override;

  //! Always a source: UNION drains its arms' port repositories and re-emits each batch.
  bool is_source() const override;

  //! `is_sink()` is deliberately not overridden. The base rule — sink iff the tree parent is a
  //! PARTITION or RIGHT_DELIM_JOIN — is already right: UNION is a sink exactly when it feeds a
  //! downstream shuffle (a UNION under a GROUP BY or a join) and a plain source otherwise.

  //! `NO_ORDER`, not `INSERTION_ORDER`. `order_preservation_recursive` stops at the first
  //! `is_source()` operator, so UNION's answer decides the whole plan's, and `INSERTION_ORDER`
  //! would have the plan claim an ordering a bag union explicitly does not provide. Same answer,
  //! same reason, as GROUPED_AGGREGATE and DELIM_JOIN.
  sirius::OrderPreservationType source_order() const override;

  //! The base throws for a non-sink operator with more than one child, so every multi-input
  //! operator supplies its own. This is the hash join's shape looped over N arms.
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  //! The base recurses into `children[0]` and throws when there is more than one child. Collect
  //! every arm's sources instead, so the debug plan verification pass accepts an N-ary node.
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
  //! it never needs a complete side. Under the base's FULL default, `get_next_task_hint` would
  //! refuse a task until *every* arm's pipeline finished, holding all arms' output in repositories
  //! — maximum peak memory and no producer/consumer overlap for an operator that only forwards.
  //! The UNION operator overrides the hint and does not read `port::type`, so this is the operator
  //! declaring the same contract it already implements rather than a behavior change.
  [[nodiscard]] MemoryBarrierType input_barrier_for(
    sirius_physical_operator const& producer) const override;

  //! Both task-driver methods must be overridden, for reasons independent of the port barrier.
  //!
  //! The base becomes READY only when *every* port has data and then pops one batch from *every*
  //! port into a single task. That is the right contract for a join, which consumes one input per
  //! side, and wrong for a fan-in whose arms are independent and rarely the same length: once the
  //! short arm drains and its pipeline finishes, the readiness test can never hold again and the
  //! long arm's remaining batches are stranded.
  //!
  //! UNION instead reports READY when *any* arm has a batch and pops one batch from one arm. The
  //! one-arm-per-task rule is also what keeps Phase-2 device placement: a task runs on one GPU, so
  //! bundling batches from several arms would force the cross-device migration this design exists
  //! to avoid.
  std::optional<task_creation_hint> get_next_task_hint() override;
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //! Pure forwarder: no device allocation beyond the batches already resident.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& /*stats*/) const override
  {
    return 0;
  }

 private:
  //! Arm ports in arm order, resolved once on first use. `get_port` allocates a `std::string` per
  //! lookup and `port_label` allocates another, and both task-driver methods run on every
  //! task-creation walk that reaches this operator, so the names are resolved to `port*` once and
  //! reused. Callers must hold `lock`.
  const std::vector<port*>& arm_ports();

  //! Arm to start the next pop at. Draining strictly from arm 0 would empty the first arm before
  //! touching the rest, which works against the executor's per-batch rotation across GPUs.
  //! Guarded by `lock`.
  std::size_t _arm_cursor = 0;

  //! Arm to start the next producer nomination at, so successive `WAITING_FOR_INPUT_DATA` hints do
  //! not keep naming the same arm. Kept separate from `_arm_cursor` so pop fairness and wait
  //! fairness do not perturb each other. Guarded by `lock`.
  std::size_t _wait_cursor = 0;

  //! Cache behind `arm_ports()`; empty until the wiring has been materialised.
  std::vector<port*> _arm_ports;
};

}  // namespace op
}  // namespace sirius
