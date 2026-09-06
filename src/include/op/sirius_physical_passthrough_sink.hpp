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

namespace sirius {
namespace op {

//! Terminates one arm of a UNION: a non-partitioning pipeline sink that forwards every batch
//! unchanged into the downstream UNION's per-arm input port. It replaces the join's
//! `PARTITION -> CONCAT` feeder chain, which a bag union needs neither half of.
//!
//! Two properties are load-bearing, and both come from *not* being a CONCAT. It emits plain
//! `pipelineable_operator_data`, so the batch carries no `partition_idx` and
//! `task_creator::create_task` selects a device by data locality rather than
//! `partition_idx % num_gpus` — each batch is consumed on the GPU its scan produced it on. A
//! single-partition CONCAT would instead pin every UNION task to GPU 0. And it pushes through the
//! base `sink()` rather than `push_data_batch_partitioned`, so the receiving UNION need not be a
//! `sirius_physical_partition_consumer_operator`.
//!
//! Each arm's sink owns the `"union_{i}"` port name it feeds: the consuming UNION's
//! `input_port_for` hands back a `string_view` into that member which the wiring descriptor and
//! `next_port_info` retain for the life of the query, so the storage has to live on a plan-tree
//! node.
//!
//! Deliberately *not* overridden, in both cases because the base is already right. `sink()` pushes
//! every batch of a `pipelineable_operator_data` to each `next_port_after_sink` via
//! `push_data_batch`, which is this operator's whole contract; CONCAT overrides it only to thread a
//! `partition_idx`. `get_next_task_hint` / `get_next_task_input_data` degenerate at arity 1 to the
//! right behavior, so the fan-in hazards that force UNION to override them cannot arise here.
class sirius_physical_passthrough_sink : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::PASSTHROUGH_SINK;

  //! @param types       Output schema; identical to the arm's own schema.
  //! @param port_label  The downstream port this arm feeds, `sirius_physical_union::port_label(i)`.
  explicit sirius_physical_passthrough_sink(duckdb::vector<sirius::logical_type> types,
                                            std::size_t estimated_cardinality,
                                            std::string port_label);

  std::string get_name() const override;

  //! Both, as CONCAT is: this operator produces its arm's tasks and terminates the arm pipeline.
  //! `is_source()` also satisfies `sirius_pipeline::reset_source` for an arm whose root is itself
  //! an unconditional sink, where this operator ends up alone in its pipeline.
  bool is_source() const override;
  bool is_sink() const override;

  //! Identity forward, overridden only because the base returns an empty batch vector.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //! The `"union_{i}"` label this arm feeds; read by `sirius_physical_union::input_port_for`.
  [[nodiscard]] const std::string& union_port_label() const noexcept { return _union_port_label; }

  //! Pure forwarder: no device allocation beyond the batches already resident.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& /*stats*/) const override
  {
    return 0;
  }

 private:
  //! Owns the storage behind the `string_view` in the wiring descriptor and in the upstream
  //! operator's `next_port_info`, both of which outlive plan conversion.
  std::string _union_port_label;
};

}  // namespace op
}  // namespace sirius
