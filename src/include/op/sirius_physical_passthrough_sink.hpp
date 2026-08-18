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
//! unchanged into the downstream UNION's per-arm input port. It is the UNION counterpart of the
//! join's `PARTITION -> CONCAT` feeder chain, collapsed to a single operator because a bag union
//! needs neither a shuffle nor a coalesce.
//!
//! Two properties are load-bearing, and both come from *not* being a CONCAT:
//!   - It emits plain `pipelineable_operator_data`, so the batch carries no `partition_idx`. Task
//!     device selection therefore falls through to data locality instead of
//!     `partition_idx % num_gpus`, and each batch is consumed on the GPU its scan produced it on
//!     (`task_creator::create_task`). A single-partition CONCAT would pin every UNION task to GPU
//!     0.
//!   - It pushes through the base `sink()`, not `push_data_batch_partitioned`, so the receiving
//!     UNION need not be a `sirius_physical_partition_consumer_operator`.
//!
//! Each arm's sink owns the `"union_{i}"` port name it feeds. The consuming UNION's
//! `input_port_for` hands back a
//! `string_view` into that member and both the wiring descriptor and `next_port_info` retain the
//! view for the life of the query, so the storage has to live on a plan-tree node.
class sirius_physical_passthrough_sink : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::PASSTHROUGH_SINK;

  //! @param types       Output schema; identical to the arm's own schema (nothing is reshaped).
  //! @param port_label  The downstream port this arm feeds, `sirius_physical_union::port_label(i)`.
  sirius_physical_passthrough_sink(duckdb::vector<sirius::logical_type> types,
                                   std::size_t estimated_cardinality,
                                   std::string port_label);

  std::string get_name() const override;

  //! Mirrors CONCAT's dual role: it both produces its arm's tasks and terminates the arm pipeline.
  //! `is_source()` also satisfies `sirius_pipeline::reset_source` for the arm shapes where this
  //! sink ends up alone in its pipeline (an arm whose root is itself an unconditional sink, e.g.
  //! an aggregate); in the common scan-rooted arm it is the last operator of the scan's pipeline.
  bool is_source() const override;
  bool is_sink() const override;

  //! Identity forward. The only reason to override `execute` at all is that the base returns an
  //! empty batch vector.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //! No `sink()` override: the base already pushes every batch of a `pipelineable_operator_data`
  //! to each `next_port_after_sink` via `push_data_batch`, which is exactly this operator's
  //! contract. Contrast CONCAT, which overrides it only to thread a `partition_idx`.

  //! No `get_next_task_hint` / `get_next_task_input_data` overrides either. This sink is
  //! single-input, so the base's all-ports readiness test and one-batch-per-port pop both
  //! degenerate to the correct behavior; the fan-in hazards that force UNION to override them
  //! cannot arise at arity 1.

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
