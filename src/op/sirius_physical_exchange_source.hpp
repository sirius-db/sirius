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

#include "exchange/transport.hpp"
#include "op/sirius_physical_operator.hpp"

#include <memory>
#include <set>

namespace sirius::exchange {
class exchange_service;
}

namespace sirius::exec {
class query_lifecycle_registry;
}

namespace sirius::op {

/**
 * @brief The receiving half of an exchange: starts a pipeline with the batches every sending
 * rank sent on its stream.
 *
 * Its child is the stream's EXCHANGE_SINK on this rank. That sink's pipeline wires a port into
 * this operator that carries no data: it lets the task creator's hint walk reach this rank's own
 * producers while the source waits, and keeps this pipeline open until they finish.
 *
 * Unordered, each batch becomes a task as it arrives. Ordered, nothing is released until every
 * sender has ended; then everything goes out as one task, by sender rank and within a sender in
 * send order, which is what a gather after a range-partitioned sort needs.
 *
 * The stream ends once every expected sender has ended. A failed exchange surfaces as the
 * exception of the next task this source hands out.
 */
class sirius_physical_exchange_source : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::EXCHANGE_SOURCE;

  /// @param senders The ranks that send on @p stream: every rank, or none on a rank a gather
  ///        does not target.
  /// @param ordered Release by (rank, batch) once every sender has ended.
  sirius_physical_exchange_source(duckdb::vector<sirius::logical_type> types,
                                  std::size_t estimated_cardinality,
                                  std::shared_ptr<exchange::exchange_service> service,
                                  exchange::stream_id stream,
                                  std::set<exchange::rank_t> senders,
                                  bool ordered);
  ~sirius_physical_exchange_source() override;

  [[nodiscard]] exchange::stream_id stream() const noexcept { return _stream; }
  [[nodiscard]] bool ordered() const noexcept;

  /// Unordered arrivals land in this source's port repository, where they can be spilled, only
  /// while @p lifecycle grants this query work; without one they stay in a private queue.
  void set_query_lifecycle(exec::query_lifecycle_registry& lifecycle);

  /// Where to make the one empty batch handed on when the stream ends having brought nothing.
  void set_memory_space(cucascade::memory::memory_space& gpu);

  /// Arrivals schedule this pipeline; the last sender's end re-checks whether it finished.
  void set_pipeline(std::shared_ptr<pipeline::sirius_pipeline> pipeline) override;

  bool is_source() const override { return true; }

  /// Also ends its pipeline under MERGE_AGGREGATE / MERGE_TOP_N, which otherwise rely on their
  /// partial child being a sink: pipelined behind this source, a merge would run once per
  /// arriving batch instead of once over every rank's partials.
  bool is_sink() const override;

  sirius::OrderPreservationType source_order() const override;

  /// The sink's port is a dependency, not a barrier.
  [[nodiscard]] MemoryBarrierType input_barrier_for(
    sirius_physical_operator const& producer) const override;

  /// READY with a batch (or an error) to hand out; else WAITING on this rank's unfinished
  /// producer; else nothing until an arrival schedules it.
  [[nodiscard]] std::optional<task_creation_hint> get_next_task_hint() override;

  /// Nothing queued and nothing more to come from any sender. With the base check that this
  /// rank's producer pipeline finished, this is what ends the pipeline.
  [[nodiscard]] bool all_ports_empty() override;

  /// @throws the exchange's error.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input,
                                         ::cuda::stream_ref stream) override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

 private:
  struct state;
  struct receiver;

  /// Unordered: queues arrivals in the port's repository from now on. Under the state mutex.
  void bind_repository();

  std::shared_ptr<exchange::exchange_service> _service;
  exchange::stream_id _stream;
  std::shared_ptr<state> _state;
};

}  // namespace sirius::op
