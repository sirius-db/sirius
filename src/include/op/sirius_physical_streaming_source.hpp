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

#include "exec/stream_lifecycle.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cucascade/data/data_repository.hpp>

#include <memory>
#include <optional>
#include <set>

namespace sirius::op {

/// Source operator that marks the bottom boundary of a streaming plan fragment: producers push
/// `cucascade::data_batch`es into its input repository and the operator publishes them into the
/// pipeline, one batch per task.
///
/// The repository *is* the queue — batches sit in it, spillable by the downgrade executor, until
/// a task claims one. The `exec::stream_lifecycle` beside it owns everything the repository
/// lacks: the sender-aware end-of-stream, the WAITING-vs-EOS distinction, and the waker that
/// re-nominates a starved source.
class sirius_physical_streaming_source : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::STREAMING_SOURCE;

  /// @param input_repository The queue this source drains. Must not be null.
  /// @param expected_senders Every sender that must call `close_input` before the stream ends —
  ///        `{0}` for a single producer, `{0 … N-1}` for an N-way fan-in.
  /// @throws sirius::invalid_input_exception when `input_repository` is null.
  sirius_physical_streaming_source(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::shared_ptr<cucascade::shared_data_repository> input_repository,
    std::set<exec::sender_id_t> expected_senders);

  /// Wires the two pipeline-facing lifecycle hooks. Both fire on a producer thread, outside the
  /// lifecycle lock, and both weak-capture the pipeline rather than `this`:
  ///
  /// - end-of-stream → `update_pipeline_status(false)`, so a stream that ends with no task in
  ///   flight still finishes the pipeline *and* re-arms its downstream consumers;
  /// - waker → `task_creator::schedule(head)`, the live re-arm that wakes a starved source when
  ///   a batch arrives.
  void set_pipeline(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline) override;

  // -----------------------------------------------------------------------
  // Producer side — called by the session / wrapper, from any thread
  // -----------------------------------------------------------------------

  /// Register `batch` in the input repository and wake a starved source.
  /// @return false when the stream already ended and the batch was refused.
  bool push(std::shared_ptr<cucascade::data_batch> batch);

  /// Record that `sender` has finished producing. Idempotent per sender; the stream ends only
  /// once every expected sender has closed.
  /// @throws sirius::invalid_input_exception when `sender` is not an expected sender.
  void close_input(exec::sender_id_t sender);

  /// The stream's lifecycle, for the session and for diagnostics.
  [[nodiscard]] exec::stream_lifecycle& lifecycle() { return _lifecycle; }

  // -----------------------------------------------------------------------
  // Source interface
  // -----------------------------------------------------------------------

  bool is_source() const override { return true; }

  /// `READY{this}` when a batch is queued (open or ended); `WAITING{nullptr}` when open and
  /// empty — which also arms the one-shot waker; `nullopt` at end-of-stream.
  std::optional<task_creation_hint> get_next_task_hint() override;

  /// True only at end-of-stream: every expected sender closed AND the queue is empty. Drives
  /// both the task-creation guard and the port-less source pipeline-finish predicate.
  [[nodiscard]] bool all_ports_empty() override;

  /// Non-blocking: pop one batch and wrap it in a `pipelineable_operator_data`.
  /// Returns nullptr when nothing is queued.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  // -----------------------------------------------------------------------
  // Execution
  // -----------------------------------------------------------------------

  /// Pass-through: returns the input batches unchanged.
  std::unique_ptr<operator_data> execute(const operator_data& input,
                                         rmm::cuda_stream_view stream) override;

  /// Pass-through allocates nothing new; return input bytes so the reservation system
  /// does not over-reserve with the default 2× heuristic.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

 private:
  std::shared_ptr<cucascade::shared_data_repository> _input_repository;
  exec::stream_lifecycle _lifecycle;

  /// Re-installed on every `WAITING` hint (the waker is one-shot). Built once in
  /// `set_pipeline`; empty until then, which is why an operator driven without a pipeline —
  /// most unit tests — simply never re-arms.
  std::function<void()> _waker;
};

}  // namespace sirius::op
