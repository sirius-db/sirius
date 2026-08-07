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

#include "exec/batch_stream.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cucascade/data/data_repository.hpp>

#include <exception>
#include <memory>
#include <optional>
#include <set>

namespace sirius::op {

/// The plan leaf for a stream. A scan stands where the plan reads a table; this stands where
/// there is no table at all — a fragment boundary, fed by whatever another fragment or external
/// producer pushes in at runtime. Publishes one batch per task, in push order.
///
/// While a scan owns its input this source owns nothing and its senders are remote: "empty" means
/// wait, waiting must not block an engine thread, and the stream ends only once every expected
/// sender has closed.
///
/// The repository is the queue, spillable only if the caller registered it with the memory
/// manager. `exec::batch_stream` binds it to the stream state: who is still producing, whether
/// empty means wait or over, and how a producer failure reaches the consumer. The rest is
/// task-protocol glue over that one stream.
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

  /// Wires the two pipeline-facing stream hooks. Both fire on a producer thread, outside the
  /// stream's lock, and both weak-capture the pipeline rather than `this`:
  ///
  /// - end-of-stream → `update_pipeline_status(false)`, so a stream that ends with no task in
  ///   flight still finishes the pipeline *and* re-arms its downstream consumers;
  /// - on-data → `task_creator::schedule(head)`, this source's **self-nomination**.
  ///
  /// The self-nomination is what makes a stream-fed source work at all. The engine is
  /// pull-scheduled: a source that answers `WAITING_FOR_INPUT_DATA` is dropped, and the only
  /// built-in re-nomination is task completion — which a starved source never gets, because it
  /// has no task in flight to complete. The `on_data` hook *is* that missing re-nomination.
  /// Without it, a batch that arrives after the source was dropped is never looked at and the
  /// fragment hangs with data sitting in the repository. Firing from a producer thread is safe
  /// because `schedule()` only enqueues onto a thread-safe queue; it runs nothing inline.
  ///
  /// A stream that has *already* ended fires the first hook inside this call.
  void set_pipeline(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline) override;

  // -----------------------------------------------------------------------
  // Producer side — called by the session / wrapper, from any thread
  // -----------------------------------------------------------------------

  /// Register `batch` in the input repository, then fire `on_data`.
  /// @return false when the stream already ended and the batch was refused (S1). Dropping the
  ///         result silently loses the batch, so the caller must act on it.
  [[nodiscard]] bool push(std::shared_ptr<cucascade::data_batch> batch);

  /// Record that `sender` has finished producing cleanly. Idempotent per sender; the stream ends
  /// only once every expected sender has closed.
  /// @throws sirius::invalid_input_exception when `sender` is not an expected sender.
  void close_input(exec::sender_id_t sender);

  /// A producer failed: poison the input stream. It ends immediately, and the failure is
  /// rethrown out of `get_next_task_input_data()` — never a clean end-of-stream.
  /// @throws sirius::invalid_input_exception when `error` is null.
  void fail_input(std::exception_ptr error);

  /// The input stream, for the session and for diagnostics.
  [[nodiscard]] exec::batch_stream& stream() { return *_input; }

  // -----------------------------------------------------------------------
  // Source interface
  // -----------------------------------------------------------------------

  bool is_source() const override { return true; }

  /// `READY{this}` whenever the stream classifies as `HAS_DATA`: a batch is queued (open or
  /// ended), *or* a producer error is pending even over an empty queue (P4). That second case is
  /// the only route a failure has to the consumer — the source is nominated once more so the
  /// rethrow comes out of `get_next_task_input_data()` rather than the stream retiring quietly.
  /// `WAITING_FOR_INPUT_DATA{nullptr}` while the stream is open and empty; `nullopt` only at a
  /// *clean* end-of-stream.
  ///
  /// The `nullptr` producer is deliberate — there is no upstream operator for `task_creator` to
  /// redirect the request to.
  [[nodiscard]] std::optional<task_creation_hint> get_next_task_hint() override;

  /// True only at a clean end-of-stream: every expected sender closed, queue empty, no producer
  /// error pending. Drives the task-creation guard and the port-less pipeline-finish predicate,
  /// so an errored stream is never mistaken for a finished one — it ends by the rethrow out of
  /// `get_next_task_input_data()`.
  [[nodiscard]] bool all_ports_empty() override;

  /// Non-blocking: pop one batch and wrap it in a `pipelineable_operator_data`.
  /// Returns nullptr when nothing is queued.
  /// @throws the producer's error, if one is pending, ahead of anything still queued.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  // -----------------------------------------------------------------------
  // Execution
  // -----------------------------------------------------------------------

  /// Pass-through: the sender already materialized these batches; the task only routes them
  /// downstream.
  std::unique_ptr<operator_data> execute(const operator_data& input,
                                         rmm::cuda_stream_view stream) override;

  /// Pass-through allocates nothing new; return input bytes so the reservation system
  /// does not over-reserve with the default 2× heuristic.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

 private:
  /// Shared: producer threads co-own it, so the stream they push and close through outlives
  /// this operator exactly as the repository already did.
  std::shared_ptr<exec::batch_stream> _input;
};

}  // namespace sirius::op
