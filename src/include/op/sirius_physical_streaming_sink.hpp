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

#include <cstddef>
#include <memory>
#include <optional>

namespace sirius::op {

/// Terminal operator of a streaming fragment: every batch its pipeline produces is pushed into
/// an output `cucascade::shared_data_repository`, where an external consumer pulls it.
///
/// Batches are exposed **natively** — in whatever tier they currently sit — so nothing is
/// materialized or converted to Arrow on the way out, and a queued batch stays spillable by the
/// downgrade executor until it is pulled.
///
/// The sink is deliberately thin. It owns no parking buffer and no closing state machine: those
/// existed only to absorb a full bounded output channel, and there is no channel any more. It
/// overrides `sink()` (push), `on_finalize_operator()` (the pipeline is the single sender, so
/// finishing it is end-of-stream), and the pass-through memory estimate.
///
/// Unlike the source, the sink registers **no re-arm waker**: its consumer is an external
/// thread blocking in `wait()`, not an engine task that needs re-nominating.
class sirius_physical_streaming_sink : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::STREAMING_SINK;

  /// The pipeline feeding this sink is its one and only sender.
  static constexpr exec::sender_id_t PIPELINE_SENDER = 0;

  /// Single-destination sink: one output stream, no partitioning.
  ///
  /// @throws sirius::invalid_input_exception when `output_repository` is null.
  sirius_physical_streaming_sink(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::shared_ptr<cucascade::shared_data_repository> output_repository);

  bool is_sink() const override { return true; }

  // -----------------------------------------------------------------------
  // Producer side (engine)
  // -----------------------------------------------------------------------

  /// Push this task's output batches into the output repository, natively — no Arrow, no GPU
  /// upgrade, no copy. A push after end-of-stream is dropped by the lifecycle rather than
  /// landing behind a consumer that already saw EOS.
  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

  /// Pass-through: pushing a batch allocates nothing, so report the input bytes instead of the
  /// default 2× heuristic (matches the streaming source).
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  // -----------------------------------------------------------------------
  // Consumer side (external wrapper / session, never an engine task)
  // -----------------------------------------------------------------------

  /// Number of output streams this sink exposes. One per destination.
  [[nodiscard]] std::size_t num_output_streams() const { return _output_repositories.size(); }

  /// Non-blocking pull of the next batch of output stream `index`.
  /// @return nullopt when nothing is available right now — which is *not* the same as EOS; ask
  ///         `drained(index)` to tell the two apart.
  /// @throws sirius::invalid_input_exception when `index` is out of range.
  std::optional<std::shared_ptr<cucascade::data_batch>> pull(std::size_t index = 0);

  /// Block until output stream `index` has a batch to pull or the stream has ended.
  /// External threads only.
  void wait(std::size_t index = 0);

  /// True when the pipeline has finished AND output stream `index` is empty.
  [[nodiscard]] bool drained(std::size_t index = 0) const;

  /// Non-blocking classification of output stream `index`: HAS_DATA / WAITING / END_OF_STREAM.
  [[nodiscard]] exec::stream_lifecycle::availability availability(std::size_t index = 0) const;

 protected:
  /// The pipeline finishing is this stream's end-of-stream: it is the single expected sender.
  /// All output streams go terminal together.
  void on_finalize_operator() override;

  /// @throws sirius::invalid_input_exception when `index` is out of range.
  void validate_index(std::size_t index) const;

  /// One repository per destination. Output stream id, partition index and repository
  /// correspond positionally.
  std::vector<std::shared_ptr<cucascade::shared_data_repository>> _output_repositories;

  /// Shared by every output stream — the pipeline is one sender feeding all of them, so they
  /// reach end-of-stream together. Per-stream `drained()` / `wait()` still AND this terminal
  /// flag with that stream's own emptiness, so a slow destination stays distinguishable.
  exec::stream_lifecycle _lifecycle;
};

}  // namespace sirius::op
