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

#include "exec/exchange_channel.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cucascade/data/data_repository.hpp>

#include <deque>
#include <memory>
#include <mutex>
#include <optional>

namespace sirius::op {

/// Boundary operator (source + sink) that marks the top of a pipeline fragment. Pops batches
/// from its input port, registers each in the output repository (making it idle and spillable),
/// and pushes a lightweight handle into the exchange channel.
///
/// Key design invariants:
/// - sink() never blocks a worker: a full channel parks the handle in _pending and returns.
/// - _pending is flushed FIFO before any newer handle is pushed, preserving order.
/// - EOS: on_finalize_operator() flushes what fits and closes immediately, or sets
///   _close_when_flushed so the last try_flush_pending() call closes the channel.
/// - Backpressure is admission control: a full channel or non-empty _pending makes
///   get_next_task_hint() report WAITING, so no new sink tasks are created.
/// - _pending holds one handle per sink task created ahead of the consumer; admission control
///   stops new pulls while it is non-empty or the channel is full. Its depth is bounded only
///   once task creation is paced against completion — the session wiring's job (#839). Parked
///   batches stay registered and spill-visible.
class sirius_physical_streaming_sink : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::STREAMING_SINK;
  static constexpr std::string_view INPUT_PORT     = "input";

  /// Throws invalid_input_exception when output_channel or output_repository is null.
  sirius_physical_streaming_sink(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::shared_ptr<exec::exchange_channel> output_channel,
    std::shared_ptr<cucascade::shared_data_repository> output_repository);

  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }
  bool sink_order_dependent() const override { return false; }

  /// Flushes pending handles first; returns WAITING{nullptr} when the channel is full or
  /// handles remain pending, READY{this} when the input port has data, nullopt when the
  /// upstream pipeline is done and the port is empty, or WAITING{upstream source} otherwise.
  std::optional<task_creation_hint> get_next_task_hint() override;

  /// Per-pull admission check: returns nullptr when the channel has no free slot (races
  /// absorbed by _pending); otherwise pops one batch FIFO from the input port.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  /// Pass-through: returns the input batches unchanged.
  std::unique_ptr<operator_data> execute(const operator_data& input,
                                         rmm::cuda_stream_view stream) override;

  /// Zero-copy emit: registers each batch in the output repository, then pushes its handle
  /// to the channel. Handles that cannot be admitted are appended to _pending (never blocks).
  void sink(const operator_data& output, rmm::cuda_stream_view stream) override;

  /// Pass-through allocates nothing new; return input bytes to avoid over-reserving.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  /// Flush pending handles into the channel. Closes the channel once _pending empties and
  /// _close_when_flushed is set. Handle-only; callable from any thread.
  void try_flush_pending();

 protected:
  void on_finalize_operator() override;

 private:
  /// Push queued handles into the channel in FIFO order until it rejects one (full).
  /// Caller must hold _pending_lock. Does not close the channel — callers decide that.
  void flush_pending_locked();

  std::shared_ptr<exec::exchange_channel> _output_channel;
  std::shared_ptr<cucascade::shared_data_repository> _output_repository;
  std::mutex _pending_lock;
  std::deque<exec::exchange_batch_handle> _pending;
  bool _close_when_flushed{false};  // guarded by _pending_lock
  // Set by on_finalize_operator() under _pending_lock; sink() checks it under the same lock and
  // throws, so a late batch cannot strand behind the closed channel.
  bool _closing{false};
};

}  // namespace sirius::op
