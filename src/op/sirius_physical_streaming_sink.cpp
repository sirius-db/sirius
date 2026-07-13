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

#include "op/sirius_physical_streaming_sink.hpp"

#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <cucascade/data/data_batch.hpp>

#include <memory>
#include <vector>

namespace sirius::op {

sirius_physical_streaming_sink::sirius_physical_streaming_sink(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::shared_ptr<exec::exchange_channel> output_channel,
  std::shared_ptr<cucascade::shared_data_repository> output_repository)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::STREAMING_SINK, std::move(types), estimated_cardinality),
    _output_channel(std::move(output_channel)),
    _output_repository(std::move(output_repository))
{
  if (!_output_channel) {
    throw sirius::invalid_input_exception(
      "sirius_physical_streaming_sink: output_channel must not be null");
  }
  if (!_output_repository) {
    throw sirius::invalid_input_exception(
      "sirius_physical_streaming_sink: output_repository must not be null");
  }
}

void sirius_physical_streaming_sink::flush_pending_locked()
{
  while (!_pending.empty()) {
    if (!_output_channel->try_push(_pending.front())) break;
    _pending.pop_front();
  }
}

void sirius_physical_streaming_sink::try_flush_pending()
{
  std::lock_guard<std::mutex> lk(_pending_lock);
  flush_pending_locked();
  if (_pending.empty() && _close_when_flushed.load(std::memory_order_acquire)) {
    _output_channel->close();
  }
}

std::optional<task_creation_hint> sirius_physical_streaming_sink::get_next_task_hint()
{
  // Flush any backed-up handles first — cheap, handle-only, no GPU work.
  try_flush_pending();

  // Backpressure: if pending handles remain or the channel is full, no new sink tasks.
  {
    std::lock_guard<std::mutex> lk(_pending_lock);
    if (!_pending.empty()) {
      return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, nullptr};
    }
  }
  if (_output_channel->full()) {
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, nullptr};
  }

  std::lock_guard<std::mutex> lg(lock);
  auto it = ports.find(std::string(INPUT_PORT));
  if (it == ports.end() || !it->second || !it->second->repo) {
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, nullptr};
  }
  auto* port_ptr = it->second;

  if (port_ptr->repo->total_size() > 0) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }

  bool upstream_finished = port_ptr->src_pipeline && port_ptr->src_pipeline->is_pipeline_finished();
  if (upstream_finished) { return std::nullopt; }

  // Upstream still running; propagate the hint up the chain.
  // Null-check src_pipeline: hand-wired test ports carry nullptr.
  if (port_ptr->src_pipeline) {
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                              &(port_ptr->src_pipeline->get_operators()[0].get())};
  }
  return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, nullptr};
}

std::unique_ptr<operator_data> sirius_physical_streaming_sink::get_next_task_input_data()
{
  // Per-pull admission: the task-creation loop pulls without re-polling the hint, so we must
  // re-check here. Approximate admission is fine; races are absorbed by _pending.
  {
    std::lock_guard<std::mutex> lk(_pending_lock);
    if (!_pending.empty()) return nullptr;
  }
  if (_output_channel->full()) return nullptr;

  std::lock_guard<std::mutex> lg(lock);
  auto it = ports.find(std::string(INPUT_PORT));
  if (it == ports.end() || !it->second || !it->second->repo) return nullptr;
  auto* port_ptr = it->second;

  auto batch_ids = port_ptr->repo->get_batch_ids(0);
  if (batch_ids.empty()) return nullptr;

  auto batch = port_ptr->repo->pop_data_batch_by_id(batch_ids[0], 0);
  if (!batch) return nullptr;

  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(batch)});
}

std::unique_ptr<operator_data> sirius_physical_streaming_sink::execute(
  const operator_data& input, rmm::cuda_stream_view /*stream*/)
{
  // Pass-through: the sink holds no intermediate state; coalescing/splitting is deferred.
  const auto& pod = static_cast<const pipelineable_operator_data&>(input);
  return std::make_unique<pipelineable_operator_data>(pod.get_data_batches());
}

void sirius_physical_streaming_sink::sink(const operator_data& output_data,
                                          rmm::cuda_stream_view /*stream*/)
{
  const auto& pod     = static_cast<const pipelineable_operator_data&>(output_data);
  const auto& batches = pod.get_data_batches();

  for (const auto& batch : batches) {
    // Read size via a shared (read-only) lock — pure metadata access, no data copy or GPU work.
    // The lock is released before add_data_batch so the repo receives the batch in idle state.
    std::size_t size_bytes = 0;
    {
      auto ro = batch->to_read_only();
      if (const auto* d = ro.get_data()) { size_bytes = d->get_size_in_bytes(); }
    }

    // Register first: the repo becomes the owner of record and the batch is spill-visible
    // before any channel push, so _pending handles always point at accountable batches.
    _output_repository->add_data_batch(batch);

    exec::exchange_batch_handle handle{batch->get_batch_id(), size_bytes};

    // Drain the backlog first, then push the new handle — or park it if anything is still
    // queued or the channel is full. One lock hold keeps FIFO order intact and never blocks.
    std::lock_guard<std::mutex> lk(_pending_lock);
    flush_pending_locked();
    if (!_pending.empty() || !_output_channel->try_push(handle)) { _pending.push_back(handle); }
  }
}

std::size_t sirius_physical_streaming_sink::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  // Pass-through: no additional GPU allocation; return stats.bytes to avoid 2x over-reserving.
  return stats.bytes;
}

void sirius_physical_streaming_sink::on_finalize_operator()
{
  // Hold the lock across flush + close decision so they are atomic: close now if the backlog
  // drained, otherwise defer the close to whichever try_flush_pending() delivers the last handle.
  std::lock_guard<std::mutex> lk(_pending_lock);
  flush_pending_locked();
  if (_pending.empty()) {
    _output_channel->close();
  } else {
    _close_when_flushed.store(true, std::memory_order_release);
  }
}

}  // namespace sirius::op
