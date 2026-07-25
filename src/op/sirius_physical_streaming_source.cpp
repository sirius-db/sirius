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

#include "op/sirius_physical_streaming_source.hpp"

#include "creator/task_creator.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <cucascade/data/data_batch.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace sirius::op {

sirius_physical_streaming_source::sirius_physical_streaming_source(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::shared_ptr<cucascade::shared_data_repository> input_repository,
  std::set<exec::sender_id_t> expected_senders)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::STREAMING_SOURCE, std::move(types), estimated_cardinality),
    _input_repository(std::move(input_repository)),
    _lifecycle(std::move(expected_senders))
{
  if (!_input_repository) {
    throw sirius::invalid_input_exception(
      "sirius_physical_streaming_source: input_repository must not be null");
  }
}

void sirius_physical_streaming_source::set_pipeline(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline)
{
  sirius_physical_operator::set_pipeline(pipeline);

  // Both callbacks weak-capture the pipeline. `_pipeline` above is already the strong reference
  // that keeps it alive for as long as this operator lives, so a second strong reference held
  // inside the operator's own lifecycle would only make ownership harder to reason about — and
  // lock() gives the callbacks a defined no-op if that ever stops holding. Mirrors the weak
  // discipline the on-close wiring has used since the channel-based source.
  duckdb::weak_ptr<pipeline::sirius_pipeline> weak_pipeline = pipeline;

  // End-of-stream is the only event that can finish this pipeline with no task in flight to
  // call mark_task_completed() -> update_pipeline_status() for it (an empty stream, or a close
  // arriving after the last task already completed). `original_pipeline=false` so
  // notify_downstream_pipelines() also schedules this pipeline's output consumers — with the
  // default (true) an empty or late-closed stream would finish this pipeline but never re-arm
  // its downstream. set_on_end_of_stream fires immediately if the stream already ended, which
  // covers a close that raced ahead of this wiring.
  _lifecycle.set_on_end_of_stream([weak_pipeline] {
    if (auto p = weak_pipeline.lock()) { p->update_pipeline_status(false); }
  });

  // The live re-arm. A head that answered WAITING{nullptr} is dropped by the task creator, and
  // the only built-in re-nomination is task completion — which a starved stream-fed source will
  // never see. schedule() only enqueues onto the thread-safe _task_creation_queue: it does not
  // re-enter the operator or take pipeline locks, so firing it from a producer thread is safe.
  // The head is resolved through the pipeline rather than captured, mirroring
  // notify_downstream_pipelines().
  _waker = [weak_pipeline] {
    auto p = weak_pipeline.lock();
    if (!p) { return; }
    auto* creator = p->get_task_creator();
    auto head     = p->get_source();
    if (creator && head) { creator->schedule(head.get()); }
  };
}

bool sirius_physical_streaming_source::push(std::shared_ptr<cucascade::data_batch> batch)
{
  // admit() runs the insert under the lifecycle lock and fires the armed waker afterwards, so
  // the batch is registered before anything can be scheduled for it, and nothing lands after EOS.
  return _lifecycle.admit([&] { _input_repository->add_data_batch(std::move(batch)); });
}

void sirius_physical_streaming_source::close_input(exec::sender_id_t sender)
{
  _lifecycle.mark_sender_done(sender);
}

std::optional<task_creation_hint> sirius_physical_streaming_source::get_next_task_hint()
{
  switch (_lifecycle.classify(_input_repository->all_empty())) {
    case exec::stream_lifecycle::availability::END_OF_STREAM: return std::nullopt;
    case exec::stream_lifecycle::availability::HAS_DATA:
      return task_creation_hint{TaskCreationHint::READY, this};
    case exec::stream_lifecycle::availability::WAITING: break;
  }

  // Starved: park, but arm the one-shot waker first so a later push re-schedules us. The
  // emptiness re-check happens under the lifecycle lock, which is the same lock push() holds
  // while it inserts — so a push that lands in this window either is seen by the predicate
  // (arm_waker returns false, and we are actually READY) or fires the waker we just armed.
  if (_waker && !_lifecycle.arm_waker(_waker, [this] { return _input_repository->all_empty(); })) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }
  return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, nullptr};
}

bool sirius_physical_streaming_source::all_ports_empty()
{
  return _lifecycle.drained(_input_repository->all_empty());
}

std::unique_ptr<operator_data> sirius_physical_streaming_source::get_next_task_input_data()
{
  auto batch = _input_repository->pop_next_data_batch();
  if (!batch) return nullptr;

  std::vector<std::shared_ptr<cucascade::data_batch>> batches{std::move(batch)};
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::unique_ptr<operator_data> sirius_physical_streaming_source::execute(
  const operator_data& input, rmm::cuda_stream_view /*stream*/)
{
  const auto& pod = static_cast<const pipelineable_operator_data&>(input);
  return std::make_unique<pipelineable_operator_data>(pod.get_data_batches());
}

std::size_t sirius_physical_streaming_source::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  // Pass-through: the input is already resident; no additional GPU allocation happens.
  // Return stats.bytes instead of the default 2× to avoid over-reserving under memory pressure.
  return stats.bytes;
}

}  // namespace sirius::op
