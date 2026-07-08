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

#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <cucascade/data/data_batch.hpp>

#include <memory>
#include <vector>

namespace sirius::op {

sirius_physical_streaming_source::sirius_physical_streaming_source(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::shared_ptr<exec::exchange_channel> input_channel,
  std::shared_ptr<cucascade::shared_data_repository> input_repository)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::STREAMING_SOURCE, std::move(types), estimated_cardinality),
    _input_channel(std::move(input_channel)),
    _input_repository(std::move(input_repository))
{
  // Close is the only event that can make this pipeline finish without a task in flight to
  // call mark_task_completed() -> update_pipeline_status() for it (empty stream, or the last
  // task already completed while the channel was still open). Wire close directly to a
  // re-evaluation instead of relying on task_creator to notice a dropped nullopt hint — see
  // docs/super-sirius/streaming-source-p1-fix-plan-v2-no-task-creator.md, Finding 1.
  //
  // get_pipeline() is resolved lazily (at callback-fire time, not here at construction time):
  // set_pipeline() runs once per operator during query::build_indices(), which happens after
  // every operator is constructed but before any channel activity starts in production, so by
  // the time close() is actually called the pipeline is always set. The null check makes this
  // safe even if that ordering assumption is ever violated (e.g. in tests that never wire a
  // pipeline at all).
  _input_channel->set_on_close([this] {
    if (auto pipeline = get_pipeline()) { pipeline->update_pipeline_status(); }
  });
}

sirius_physical_streaming_source::~sirius_physical_streaming_source()
{
  // The channel may outlive this operator (e.g. held by a producer on another thread); clear
  // the callback before `this` becomes dangling, per exchange_channel's documented contract.
  _input_channel->set_on_close(nullptr);
}

std::optional<task_creation_hint> sirius_physical_streaming_source::get_next_task_hint()
{
  // Terminal EOS: channel closed and fully drained.
  if (_input_channel->drained()) return std::nullopt;

  // Data available (open or closed-but-non-empty): schedule a task.
  if (!_input_channel->empty()) { return task_creation_hint{TaskCreationHint::READY, this}; }

  // Open but empty: drop the request. A future push must re-schedule via the session (#839).
  return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, nullptr};
}

bool sirius_physical_streaming_source::all_ports_empty() { return _input_channel->drained(); }

std::unique_ptr<operator_data> sirius_physical_streaming_source::get_next_task_input_data()
{
  auto maybe_handle = _input_channel->try_pop();
  if (!maybe_handle) return nullptr;

  auto batch = _input_repository->pop_data_batch_by_id(maybe_handle->batch_id);
  if (!batch) {
    throw sirius::internal_exception(
      "sirius_physical_streaming_source: batch_id " + std::to_string(maybe_handle->batch_id) +
      " not found in input repository — producer must register before pushing handle");
  }

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
