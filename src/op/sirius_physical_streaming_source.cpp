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
      SiriusPhysicalOperatorType::STREAMING_SOURCE, std::move(types), estimated_cardinality)
{
  if (!input_repository) {
    throw sirius::invalid_input_exception(
      "sirius_physical_streaming_source: input_repository must not be null");
  }
  _input =
    std::make_shared<exec::batch_stream>(std::move(input_repository), std::move(expected_senders));
}

void sirius_physical_streaming_source::set_pipeline(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline)
{
  sirius_physical_operator::set_pipeline(pipeline);

  // Weak: callbacks run on producer threads.
  duckdb::weak_ptr<pipeline::sirius_pipeline> weak_pipeline = pipeline;

  // Empty/late-closed stream finishes with no task in flight; original_pipeline=false re-arms
  // downstream consumers.
  _input->set_on_end_of_stream([weak_pipeline] {
    if (auto p = weak_pipeline.lock()) { p->update_pipeline_status(false); }
  });

  // Self-nomination (on_data → schedule(head)); schedule() only enqueues.
  _input->set_on_data([weak_pipeline] {
    auto p = weak_pipeline.lock();
    if (!p) { return; }
    auto* creator = p->get_task_creator();
    auto head     = p->get_source();
    if (creator && head) { creator->schedule(head.get()); }
  });
}

bool sirius_physical_streaming_source::push(std::shared_ptr<cucascade::data_batch> batch)
{
  return _input->push(std::move(batch));
}

void sirius_physical_streaming_source::close_input(exec::sender_id_t sender)
{
  _input->close(sender);
}

void sirius_physical_streaming_source::fail_input(std::exception_ptr error)
{
  _input->fail(std::move(error));
}

std::optional<task_creation_hint> sirius_physical_streaming_source::get_next_task_hint()
{
  switch (_input->classify()) {
    case exec::batch_stream::availability::END_OF_STREAM: return std::nullopt;
    case exec::batch_stream::availability::HAS_DATA:
      return task_creation_hint{TaskCreationHint::READY, this};
    case exec::batch_stream::availability::WAITING: break;
  }
  return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, nullptr};
}

bool sirius_physical_streaming_source::all_ports_empty() { return _input->drained(); }

std::unique_ptr<operator_data> sirius_physical_streaming_source::get_next_task_input_data()
{
  auto batch = _input->try_pull();
  if (!batch) return nullptr;

  std::vector<std::shared_ptr<cucascade::data_batch>> batches{std::move(batch)};
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::unique_ptr<operator_data> sirius_physical_streaming_source::execute(
  const operator_data& input, rmm::cuda_stream_view /*stream*/)
{
  const auto& pod = dynamic_cast<const pipelineable_operator_data&>(input);
  return std::make_unique<pipelineable_operator_data>(pod.get_data_batches());
}

std::size_t sirius_physical_streaming_source::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  // Pass-through: avoid default 2× reserve.
  return stats.bytes;
}

}  // namespace sirius::op
