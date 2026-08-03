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

#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <cucascade/data/data_batch.hpp>
#include <log/logging.hpp>

#include <string>
#include <utility>

namespace sirius::op {

sirius_physical_streaming_sink::sirius_physical_streaming_sink(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::shared_ptr<cucascade::shared_data_repository> output_repository)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::STREAMING_SINK, std::move(types), estimated_cardinality)
{
  if (!output_repository) {
    throw sirius::invalid_input_exception(
      "sirius_physical_streaming_sink: output_repository must not be null");
  }
  _outputs.push_back(std::make_shared<exec::batch_stream>(
    std::move(output_repository), std::set<exec::sender_id_t>{PIPELINE_SENDER}));
}

std::unique_ptr<operator_data> sirius_physical_streaming_sink::execute(
  const operator_data& input_data, rmm::cuda_stream_view /*stream*/)
{
  // As sirius_physical_result_collector does: hand the batches back so publish_output() can
  // deliver them to sink(). The base implementation would drop them.
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_read_only_batches());
}

void sirius_physical_streaming_sink::sink(const operator_data& input_data,
                                          rmm::cuda_stream_view /*stream*/)
{
  const auto& input = dynamic_cast<const pipelineable_operator_data&>(input_data);

  // Pushed in their current tier: no Arrow, no forced GPU upgrade, no copy. The batch stays
  // spillable in the repository until a consumer pulls it.
  for (const auto& batch : input.get_data_batches()) {
    // push() refuses once the stream is terminal. Ignoring that return silently drops the
    // batch, which surfaces as a fragment that "succeeds" with an empty output.
    if (!_outputs[0]->push(batch)) {
      SIRIUS_LOG_WARN(
        "sirius_physical_streaming_sink: batch refused after end-of-stream and dropped");
    }
  }
}

std::size_t sirius_physical_streaming_sink::no_history_peak_memory_estimate(
  const input_stats& /*stats*/) const
{
  // Pushing a handle into the output stream allocates nothing on top of the already-materialized
  // input, which is what this estimate measures. 0 is the "no additional peak" answer — reporting
  // the input bytes instead would raise the whole pipeline's cold-start reservation, since the
  // caller takes the max across operators.
  return 0;
}

void sirius_physical_streaming_sink::build_pipelines(pipeline::sirius_pipeline& current,
                                                     pipeline::sirius_meta_pipeline& meta_pipeline)
{
  if (children.size() != 1) {
    throw sirius::internal_exception(
      "sirius_physical_streaming_sink: expects exactly one child subtree");
  }

  // Append to `current` so the terminal operator is part of the root pipeline, then start the
  // real pipeline from the child. Same shape as RESULT_COLLECTOR.
  auto& state = meta_pipeline.get_state();
  state.add_pipeline_operator(current, *this);

  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(*children[0]);
}

void sirius_physical_streaming_sink::on_finalize_operator()
{
  // The pipeline is the single sender, so its completion is end-of-stream for all outputs.
  for (auto& s : _outputs) {
    s->close(PIPELINE_SENDER);
  }
}

void sirius_physical_streaming_sink::validate_index(std::size_t index) const
{
  if (index >= _outputs.size()) {
    throw sirius::invalid_input_exception("sirius_physical_streaming_sink: output stream index " +
                                          std::to_string(index) + " out of range (" +
                                          std::to_string(_outputs.size()) + " streams)");
  }
}

std::optional<std::shared_ptr<cucascade::data_batch>> sirius_physical_streaming_sink::pull(
  std::size_t index)
{
  validate_index(index);
  auto batch = _outputs[index]->try_pull();
  if (!batch) { return std::nullopt; }
  return batch;
}

void sirius_physical_streaming_sink::wait(std::size_t index)
{
  validate_index(index);
  _outputs[index]->wait();
}

bool sirius_physical_streaming_sink::drained(std::size_t index) const
{
  validate_index(index);
  return _outputs[index]->drained();
}

exec::batch_stream::availability sirius_physical_streaming_sink::availability(
  std::size_t index) const
{
  validate_index(index);
  return _outputs[index]->classify();
}

void sirius_physical_streaming_sink::fail_output(std::exception_ptr error)
{
  for (auto& s : _outputs) {
    s->fail(error);
  }
}

}  // namespace sirius::op
