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

#include "data/data_batch_utils.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "sirius/exception.hpp"
#include <log/logging.hpp>

#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cucascade/data/data_batch.hpp>

#include <string>
#include <utility>

namespace sirius::op {

sirius_physical_streaming_sink::sirius_physical_streaming_sink(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::shared_ptr<cucascade::shared_data_repository> output_repository)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::STREAMING_SINK, std::move(types), estimated_cardinality),
    _lifecycle(std::set<exec::sender_id_t>{PIPELINE_SENDER})
{
  if (!output_repository) {
    throw sirius::invalid_input_exception(
      "sirius_physical_streaming_sink: output_repository must not be null");
  }
  _output_repositories.push_back(std::move(output_repository));
}

sirius_physical_streaming_sink::sirius_physical_streaming_sink(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::vector<std::shared_ptr<cucascade::shared_data_repository>> output_repositories,
  partition_spec spec)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::STREAMING_SINK, std::move(types), estimated_cardinality),
    _output_repositories(std::move(output_repositories)),
    _spec(std::move(spec)),
    _lifecycle(std::set<exec::sender_id_t>{PIPELINE_SENDER})
{
  if (_output_repositories.empty()) {
    throw sirius::invalid_input_exception(
      "sirius_physical_streaming_sink: at least one output repository is required");
  }
  for (std::size_t i = 0; i < _output_repositories.size(); ++i) {
    if (!_output_repositories[i]) {
      throw sirius::invalid_input_exception("sirius_physical_streaming_sink: output repository " +
                                            std::to_string(i) + " must not be null");
    }
  }
  if (_output_repositories.size() > 1 && _spec.key_columns.empty()) {
    throw sirius::invalid_input_exception("sirius_physical_streaming_sink: a sink with " +
                                          std::to_string(_output_repositories.size()) +
                                          " destinations needs partition key columns to route by");
  }
}

std::unique_ptr<operator_data> sirius_physical_streaming_sink::execute(
  const operator_data& input_data, rmm::cuda_stream_view /*stream*/)
{
  // Mirrors sirius_physical_result_collector::execute: hand the chain's batches straight back so
  // publish_output() can deliver them to sink(). The base implementation would drop them.
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_read_only_batches());
}

void sirius_physical_streaming_sink::sink(const operator_data& input_data,
                                          rmm::cuda_stream_view stream)
{
  const auto& input = dynamic_cast<const pipelineable_operator_data&>(input_data);

  if (_output_repositories.size() == 1) {
    const auto& batches = input.get_data_batches();
    // Pushed in their current tier: no Arrow, no forced GPU upgrade, no copy. The batch stays
    // spillable in the repository until a consumer pulls it.
    for (const auto& batch : batches) {
      // admit() refuses once the stream is terminal. Ignoring that return silently drops the
      // batch, which surfaces as a fragment that "succeeds" with an empty output.
      if (!_lifecycle.admit([&] { _output_repositories[0]->add_data_batch(batch); })) {
        SIRIUS_LOG_WARN(
          "sirius_physical_streaming_sink: batch refused after end-of-stream and dropped");
      }
    }
    return;
  }

  const auto num_partitions = static_cast<int>(_output_repositories.size());
  for (const auto& input_batch : input.get_read_only_batches()) {
    auto* space = input_batch.get_memory_space();
    if (space == nullptr) {
      throw sirius::internal_exception(
        "sirius_physical_streaming_sink: partitioned sink requires a resident input batch");
    }

    // Same kernel the PARTITION operator uses — only the routing (slice i → repository i) is
    // new here. Any consistent hash co-locates equal keys, which is what a local, single-node
    // cut needs; reproducing StarRocks' exact partition function is translation's job.
    auto slices = gpu_partition_impl::hash_partition(input_batch,
                                                     _spec.key_columns,
                                                     _spec.key_cast_types,
                                                     num_partitions,
                                                     stream,
                                                     *space,
                                                     batch_telemetry());

    for (std::size_t i = 0; i < slices.size(); ++i) {
      // Skip empty slices rather than publish zero-row batches a consumer would have to pull
      // and discard. An empty partition simply stays WAITING until the pipeline finishes.
      if (sirius::get_cudf_table_view(*slices[i]).num_rows() == 0) { continue; }
      _lifecycle.admit([&] { _output_repositories[i]->add_data_batch(slices[i]); });
    }
  }
}

std::size_t sirius_physical_streaming_sink::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  // Pushing a handle into the output repository allocates nothing new.
  return stats.bytes;
}

void sirius_physical_streaming_sink::build_pipelines(pipeline::sirius_pipeline& current,
                                                     pipeline::sirius_meta_pipeline& meta_pipeline)
{
  if (children.size() != 1) {
    throw sirius::internal_exception(
      "sirius_physical_streaming_sink: expects exactly one child subtree");
  }

  // Same shape as the RESULT_COLLECTOR: append to `current` so the terminal operator is part of
  // the root pipeline, then start the real pipeline from the child.
  auto& state = meta_pipeline.get_state();
  state.add_pipeline_operator(current, *this);

  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(*children[0]);
}

void sirius_physical_streaming_sink::on_finalize_operator()
{
  // The pipeline is this stream's single sender, so its completion *is* end-of-stream. Until
  // this point a consumer that finds the repository empty must be told WAITING, never EOS.
  _lifecycle.mark_sender_done(PIPELINE_SENDER);
}

void sirius_physical_streaming_sink::validate_index(std::size_t index) const
{
  if (index >= _output_repositories.size()) {
    throw sirius::invalid_input_exception(
      "sirius_physical_streaming_sink: output stream index " + std::to_string(index) +
      " out of range (" + std::to_string(_output_repositories.size()) + " streams)");
  }
}

std::optional<std::shared_ptr<cucascade::data_batch>> sirius_physical_streaming_sink::pull(
  std::size_t index)
{
  validate_index(index);
  auto batch = _output_repositories[index]->pop_next_data_batch();
  if (!batch) { return std::nullopt; }
  return batch;
}

void sirius_physical_streaming_sink::wait(std::size_t index)
{
  validate_index(index);
  auto* repo = _output_repositories[index].get();
  _lifecycle.wait([repo] { return repo->all_empty(); });
}

bool sirius_physical_streaming_sink::drained(std::size_t index) const
{
  validate_index(index);
  // ANDed with *this* stream's emptiness: the terminal flag is shared across destinations, so a
  // partition with a backlog must not read as drained just because its siblings are.
  return _lifecycle.drained(_output_repositories[index]->all_empty());
}

exec::stream_lifecycle::availability sirius_physical_streaming_sink::availability(
  std::size_t index) const
{
  validate_index(index);
  return _lifecycle.classify(_output_repositories[index]->all_empty());
}

}  // namespace sirius::op
