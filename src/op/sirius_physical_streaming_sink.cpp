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
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <cucascade/data/data_batch.hpp>
#include <log/logging.hpp>

#include <set>
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

sirius_physical_streaming_sink::sirius_physical_streaming_sink(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::vector<std::shared_ptr<cucascade::shared_data_repository>> output_repositories,
  partition_spec spec)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::STREAMING_SINK, std::move(types), estimated_cardinality),
    _spec(std::move(spec))
{
  if (output_repositories.empty()) {
    throw sirius::invalid_input_exception(
      "sirius_physical_streaming_sink: at least one output repository is required");
  }
  for (std::size_t i = 0; i < output_repositories.size(); ++i) {
    if (!output_repositories[i]) {
      throw sirius::invalid_input_exception("sirius_physical_streaming_sink: output repository " +
                                            std::to_string(i) + " must not be null");
    }
  }
  if (_spec.mode == partition_mode::hash && output_repositories.size() > 1 &&
      _spec.key_columns.empty()) {
    throw sirius::invalid_input_exception("sirius_physical_streaming_sink: hash mode with " +
                                          std::to_string(output_repositories.size()) +
                                          " destinations requires at least one key column");
  }
  if (_spec.mode == partition_mode::broadcast && !_spec.key_columns.empty()) {
    throw sirius::invalid_input_exception(
      "sirius_physical_streaming_sink: broadcast mode must have no key columns (routing is "
      "replication, not hashing)");
  }
  // Fail here: bad cast list / key index is device-side UB in hash_partition.
  if (!_spec.key_cast_types.empty() && _spec.key_cast_types.size() != _spec.key_columns.size()) {
    throw sirius::invalid_input_exception(
      "sirius_physical_streaming_sink: partition_spec has " +
      std::to_string(_spec.key_cast_types.size()) + " cast types for " +
      std::to_string(_spec.key_columns.size()) + " key columns; expected none or one per key");
  }
  // `this->types` — the ctor parameter of the same name has been moved from by now.
  const auto num_columns = this->types.size();
  for (auto key : _spec.key_columns) {
    if (key < 0 || static_cast<std::size_t>(key) >= num_columns) {
      throw sirius::invalid_input_exception(
        "sirius_physical_streaming_sink: partition key column " + std::to_string(key) +
        " is out of range for a " + std::to_string(num_columns) + "-column input");
    }
  }
  for (auto& repo : output_repositories) {
    _outputs.push_back(std::make_shared<exec::batch_stream>(
      std::move(repo), std::set<exec::sender_id_t>{PIPELINE_SENDER}));
  }
}

std::unique_ptr<operator_data> sirius_physical_streaming_sink::execute(
  const operator_data& input_data, rmm::cuda_stream_view /*stream*/)
{
  // Match RESULT_COLLECTOR: return batches for sink().
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_read_only_batches());
}

void sirius_physical_streaming_sink::sink(const operator_data& input_data,
                                          rmm::cuda_stream_view stream)
{
  const auto& input = dynamic_cast<const pipelineable_operator_data&>(input_data);

  if (_outputs.size() == 1) {
    for (const auto& batch : input.get_data_batches()) {
      // Refused push = silent data loss.
      if (!_outputs[0]->push(batch)) {
        SIRIUS_LOG_WARN(
          "sirius_physical_streaming_sink: batch refused after end-of-stream and dropped");
      }
    }
    return;
  }

  if (_spec.mode == partition_mode::broadcast) {
    // Output 0 gets the original handle (zero-copy); outputs 1..N-1 each get an independent
    // deep copy in the batch's current memory space so destinations cannot race over one handle's
    // residency. Clones first so output 0's push does not advance the stream's terminal state
    // before the copies are made.
    const auto& batches  = input.get_data_batches();
    const auto read_only = input.get_read_only_batches();
    for (std::size_t b = 0; b < batches.size(); ++b) {
      for (std::size_t i = 1; i < _outputs.size(); ++i) {
        auto copy = read_only[b].clone(sirius::get_next_batch_id(), stream);
        if (!_outputs[i]->push(std::move(copy))) {
          SIRIUS_LOG_WARN(
            "sirius_physical_streaming_sink: broadcast clone {} refused after end-of-stream and "
            "dropped",
            i);
        }
      }
      if (!_outputs[0]->push(batches[b])) {
        SIRIUS_LOG_WARN(
          "sirius_physical_streaming_sink: broadcast batch refused after end-of-stream and "
          "dropped");
      }
    }
    return;
  }

  const auto num_partitions = static_cast<int>(_outputs.size());
  for (const auto& input_batch : input.get_read_only_batches()) {
    auto* space = input_batch.get_memory_space();
    if (space == nullptr) {
      throw sirius::internal_exception(
        "sirius_physical_streaming_sink: partitioned sink requires a resident input batch");
    }

    // Same kernel as PARTITION; routing only.
    auto slices = gpu_partition_impl::hash_partition(input_batch,
                                                     _spec.key_columns,
                                                     _spec.key_cast_types,
                                                     num_partitions,
                                                     stream,
                                                     *space,
                                                     batch_telemetry());

    for (std::size_t i = 0; i < slices.size(); ++i) {
      // Empty partition stays WAITING until finalize.
      if (sirius::get_cudf_table_view(*slices[i]).num_rows() == 0) { continue; }
      // Refused push = silent data loss.
      if (!_outputs[i]->push(slices[i])) {
        SIRIUS_LOG_WARN(
          "sirius_physical_streaming_sink: partition {} batch refused after end-of-stream and "
          "dropped",
          i);
      }
    }
  }
}

std::size_t sirius_physical_streaming_sink::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  // 0 if N==1; ~2× when partitioned (hash_partition holds reorder + slices).
  if (_outputs.size() == 1) { return 0; }
  return stats.bytes * 2;
}

void sirius_physical_streaming_sink::build_pipelines(pipeline::sirius_pipeline& current,
                                                     pipeline::sirius_meta_pipeline& meta_pipeline)
{
  if (children.size() != 1) {
    throw sirius::internal_exception(
      "sirius_physical_streaming_sink: expects exactly one child subtree");
  }

  // operators[] membership required for finalize/EOS (like RESULT_COLLECTOR).
  auto& state = meta_pipeline.get_state();
  state.add_pipeline_operator(current, *this);

  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(*children[0]);
}

void sirius_physical_streaming_sink::on_finalize_operator()
{
  // Pipeline completion = sender-set EOS (PIPELINE_SENDER).
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
