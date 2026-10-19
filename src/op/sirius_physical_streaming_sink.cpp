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

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>

#include <rmm/error.hpp>

#include <cucascade/data/data_batch.hpp>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <set>
#include <string>
#include <utility>

namespace sirius::op {

namespace {

/// A refused push means the output stream went terminal while this fragment was still
/// producing — either `close()` saw every sender finish, or a peer called `fail()`. Dropping
/// the batch would silently shorten the consumer's result, so poison the stream instead: the
/// consumer then learns about it from `try_pull()`'s rethrow rather than by noticing rows that
/// never arrived. `fail()` is first-failure-wins (P2), so this never displaces a real cause.
void poison_refused_output(exec::batch_stream& output, std::size_t index)
{
  output.fail(std::make_exception_ptr(sirius::internal_exception(
    "sirius_physical_streaming_sink: output stream " + std::to_string(index) +
    " refused a batch after end-of-stream; the fragment produced data after its output closed")));
}

/// SIRIUS_CN_EXCHANGE_BATCH_BYTES: bytes, or with a KiB/MiB/GiB suffix; 0 when unset or bad.
std::size_t combine_bytes_from_env()
{
  auto const* value = std::getenv("SIRIUS_CN_EXCHANGE_BATCH_BYTES");
  if (value == nullptr) { return 0; }
  char* end         = nullptr;
  auto const number = std::strtoull(value, &end, 10);
  std::string const unit{end};
  if (unit.empty() || unit == "B") { return number; }
  if (unit == "KiB") { return number << 10; }
  if (unit == "MiB") { return number << 20; }
  if (unit == "GiB") { return number << 30; }
  return 0;
}

}  // namespace

sirius_physical_streaming_sink::sirius_physical_streaming_sink(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::shared_ptr<cucascade::shared_data_repository> output_repository)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::STREAMING_SINK, std::move(types), estimated_cardinality),
    _combine_bytes(combine_bytes_from_env())
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
    _spec(std::move(spec)),
    _combine_bytes(combine_bytes_from_env())
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
  _pending.resize(_outputs.size());
  _pending_bytes.resize(_outputs.size(), 0);
}

std::unique_ptr<operator_data> sirius_physical_streaming_sink::execute(
  const operator_data& input_data, ::cuda::stream_ref /*stream*/)
{
  // Match RESULT_COLLECTOR: return batches for sink().
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_data_batches());
}

void sirius_physical_streaming_sink::sink(const operator_data& input_data,
                                          ::cuda::stream_ref stream)
{
  const auto& input = dynamic_cast<const pipelineable_operator_data&>(input_data);

  if (_outputs.size() == 1) {
    for (const auto& batch : input.get_data_batches()) {
      if (!_outputs[0]->push(batch)) { poison_refused_output(*_outputs[0], 0); }
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
        // clone()'s probe argument defaults to a no-op, so an omitted probe silently drops the
        // clone from quent. Bind the id and the probe together, as PARTITION does.
        const auto clone_batch_id = sirius::get_next_batch_id();
        auto copy                 = read_only[b].clone(
          clone_batch_id,
          stream,
          telemetry::quent_data_batch_probe::create(batch_telemetry(), clone_batch_id));
        if (!_outputs[i]->push(std::move(copy))) { poison_refused_output(*_outputs[i], i); }
      }
      if (!_outputs[0]->push(batches[b])) { poison_refused_output(*_outputs[0], 0); }
    }
    return;
  }

  const auto num_partitions = static_cast<int>(_outputs.size());
  // Partition everything first, publish after: an OOM here leaves nothing published, so the task
  // can retry the whole input (sink_publishes_on_success_only). Pieces bound hash_partition's
  // peak, ~3x its input, whatever feeds the sink: a scan batch or an aggregate's output.
  const auto& batches  = input.get_data_batches();
  const auto read_only = input.get_read_only_batches();
  std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> partitioned;
  auto const partition = [&](auto const& batch, cucascade::memory::memory_space& space) {
    // Same kernel as PARTITION; routing only.
    partitioned.push_back(gpu_partition_impl::hash_partition(batch,
                                                             _spec.key_columns,
                                                             _spec.key_cast_types,
                                                             num_partitions,
                                                             stream,
                                                             space,
                                                             batch_telemetry()));
  };
  for (std::size_t b = 0; b < read_only.size(); ++b) {
    const auto& input_batch = read_only[b];
    auto* space             = input_batch.get_memory_space();
    if (space == nullptr) {
      throw sirius::internal_exception(
        "sirius_physical_streaming_sink: partitioned sink requires a resident input batch");
    }
    const auto view   = sirius::get_cudf_table_view(input_batch);
    const auto rows   = static_cast<std::size_t>(view.num_rows());
    const auto bytes  = input_batch.get_data() ? input_batch.get_data()->get_size_in_bytes() : 0;
    const auto pieces = partition_pieces(rows, bytes, _max_piece_bytes);
    if (pieces.size() == 1) {
      partition(input_batch, *space);
      continue;
    }
    for (auto const& [begin, end] : pieces) {
      auto const slice =
        cudf::slice(view, {static_cast<cudf::size_type>(begin), static_cast<cudf::size_type>(end)})
          .front();
      // A view of the input batch, which the caller holds for the whole sink().
      auto piece =
        sirius::make_data_batch_from_view(slice, batches[b], 0, *space, stream, batch_telemetry());
      partition(piece->to_read_only(), *space);
    }
  }
  for (const auto& slices : partitioned) {
    for (std::size_t i = 0; i < slices.size(); ++i) {
      // Empty partition stays WAITING until finalize.
      if (sirius::get_cudf_table_view(*slices[i]).num_rows() == 0) { continue; }
      if (_combine_bytes == 0) {
        if (!_outputs[i]->push(slices[i])) { poison_refused_output(*_outputs[i], i); }
        continue;
      }
      std::vector<std::shared_ptr<cucascade::data_batch>> full;
      {
        std::lock_guard lock{_pending_mutex};
        _pending[i].push_back(slices[i]);
        _pending_bytes[i] += slices[i]->to_read_only().get_data()->get_size_in_bytes();
        if (_pending_bytes[i] < _combine_bytes) { continue; }
        full              = std::exchange(_pending[i], {});
        _pending_bytes[i] = 0;
      }
      publish_combined(i, std::move(full), stream);
    }
  }
}

void sirius_physical_streaming_sink::publish_combined(
  std::size_t index,
  std::vector<std::shared_ptr<cucascade::data_batch>> batches,
  ::cuda::stream_ref stream)
{
  std::shared_ptr<cucascade::data_batch> combined;
  if (batches.size() > 1) {
    try {
      std::vector<cucascade::read_only_data_batch> held;
      std::vector<cudf::table_view> views;
      for (auto const& batch : batches) {
        held.push_back(batch->to_read_only());
        // A slice another task produced was written on that task's stream.
        if (cudaEvent_t writer = held.back().get_data()->get_writer_event()) {
          RMM_CUDA_TRY(cudaStreamWaitEvent(stream.get(), writer, 0));
        }
        views.push_back(sirius::get_cudf_table_view(held.back()));
      }
      auto* space = held.front().get_memory_space();
      auto table  = cudf::concatenate(views, stream, space->get_default_allocator());
      combined    = sirius::make_data_batch(std::move(table), *space, stream, batch_telemetry());
      // The slices are freed on their own streams once released: the copy reads them first.
      stream.wait();
    } catch (rmm::out_of_memory const&) {
      combined.reset();
    }
  }
  if (combined) { batches = {std::move(combined)}; }
  for (auto& batch : batches) {
    if (!_outputs[index]->push(std::move(batch))) {
      poison_refused_output(*_outputs[index], index);
    }
  }
}

std::vector<std::pair<std::size_t, std::size_t>> sirius_physical_streaming_sink::partition_pieces(
  std::size_t rows, std::size_t bytes, std::size_t max_piece)
{
  if (rows <= 1 || bytes <= max_piece) { return {{0, rows}}; }
  auto const count = (bytes + max_piece - 1) / max_piece;
  auto const step  = std::max<std::size_t>(1, (rows + count - 1) / count);
  std::vector<std::pair<std::size_t, std::size_t>> out;
  for (std::size_t begin = 0; begin < rows; begin += step) {
    out.emplace_back(begin, std::min(rows, begin + step));
  }
  return out;
}

std::size_t sirius_physical_streaming_sink::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  // Single destination pushes the input handle through untouched.
  if (_outputs.size() == 1) { return 0; }
  // Broadcast holds the original plus one clone per extra destination, and each one stays
  // resident until its own repository drains — so the whole fan-out is live at once, not 2×.
  if (_spec.mode == partition_mode::broadcast) { return stats.bytes * _outputs.size(); }
  // Hash holds the slices of the whole input, plus hash_partition's reorder buffer for one piece at
  // a time.
  return stats.bytes + std::min<std::size_t>(stats.bytes, _max_piece_bytes);
}

void sirius_physical_streaming_sink::build_pipelines(pipeline::sirius_pipeline& current,
                                                     pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // children[] is producers, not outputs (Volcano/DuckDB orientation): a sink consumes exactly
  // one producing subtree.
  if (children.size() != 1) {
    throw sirius::internal_exception(
      "sirius_physical_streaming_sink: expects exactly one producer child subtree");
  }

  // operators[] membership required for finalize/EOS (like RESULT_COLLECTOR).
  auto& state = meta_pipeline.get_state();
  state.add_pipeline_operator(current, *this);

  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(*children[0]);
}

void sirius_physical_streaming_sink::on_finalize_operator()
{
  // What each destination still holds goes out before its end.
  std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> pending;
  {
    std::lock_guard lock{_pending_mutex};
    pending = std::exchange(
      _pending, std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>>(_outputs.size()));
    std::fill(_pending_bytes.begin(), _pending_bytes.end(), 0);
  }
  for (std::size_t i = 0; i < pending.size(); ++i) {
    if (pending[i].empty()) { continue; }
    auto* space = pending[i].front()->to_read_only().get_memory_space();
    publish_combined(i, std::move(pending[i]), space->acquire_stream());
  }
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
