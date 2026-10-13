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

#include "exec/stream_session.hpp"

#include "data/data_batch_utils.hpp"
#include "sirius/exception.hpp"
#include "telemetry/data_batch_probe.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_device.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <utility>

namespace sirius::exec {

void stream_session::add_source(stream_id_t id, op::sirius_physical_streaming_source& source)
{
  add_sources(id, {&source});
}

void stream_session::add_sources(stream_id_t id,
                                 const std::vector<op::sirius_physical_streaming_source*>& sources)
{
  if (_sources.find(id) != _sources.end()) {
    throw sirius::invalid_input_exception("stream_session: input stream " + std::to_string(id) +
                                          " is already registered");
  }
  if (sources.empty()) {
    throw sirius::invalid_input_exception("stream_session: input stream " + std::to_string(id) +
                                          " has no source");
  }
  std::set<op::sirius_physical_streaming_source*> seen;
  for (auto* source : sources) {
    if (source == nullptr || !seen.insert(source).second) {
      throw sirius::invalid_input_exception("stream_session: input stream " + std::to_string(id) +
                                            " lists a null or repeated source");
    }
  }
  _sources.emplace(id, sources);
}

void stream_session::add_sink(std::vector<stream_id_t> ids,
                              op::sirius_physical_streaming_sink& sink)
{
  if (ids.size() != sink.num_output_streams()) {
    throw sirius::invalid_input_exception(
      "stream_session: sink exposes " + std::to_string(sink.num_output_streams()) +
      " output streams but " + std::to_string(ids.size()) + " ids were given");
  }

  // Validate every id before inserting any. Throwing mid-insert would leave the partitions
  // already registered behind in the map, so a caller that catches and retries would then trip
  // over its own leftovers as spurious duplicates.
  std::set<stream_id_t> incoming;
  for (auto id : ids) {
    if (_sinks.find(id) != _sinks.end()) {
      throw sirius::invalid_input_exception("stream_session: output stream " + std::to_string(id) +
                                            " is already registered");
    }
    if (!incoming.insert(id).second) {
      throw sirius::invalid_input_exception("stream_session: output stream " + std::to_string(id) +
                                            " appears more than once in the same registration");
    }
  }

  // Positional by contract: ids[i] ↔ sink partition i ↔ output repository i.
  for (std::size_t partition = 0; partition < ids.size(); ++partition) {
    _sinks.emplace(ids[partition], sink_output{&sink, partition});
  }
}

std::vector<stream_id_t> stream_session::input_streams() const
{
  std::vector<stream_id_t> ids;
  ids.reserve(_sources.size());
  for (const auto& [id, _] : _sources) {
    ids.push_back(id);
  }
  return ids;
}

std::vector<stream_id_t> stream_session::output_streams() const
{
  std::vector<stream_id_t> ids;
  ids.reserve(_sinks.size());
  for (const auto& [id, _] : _sinks) {
    ids.push_back(id);
  }
  return ids;
}

const std::vector<op::sirius_physical_streaming_source*>& stream_session::resolve_sources(
  stream_id_t id) const
{
  auto it = _sources.find(id);
  if (it == _sources.end()) {
    throw sirius::invalid_input_exception("stream_session: no input stream with id " +
                                          std::to_string(id));
  }
  return it->second;
}

const stream_session::sink_output& stream_session::resolve_sink(stream_id_t id) const
{
  auto it = _sinks.find(id);
  if (it == _sinks.end()) {
    throw sirius::invalid_input_exception("stream_session: no output stream with id " +
                                          std::to_string(id));
  }
  return it->second;
}

bool stream_session::push(stream_id_t id, std::shared_ptr<cucascade::data_batch> batch)
{
  const auto& sources = resolve_sources(id);
  if (sources.size() == 1) { return sources.front()->push(std::move(batch)); }

  // Fan-out: leaves 1..N-1 each get an independent deep copy in the batch's current memory space
  // (as a broadcast sink does), made before leaf 0 receives the original handle so its pipeline
  // cannot touch the batch while the copies are read.
  std::vector<std::shared_ptr<cucascade::data_batch>> copies;
  if (batch != nullptr) {
    copies.reserve(sources.size() - 1);
    auto read_only = batch->to_read_only();
    auto* space    = read_only.get_memory_space();
    std::optional<rmm::cuda_set_device_raii> device_guard;
    if (space != nullptr && space->get_tier() == cucascade::memory::Tier::GPU) {
      device_guard.emplace(rmm::cuda_device_id{space->get_device_id()});
    }
    auto stream = cudf::get_default_stream();
    for (std::size_t i = 1; i < sources.size(); ++i) {
      const auto clone_batch_id = sirius::get_next_batch_id();
      copies.push_back(read_only.clone(
        clone_batch_id,
        stream,
        telemetry::quent_data_batch_probe::create(sources[i]->batch_telemetry(), clone_batch_id)));
    }
    // The leaves run on their own streams; the copies must be complete before any reads them.
    stream.synchronize();
  }
  bool accepted = true;
  for (std::size_t i = 1; i < sources.size(); ++i) {
    accepted = sources[i]->push(batch != nullptr ? std::move(copies[i - 1]) : nullptr) && accepted;
  }
  return sources.front()->push(std::move(batch)) && accepted;
}

void stream_session::close_input(stream_id_t id, sender_id_t sender)
{
  for (auto* source : resolve_sources(id)) {
    source->close_input(sender);
  }
}

bool stream_session::input_closed(stream_id_t id) const
{
  const auto& sources = resolve_sources(id);
  return std::all_of(
    sources.begin(), sources.end(), [](auto* source) { return source->stream().terminal(); });
}

bool stream_session::input_sender_closed(stream_id_t id, sender_id_t sender) const
{
  const auto& sources = resolve_sources(id);
  return std::all_of(sources.begin(), sources.end(), [&](auto* source) {
    const auto& stream = source->stream();
    return stream.terminal() || stream.sender_closed(sender);
  });
}

std::optional<std::shared_ptr<cucascade::data_batch>> stream_session::pull(stream_id_t id)
{
  const auto& out = resolve_sink(id);
  return out.sink->pull(out.partition);
}

void stream_session::wait(stream_id_t id)
{
  const auto& out = resolve_sink(id);
  out.sink->wait(out.partition);
}

bool stream_session::drained(stream_id_t id) const
{
  const auto& out = resolve_sink(id);
  return out.sink->drained(out.partition);
}

void stream_session::fail_output(stream_id_t id, std::exception_ptr error)
{
  // fail_output poisons all partitions of the sink (one pipeline = one sender).
  resolve_sink(id).sink->fail_output(std::move(error));
}

}  // namespace sirius::exec
