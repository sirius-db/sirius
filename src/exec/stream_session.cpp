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

#include "sirius/exception.hpp"

#include <string>
#include <utility>

namespace sirius::exec {

void stream_session::add_source(stream_id_t id, op::sirius_physical_streaming_source& source)
{
  if (!_sources.emplace(id, &source).second) {
    throw sirius::invalid_input_exception("stream_session: input stream " + std::to_string(id) +
                                          " is already registered");
  }
}

void stream_session::add_sink(std::vector<stream_id_t> ids,
                              op::sirius_physical_streaming_sink& sink)
{
  if (ids.size() != sink.num_output_streams()) {
    throw sirius::invalid_input_exception(
      "stream_session: sink exposes " + std::to_string(sink.num_output_streams()) +
      " output streams but " + std::to_string(ids.size()) + " ids were given");
  }

  // Positional by contract: ids[i] ↔ sink partition i ↔ output repository i.
  for (std::size_t partition = 0; partition < ids.size(); ++partition) {
    if (!_sinks.emplace(ids[partition], sink_output{&sink, partition}).second) {
      throw sirius::invalid_input_exception("stream_session: output stream " +
                                            std::to_string(ids[partition]) +
                                            " is already registered");
    }
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

op::sirius_physical_streaming_source& stream_session::resolve_source(stream_id_t id) const
{
  auto it = _sources.find(id);
  if (it == _sources.end()) {
    throw sirius::invalid_input_exception("stream_session: no input stream with id " +
                                          std::to_string(id));
  }
  return *it->second;
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
  return resolve_source(id).push(std::move(batch));
}

void stream_session::close_input(stream_id_t id, sender_id_t sender)
{
  resolve_source(id).close_input(sender);
}

void stream_session::fail_input(stream_id_t id, std::exception_ptr error)
{
  resolve_source(id).fail_input(std::move(error));
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
