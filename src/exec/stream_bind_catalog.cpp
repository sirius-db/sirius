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

#include "exec/stream_bind_catalog.hpp"

#include "sirius/exception.hpp"

#include <string>
#include <utility>

namespace sirius::exec {

namespace {

std::string id_str(stream_id_t id) { return std::to_string(id); }

}  // namespace

void stream_bind_catalog::declare(stream_id_t id, stream_input_binding binding)
{
  if (binding.repository == nullptr) {
    throw sirius::invalid_input_exception("stream_bind_catalog: input stream " + id_str(id) +
                                          " must be declared with a repository");
  }
  if (binding.names.size() != binding.types.size()) {
    throw sirius::invalid_input_exception("stream_bind_catalog: input stream " + id_str(id) +
                                          " declares " + std::to_string(binding.names.size()) +
                                          " column names but " +
                                          std::to_string(binding.types.size()) + " types");
  }
  if (binding.names.empty()) {
    throw sirius::invalid_input_exception("stream_bind_catalog: input stream " + id_str(id) +
                                          " must declare at least one column");
  }

  std::lock_guard<std::mutex> guard(_mutex);
  _entries[id] = std::move(binding);
}

void stream_bind_catalog::clear()
{
  std::lock_guard<std::mutex> guard(_mutex);
  _entries.clear();
}

bool stream_bind_catalog::contains(stream_id_t id) const
{
  std::lock_guard<std::mutex> guard(_mutex);
  return _entries.find(id) != _entries.end();
}

const stream_input_binding& stream_bind_catalog::get(stream_id_t id) const
{
  std::lock_guard<std::mutex> guard(_mutex);
  auto it = _entries.find(id);
  if (it == _entries.end()) {
    throw sirius::invalid_input_exception("stream_bind_catalog: no input stream declared with id " +
                                          id_str(id));
  }
  return it->second;
}

void stream_bind_catalog::set_built(stream_id_t id, op::sirius_physical_streaming_source* built)
{
  std::lock_guard<std::mutex> guard(_mutex);
  auto it = _entries.find(id);
  if (it == _entries.end()) {
    throw sirius::invalid_input_exception("stream_bind_catalog: no input stream declared with id " +
                                          id_str(id));
  }
  it->second.built = built;
}

std::vector<stream_id_t> stream_bind_catalog::declared_streams() const
{
  std::lock_guard<std::mutex> guard(_mutex);
  std::vector<stream_id_t> ids;
  ids.reserve(_entries.size());
  for (const auto& [id, _] : _entries) {
    ids.push_back(id);
  }
  return ids;
}

}  // namespace sirius::exec
