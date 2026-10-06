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

void stream_bind_catalog::declare(stream_id_t id, stream_input_binding binding)
{
  if (binding.repository == nullptr) {
    throw sirius::invalid_input_exception("stream_bind_catalog: input stream " +
                                          std::to_string(id) +
                                          " must be declared with a repository");
  }
  if (binding.names.size() != binding.types.size()) {
    throw sirius::invalid_input_exception(
      "stream_bind_catalog: input stream " + std::to_string(id) + " declares " +
      std::to_string(binding.names.size()) + " column names but " +
      std::to_string(binding.types.size()) + " types");
  }
  if (binding.names.empty()) {
    throw sirius::invalid_input_exception("stream_bind_catalog: input stream " +
                                          std::to_string(id) + " must declare at least one column");
  }

  std::lock_guard<std::mutex> guard(_mutex);
  _entries[id] = std::move(binding);
}

void stream_bind_catalog::erase(stream_id_t id)
{
  std::lock_guard<std::mutex> guard(_mutex);
  _entries.erase(id);
}

bool stream_bind_catalog::contains(stream_id_t id) const
{
  std::lock_guard<std::mutex> guard(_mutex);
  return _entries.find(id) != _entries.end();
}

namespace {

[[noreturn]] void throw_undeclared(stream_id_t id)
{
  throw sirius::invalid_input_exception("stream_bind_catalog: no input stream declared with id " +
                                        std::to_string(id));
}

}  // namespace

const stream_input_binding& stream_bind_catalog::get(stream_id_t id) const
{
  std::lock_guard<std::mutex> guard(_mutex);
  auto it = _entries.find(id);
  if (it == _entries.end()) { throw_undeclared(id); }
  return it->second;
}

void stream_bind_catalog::set_built(stream_id_t id, op::sirius_physical_streaming_source* built)
{
  std::lock_guard<std::mutex> guard(_mutex);
  auto it = _entries.find(id);
  if (it == _entries.end()) { throw_undeclared(id); }
  if (it->second.built != nullptr) {
    // Each declared stream backs exactly one batch_stream; a second plan leaf reading the same
    // id (e.g. a self-join) would silently overwrite the first leaf's pointer here, so pushes
    // and closes would never reach it and its pipeline would wait forever.
    throw sirius::invalid_input_exception(
      "stream_bind_catalog: input stream " + std::to_string(id) +
      " is read by more than one operator in the same plan — fan-out reads of a single "
      "declared stream are not supported");
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

duckdb::shared_ptr<stream_bind_catalog> catalog_for(duckdb::ClientContext& context)
{
  auto catalog = context.registered_state->Get<stream_bind_catalog>(stream_bind_catalog::kStateKey);
  if (!catalog) {
    throw sirius::invalid_input_exception(
      "no stream catalog on this connection — the fragment must declare its input streams before "
      "the plan is bound");
  }
  return catalog;
}

}  // namespace sirius::exec
