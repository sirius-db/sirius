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

#pragma once

#include "duckdb/main/client_context_state.hpp"
#include "exec/batch_stream.hpp"
#include "exec/stream_session.hpp"
#include "op/sirius_physical_streaming_source.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace sirius::exec {

/// Schema + provenance for one input stream. Caller-supplied; never inferred.
struct stream_input_binding {
  std::vector<std::string> names;
  duckdb::vector<sirius::logical_type> types;
  std::shared_ptr<cucascade::shared_data_repository> repository;
  std::set<sender_id_t> expected_senders;

  /// Back-pointer into the engine-owned plan; filled during planning for session registration.
  op::sirius_physical_streaming_source* built = nullptr;
};

/// Per-connection declared input streams. ClientContextState so DuckDB bind can resolve schema
/// before physical planning. One fragment per connection at a time.
class stream_bind_catalog : public duckdb::ClientContextState {
 public:
  static constexpr const char* kStateKey = "sirius_stream_catalog";

  /// Overwrites any previous declaration for the same id.
  /// @throws sirius::invalid_input_exception on null repository, names/types size mismatch, or
  ///         empty names.
  void declare(stream_id_t id, stream_input_binding binding);

  void clear();

  [[nodiscard]] bool contains(stream_id_t id) const;

  /// @throws sirius::invalid_input_exception when `id` was never declared.
  [[nodiscard]] const stream_input_binding& get(stream_id_t id) const;

  /// @throws sirius::invalid_input_exception when `id` was never declared.
  void set_built(stream_id_t id, op::sirius_physical_streaming_source* built);

  [[nodiscard]] std::vector<stream_id_t> declared_streams() const;

 private:
  mutable std::mutex _mutex;
  std::map<stream_id_t, stream_input_binding> _entries;
};

}  // namespace sirius::exec
