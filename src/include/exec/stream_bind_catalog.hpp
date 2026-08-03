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
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace sirius::exec {

/// Everything needed to bind and then build one input stream.
///
/// The schema is supplied by the caller, never inferred: a stream has no file to probe, and the
/// front end already knows the column types from its descriptor table. `sirius_stream_source`'s
/// DuckDB bind reads `names` / `types` from here, and the physical plan generator reads
/// `repository` / `expected_senders` to construct the operator.
struct stream_input_binding {
  std::vector<std::string> names;
  duckdb::vector<sirius::logical_type> types;
  std::shared_ptr<cucascade::shared_data_repository> repository;
  std::set<sender_id_t> expected_senders;

  /// Back-pointer to the operator the plan generator built for this id, filled in during
  /// planning. The plan tree owns the operator; this is how the fragment finds it afterwards to
  /// register it with its `stream_session`. Null until `create_plan(LogicalGet&)` has run.
  op::sirius_physical_streaming_source* built = nullptr;
};

/// Per-connection registry of declared input streams, keyed by stream id.
///
/// It exists because a DuckDB table-function bind runs long before physical planning and has no
/// route back to the fragment being built — only a `ClientContext`. Registering the catalog as a
/// `ClientContextState` (exactly as the engine registers `SiriusContext` under `sirius_state`)
/// gives the bind a lookup path, so `sirius_stream_source(id)` can resolve a schema with no file
/// and no scan behind it.
///
/// One connection executes one fragment at a time, so a fragment declares its inputs before
/// planning and clears them afterwards.
class stream_bind_catalog : public duckdb::ClientContextState {
 public:
  /// ClientContextState key this catalog is registered under.
  static constexpr const char* kStateKey = "sirius_stream_catalog";

  /// Declare input stream `id`. Overwrites any previous declaration for the same id, so a reused
  /// connection cannot serve a stale schema.
  void declare(stream_id_t id, stream_input_binding binding);

  /// Drop every declaration. Called when a fragment finishes.
  void clear();

  [[nodiscard]] bool contains(stream_id_t id) const;

  /// @throws sirius::invalid_input_exception when `id` was never declared.
  [[nodiscard]] const stream_input_binding& get(stream_id_t id) const;

  /// Record the operator the plan generator built for `id`.
  /// @throws sirius::invalid_input_exception when `id` was never declared.
  void set_built(stream_id_t id, op::sirius_physical_streaming_source* built);

  /// Declared stream ids, ascending.
  [[nodiscard]] std::vector<stream_id_t> declared_streams() const;

 private:
  mutable std::mutex _mutex;
  std::map<stream_id_t, stream_input_binding> _entries;
};

}  // namespace sirius::exec
