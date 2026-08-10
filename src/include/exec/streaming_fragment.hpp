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

#include "exec/stream_bind_catalog.hpp"
#include "exec/stream_session.hpp"
#include "op/sirius_physical_streaming_sink.hpp"
#include "query_id.hpp"

#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sirius {
class sirius_engine;
class sirius_interface;
}  // namespace sirius

namespace sirius::exec {

struct stream_input_spec {
  std::vector<std::string> names;
  duckdb::vector<sirius::logical_type> types;
  /// Sender-set EOS: stream ends only once all have closed.
  std::set<sender_id_t> expected_senders;
};

/// Bound, optimized DuckDB logical plan (Substrait bytes, SQL, …).
using logical_plan_source =
  std::function<duckdb::unique_ptr<duckdb::LogicalOperator>(duckdb::ClientContext&)>;

struct fragment_spec {
  logical_plan_source plan_source;
  std::map<stream_id_t, stream_input_spec> inputs;
  /// Positional: outputs[i] addresses partition i.
  std::vector<stream_id_t> outputs;
  /// Absent = gather (single destination, no partitioning).
  std::optional<op::partition_spec> partitioning;
};

/// Owns repos/engine/session for one fragment.
/// Repositories escape data_repository_manager_ cleanup (survive the query window).
/// Engine owns the plan so the sink stays pullable after run().
class streaming_fragment {
 public:
  /// Validates the spec and creates one repository per declared stream.
  /// @throws sirius::invalid_input_exception when plan_source is unset, outputs empty, N>1
  ///         without partitioning, or on a duplicate output id.
  streaming_fragment(duckdb::ClientContext& context, fragment_spec spec);

  /// Clears this fragment's declarations from the connection's stream_bind_catalog.
  ~streaming_fragment();

  streaming_fragment(const streaming_fragment&)            = delete;
  streaming_fragment& operator=(const streaming_fragment&) = delete;

  /// Declare inputs, lower to STREAMING_SOURCE/SINK, register with session. Separate from
  /// run() so callers can push first.
  /// @throws sirius::invalid_input_exception when already built, no catalog, null plan, or a
  ///         declared input the plan never reads.
  /// @throws whatever the plan source, binder, or plan generator raises.
  void build(sirius::query_id_t query_id);

  /// Submit and block. Shared query window (don't open a second StandaloneQueryScope between
  /// build and run).
  /// @throws sirius::invalid_input_exception when build() has not run.
  /// @throws whatever the engine's execution raises.
  void run();

  [[nodiscard]] stream_session& session() { return _session; }

  [[nodiscard]] sirius::sirius_engine& engine() { return *_engine; }

  /// Physical output column types of the plan root (set during build()).
  /// Used by relay steps to validate schema agreement before any data moves.
  /// @throws sirius::invalid_input_exception when build() has not run.
  [[nodiscard]] const duckdb::vector<sirius::logical_type>& sink_types() const;

  /// @throws sirius::invalid_input_exception when `id` is not a declared input stream.
  [[nodiscard]] const std::shared_ptr<cucascade::shared_data_repository>& input_repository(
    stream_id_t id) const;

  /// @throws sirius::invalid_input_exception when `id` is not a declared output stream.
  [[nodiscard]] const std::shared_ptr<cucascade::shared_data_repository>& output_repository(
    stream_id_t id) const;

 private:
  duckdb::ClientContext& _context;
  fragment_spec _spec;

  // Declaration order IS the lifetime contract (destroyed in reverse): repositories outlive
  // the engine, the engine owns the plan, and the session (borrowing operators) is torn down first.
  std::map<stream_id_t, std::shared_ptr<cucascade::shared_data_repository>> _input_repos;
  std::map<stream_id_t, std::shared_ptr<cucascade::shared_data_repository>> _output_repos;
  std::unique_ptr<sirius::sirius_interface> _iface;
  std::unique_ptr<sirius::sirius_engine> _engine;
  stream_session _session;

  bool _built{false};
  duckdb::vector<sirius::logical_type> _sink_types;
};

}  // namespace sirius::exec
