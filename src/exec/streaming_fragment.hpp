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

#include <duckdb/main/client_context.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace duckdb {
class QueryResult;
}  // namespace duckdb

namespace sirius {
class sirius_engine;
class sirius_interface;
class sirius_prepared_statement_data;
}  // namespace sirius

namespace sirius::exec {

struct stream_input_spec {
  std::vector<std::string> names;
  duckdb::vector<sirius::logical_type> types;
  /// Sender-set EOS: stream ends only once all have closed.
  std::set<sender_id_t> expected_senders;
};

/// Bound, optimized DuckDB logical plan from Substrait bytes, SQL, or similar.
using logical_plan_source =
  std::function<duckdb::unique_ptr<duckdb::LogicalOperator>(duckdb::ClientContext&)>;

struct fragment_spec {
  logical_plan_source plan_source;
  std::map<stream_id_t, stream_input_spec> inputs;
  /// Positional: outputs[i] addresses partition i. Empty = RESULT_COLLECTOR terminal.
  std::vector<stream_id_t> outputs;
  /// Absent = gather (single destination, no partitioning). Illegal when outputs.size() < 2.
  std::optional<op::partition_spec> partitioning;
  /// Optional DuckDB prepared metadata for a RESULT_COLLECTOR terminal (column names and types).
  /// When unset, build() synthesizes names (`col_0`, `col_1`, ...) from the physical plan types.
  duckdb::shared_ptr<duckdb::PreparedStatementData> prepared;
};

/// Owns repositories, engine, session, and the query window for one fragment.
/// Repositories outlive data_repository_manager_ cleanup, so parked batches survive run().
/// The engine owns the plan, so the sink stays pullable after run().
class streaming_fragment {
 public:
  /// Validates the spec and creates one repository per declared stream.
  /// @throws sirius::invalid_input_exception when plan_source is unset, N>1 outputs have no
  ///         partitioning, partitioning is set on fewer than two outputs, or on a duplicate
  ///         output id. Empty outputs (a result fragment) are allowed.
  streaming_fragment(duckdb::ClientContext& context, fragment_spec spec);

  /// Erases this fragment's stream_bind_catalog ids and closes a still-open query window
  /// (drop after build, before run).
  ~streaming_fragment();

  streaming_fragment(const streaming_fragment&)            = delete;
  streaming_fragment& operator=(const streaming_fragment&) = delete;

  /// Open the query window, declare inputs, lower to STREAMING_SOURCE plus STREAMING_SINK or
  /// RESULT_COLLECTOR, and register with the session. Callers can push after this returns. The
  /// window stays open until run(), a failed build, or destruction.
  /// @throws sirius::invalid_input_exception when already built, no catalog, no Sirius state,
  ///         null plan, or a declared input the plan never reads.
  /// @throws whatever the plan source, binder, or plan generator raises.
  void build();

  /// Submit and block. Closes the query window on success. On failure, poisons every output,
  /// then closes the window. The query_window destructor is a backstop.
  /// @throws sirius::invalid_input_exception when build() has not run, or when already run.
  /// @throws whatever the engine's execution raises.
  void run();

  /// Move every parked batch on `source`'s output `source_stream_id` into this fragment's
  /// input `input_stream_id`, then close `sender_id` on it. Checks schema, shared context,
  /// sender, and phase before any data moves.
  /// @return number of batches moved.
  std::size_t relay_from(streaming_fragment& source,
                         stream_id_t source_stream_id,
                         stream_id_t input_stream_id,
                         sender_id_t sender_id);

  /// @return false if the input had already ended.
  bool push(stream_id_t id, std::shared_ptr<cucascade::data_batch> batch);

  void close_input(stream_id_t id, sender_id_t sender);

  /// nullopt means no batch is parked now. That is not EOS. Call drained(id) for EOS.
  std::optional<std::shared_ptr<cucascade::data_batch>> pull(stream_id_t id);

  [[nodiscard]] bool drained(stream_id_t id) const;

  void fail_output(stream_id_t id, std::exception_ptr error);

  /// Take the materialized QueryResult of a result fragment. Valid after a successful run.
  duckdb::unique_ptr<duckdb::QueryResult> take_result();

  /// Physical output column types of the plan root, set during build().
  /// Relay uses this to check schema agreement before any data moves.
  /// @throws sirius::invalid_input_exception when build() has not run.
  [[nodiscard]] const duckdb::vector<sirius::logical_type>& sink_types() const;

  /// Batches currently parked on output stream `id`. Returns 0 for a result fragment.
  /// Unknown ids throw.
  [[nodiscard]] std::size_t output_batch_count(stream_id_t id) const;

  [[nodiscard]] bool is_result() const { return _spec.outputs.empty(); }

  [[nodiscard]] bool has_run() const { return _ran; }

 private:
  void require_built(const char* what) const;
  void open_window();
  void close_window(bool finish);
  void build_streaming_sink(duckdb::unique_ptr<op::sirius_physical_operator> subtree);
  void build_result_collector(duckdb::unique_ptr<op::sirius_physical_operator> subtree);
  void register_sources();
  void poison_outputs(std::exception_ptr cause) noexcept;

  duckdb::ClientContext& _context;
  fragment_spec _spec;

  // Declaration order is the lifetime contract (C++ destroys in reverse):
  // repositories outlive the engine; `_result_plan` outlives `_engine` because a
  // RESULT_COLLECTOR holds a reference into it; `_session` is destroyed before the engine
  // whose operators it borrows; the query window is last so a drop after build() releases
  // the slot before the engine it populated is destroyed.
  std::map<stream_id_t, std::shared_ptr<cucascade::shared_data_repository>> _input_repos;
  std::map<stream_id_t, std::shared_ptr<cucascade::shared_data_repository>> _output_repos;
  duckdb::shared_ptr<sirius::sirius_prepared_statement_data> _result_plan;
  duckdb::unique_ptr<duckdb::QueryResult> _result;
  std::unique_ptr<sirius::sirius_interface> _iface;
  std::unique_ptr<sirius::sirius_engine> _engine;
  stream_session _session;
  struct query_window;
  std::unique_ptr<query_window> _lifecycle;

  bool _built{false};
  bool _ran{false};
  duckdb::vector<sirius::logical_type> _sink_types;
};

}  // namespace sirius::exec
