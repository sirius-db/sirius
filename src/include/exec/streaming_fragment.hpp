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

/// Schema + provenance of one input stream, as the front end declares it.
struct stream_input_spec {
  std::vector<std::string> names;
  duckdb::vector<sirius::logical_type> types;
  /// Every sender expected to close this stream. The stream ends only once all of them have.
  std::set<sender_id_t> expected_senders;
};

/// Produces the fragment's bound, optimized DuckDB logical plan.
///
/// A function rather than a fixed input because the two callers differ: the compute node hands
/// over Substrait protobuf bytes, while tests build a plan from SQL. Both end at the same place —
/// a `LogicalOperator` the Sirius plan generator can lower.
using logical_plan_source =
  std::function<duckdb::unique_ptr<duckdb::LogicalOperator>(duckdb::ClientContext&)>;

struct fragment_spec {
  logical_plan_source plan_source;
  /// One entry per exchange input. Ids are session-local.
  std::map<stream_id_t, stream_input_spec> inputs;
  /// One id per sink destination, positional: `outputs[i]` addresses partition i.
  std::vector<stream_id_t> outputs;
  /// Absent means gather (a single destination, no partitioning).
  std::optional<op::partition_spec> partitioning;
};

/// One plan fragment, owning everything that must outlive a single query.
///
/// The repositories are created here as plain `shared_ptr` and are **never registered with
/// `data_repository_manager_`**. `QueryEnd()`'s `clear_all_repositories()` therefore cannot touch
/// them, so a sender's output survives its own fragment teardown and is still there when the
/// receiver runs. That is the single fact that makes sequential streaming work at all.
///
/// The fragment owns the *engine*, and the engine owns the plan tree. That keeps the sink alive
/// past `run()` -- a consumer pulls from it after the fragment has finished -- while still taking
/// the ordinary `initialize()` path a normal query uses.
class streaming_fragment {
 public:
  streaming_fragment(duckdb::ClientContext& context, fragment_spec spec);
  ~streaming_fragment();

  streaming_fragment(const streaming_fragment&)            = delete;
  streaming_fragment& operator=(const streaming_fragment&) = delete;

  /// Build the plan: declare the inputs, lower them to STREAMING_SOURCEs, root the tree in a
  /// STREAMING_SINK, and register both ends with the session. Separate from `run()` so a caller
  /// can push into the inputs before execution starts.
  ///
  /// `query_id` is the caller's open `StandaloneQueryScope` window — the engine wires operators
  /// into that query's repository manager. Do not open a second window between `build` and `run`.
  void build(sirius::query_id_t query_id);

  /// Submit and block until the fragment's pipelines finish.
  ///
  /// The caller owns the query lifecycle and must bracket `build()` and `run()` together in one
  /// `StandaloneQueryScope`. Beginning a new window between them resets the task creator and
  /// scan manager that `build()` populated, and the fragment then runs zero tasks and produces
  /// an empty output with no error.
  /// @throws if `build()` has not run.
  void run();

  [[nodiscard]] stream_session& session() { return _session; }

  /// The engine backing this fragment, for tests that need to inspect pipeline state.
  [[nodiscard]] sirius::sirius_engine& engine() { return *_engine; }

  /// The repository behind input stream `id`, for a caller that wants to inspect what it pushed.
  [[nodiscard]] const std::shared_ptr<cucascade::shared_data_repository>& input_repository(
    stream_id_t id) const;

  /// The repository behind output stream `id`. This is what a downstream fragment consumes.
  [[nodiscard]] const std::shared_ptr<cucascade::shared_data_repository>& output_repository(
    stream_id_t id) const;

 private:
  duckdb::ClientContext& _context;
  fragment_spec _spec;

  // Declaration order IS the lifetime contract (members are destroyed in reverse): the
  // repositories outlive the engine, the engine owns the plan, and the session -- which only
  // borrows operators out of that plan -- is torn down first.
  std::map<stream_id_t, std::shared_ptr<cucascade::shared_data_repository>> _input_repos;
  std::map<stream_id_t, std::shared_ptr<cucascade::shared_data_repository>> _output_repos;
  std::unique_ptr<sirius::sirius_interface> _iface;
  std::unique_ptr<sirius::sirius_engine> _engine;
  stream_session _session;

  bool _built{false};
};

}  // namespace sirius::exec
