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

#include "exec/streaming_fragment.hpp"

#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius/exception.hpp"
#include "sirius_context.hpp"
#include "sirius_engine.hpp"
#include "sirius_interface.hpp"

#include <string>
#include <utility>

namespace sirius::exec {

namespace {

constexpr const char* kFragmentQueryLabel = "sirius_streaming_fragment";

duckdb::shared_ptr<stream_bind_catalog> catalog_for(duckdb::ClientContext& context)
{
  auto catalog = context.registered_state->Get<stream_bind_catalog>(stream_bind_catalog::kStateKey);
  if (!catalog) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: no stream catalog on this connection");
  }
  return catalog;
}

}  // namespace

streaming_fragment::streaming_fragment(duckdb::ClientContext& context, fragment_spec spec)
  : _context(context), _spec(std::move(spec))
{
  if (!_spec.plan_source) {
    throw sirius::invalid_input_exception("streaming_fragment: a plan source is required");
  }
  if (_spec.outputs.empty()) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: a fragment must declare at least one output stream");
  }
  if (_spec.outputs.size() > 1 && !_spec.partitioning.has_value()) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: " + std::to_string(_spec.outputs.size()) +
      " output streams need a partition spec; a gather fragment has exactly one");
  }

  // Created here, outside data_repository_manager_, so the window's mandatory cleanup
  // (StandaloneQueryScope::finish()) cannot destroy them. This is what lets a sender's output
  // outlive its own fragment.
  for (const auto& [id, _] : _spec.inputs) {
    _input_repos[id] = std::make_shared<cucascade::shared_data_repository>();
  }
  for (auto id : _spec.outputs) {
    if (_output_repos.count(id) != 0) {
      throw sirius::invalid_input_exception("streaming_fragment: duplicate output stream id " +
                                            std::to_string(id));
    }
    _output_repos[id] = std::make_shared<cucascade::shared_data_repository>();
  }
}

streaming_fragment::~streaming_fragment()
{
  // The catalog is per-connection and this fragment's declarations must not be visible to the
  // next one. Swallow, because a throwing destructor during stack unwinding would terminate.
  try {
    catalog_for(_context)->clear();
  } catch (...) {  // NOLINT(bugprone-empty-catch)
  }
}

void streaming_fragment::build(sirius::query_id_t query_id)
{
  if (_built) { throw sirius::invalid_input_exception("streaming_fragment: already built"); }

  auto catalog = catalog_for(_context);
  catalog->clear();

  // Declare every input before planning: sirius_stream_source's DuckDB bind resolves its schema
  // from here, and create_plan(LogicalGet&) reads the repository and sender set back out.
  for (const auto& [id, input] : _spec.inputs) {
    catalog->declare(
      id,
      stream_input_binding{
        input.names, input.types, _input_repos.at(id), input.expected_senders, nullptr});
  }

  auto logical_plan = _spec.plan_source(_context);
  if (!logical_plan) {
    throw sirius::invalid_input_exception("streaming_fragment: plan source produced no plan");
  }

  sirius::planner::sirius_physical_plan_generator generator(_context);
  auto subtree = generator.create_plan(std::move(logical_plan));

  // A STREAMING_SINK is a normal unary operator: the subtree goes in children[], unlike the
  // RESULT_COLLECTOR, which keeps its child outside and needs special descent in the generator.
  auto types       = subtree->types;
  auto cardinality = subtree->estimated_cardinality;

  std::vector<std::shared_ptr<cucascade::shared_data_repository>> sink_repos;
  sink_repos.reserve(_spec.outputs.size());
  for (auto id : _spec.outputs) {
    sink_repos.push_back(_output_repos.at(id));
  }

  duckdb::unique_ptr<op::sirius_physical_streaming_sink> sink;
  if (_spec.partitioning.has_value()) {
    sink = duckdb::make_uniq<op::sirius_physical_streaming_sink>(
      std::move(types), cardinality, std::move(sink_repos), *_spec.partitioning);
  } else {
    sink = duckdb::make_uniq<op::sirius_physical_streaming_sink>(
      std::move(types), cardinality, sink_repos.front());
  }
  sink->children.push_back(std::move(subtree));

  // Hand the plan to the engine and let it own the tree, which is the path a normal query
  // takes. The fragment owns the engine, so the sink still outlives run() and stays pullable.
  _iface = std::make_unique<sirius::sirius_interface>(
    _context, std::optional<std::string>(kFragmentQueryLabel));
  _engine = std::make_unique<sirius::sirius_engine>(_context, *_iface, query_id);
  _engine->initialize(std::move(sink));

  auto& sink_ref = _engine->sirius_physical_plan->Cast<op::sirius_physical_streaming_sink>();

  // Register both ends with the session. The engine owns the operators; the session borrows.
  _session.add_sink(_spec.outputs, sink_ref);
  for (const auto& [id, _] : _spec.inputs) {
    auto* built = catalog->get(id).built;
    if (built == nullptr) {
      // The plan never read this stream. Silently ignoring it would leave its senders with
      // nowhere to push and the fragment waiting on a stream nothing drains.
      throw sirius::invalid_input_exception("streaming_fragment: input stream " +
                                            std::to_string(id) +
                                            " was declared but the plan does not read it");
    }
    _session.add_source(id, *built);
  }

  _built = true;
}

void streaming_fragment::run()
{
  if (!_built) {
    throw sirius::invalid_input_exception("streaming_fragment: build() must run before run()");
  }

  // The query window is the CALLER's, not the fragment's. Opening a StandaloneQueryScope resets
  // the task creator and the operator-id counter, and its finish() resets the scan manager --
  // all of which build() has already populated. Taking the slot here would discard those
  // plan-time registrations, and the fragment would run zero tasks and return an empty output
  // silently. The caller brackets build() and run() in one window, as Context::execute_substrait
  // does.
  _engine->execute();
}

const std::shared_ptr<cucascade::shared_data_repository>& streaming_fragment::input_repository(
  stream_id_t id) const
{
  auto it = _input_repos.find(id);
  if (it == _input_repos.end()) {
    throw sirius::invalid_input_exception("streaming_fragment: no input stream with id " +
                                          std::to_string(id));
  }
  return it->second;
}

const std::shared_ptr<cucascade::shared_data_repository>& streaming_fragment::output_repository(
  stream_id_t id) const
{
  auto it = _output_repos.find(id);
  if (it == _output_repos.end()) {
    throw sirius::invalid_input_exception("streaming_fragment: no output stream with id " +
                                          std::to_string(id));
  }
  return it->second;
}

}  // namespace sirius::exec
