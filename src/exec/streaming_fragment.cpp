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

#include <cudf/types.hpp>

#include <string>
#include <utility>

namespace sirius::exec {

namespace {

constexpr const char* kFragmentQueryLabel = "sirius_streaming_fragment";

// Derive a per-key cuDF cast type so independently-planned senders always hash identically.
// Different planners may bind the same logical column to different native widths (e.g. INT32 vs
// INT64). cuDF's murmur3 hashes bytes, not values, so without normalization matching keys land in
// different partitions and groups are silently split.
//
// Rules:
//   TINYINT / SMALLINT / INTEGER → INT64   (all sub-64-bit integers → canonical 64-bit)
//   BIGINT / BOOLEAN / VARCHAR   → EMPTY   (already canonical; hash as-is)
//   DECIMAL (any precision/scale) → FLOAT64 (normalized floating representation)
//   anything else                → throw
cudf::data_type derive_key_cast_type(const sirius::logical_type& t)
{
  switch (t.id()) {
    case sirius::type_id::TINYINT:
    case sirius::type_id::SMALLINT:
    case sirius::type_id::INTEGER: return cudf::data_type{cudf::type_id::INT64};
    case sirius::type_id::BIGINT:
    case sirius::type_id::BOOLEAN:
    case sirius::type_id::VARCHAR: return cudf::data_type{cudf::type_id::EMPTY};
    case sirius::type_id::DECIMAL: return cudf::data_type{cudf::type_id::FLOAT64};
    default:
      throw sirius::invalid_input_exception(
        "streaming_fragment: unsupported partition key type — only integer, boolean, varchar, and "
        "decimal columns may be used as hash partition keys");
  }
}

// Fill partition_spec::key_cast_types when the caller left it empty.
// No-op when the caller supplied their own cast types.
void normalize_key_cast_types(op::partition_spec& spec,
                              const duckdb::vector<sirius::logical_type>& output_types)
{
  if (!spec.key_cast_types.empty()) { return; }
  spec.key_cast_types.reserve(spec.key_columns.size());
  for (int key : spec.key_columns) {
    // The sink validates key ranges too, but it is constructed after this runs — so an
    // out-of-range or negative key would index output_types out of bounds first.
    if (key < 0 || static_cast<std::size_t>(key) >= output_types.size()) {
      throw sirius::invalid_input_exception("streaming_fragment: partition key column " +
                                            std::to_string(key) + " is out of range for a " +
                                            std::to_string(output_types.size()) + "-column output");
    }
    spec.key_cast_types.push_back(derive_key_cast_type(output_types[key]));
  }
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

  // Repositories escape data_repository_manager_ cleanup so sender output outlives this fragment.
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
  // Drop only the ids this fragment declared, so a peer sharing this connection keeps its own.
  // Swallow in dtor.
  try {
    auto catalog = catalog_for(_context);
    for (const auto& [id, _] : _spec.inputs) {
      catalog->erase(id);
    }
  } catch (...) {  // NOLINT(bugprone-empty-catch)
  }
}

void streaming_fragment::build(sirius::query_id_t query_id)
{
  if (_built) { throw sirius::invalid_input_exception("streaming_fragment: already built"); }

  auto catalog = catalog_for(_context);
  // Same reason as the destructor: erase our own ids so a rebuild is idempotent without
  // discarding declarations that belong to another fragment on this connection.
  for (const auto& [id, _] : _spec.inputs) {
    catalog->erase(id);
  }

  // Declare before planning: bind resolves schema; create_plan reads repo + senders.
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

  // STREAMING_SINK is a normal unary: subtree in children[] (unlike RESULT_COLLECTOR).
  auto types       = subtree->types;
  auto cardinality = subtree->estimated_cardinality;
  _sink_types      = types;  // snapshot before types is moved into the sink constructor

  std::vector<std::shared_ptr<cucascade::shared_data_repository>> sink_repos;
  sink_repos.reserve(_spec.outputs.size());
  for (auto id : _spec.outputs) {
    sink_repos.push_back(_output_repos.at(id));
  }

  duckdb::unique_ptr<op::sirius_physical_streaming_sink> sink;
  if (_spec.partitioning.has_value()) {
    normalize_key_cast_types(*_spec.partitioning, types);
    sink = duckdb::make_uniq<op::sirius_physical_streaming_sink>(
      std::move(types), cardinality, std::move(sink_repos), *_spec.partitioning);
  } else {
    sink = duckdb::make_uniq<op::sirius_physical_streaming_sink>(
      std::move(types), cardinality, sink_repos.front());
  }
  sink->children.push_back(std::move(subtree));

  // Engine owns the plan; fragment owns the engine so the sink stays pullable after run().
  _iface = std::make_unique<sirius::sirius_interface>(
    _context, std::optional<std::string>(kFragmentQueryLabel));
  _engine = std::make_unique<sirius::sirius_engine>(_context, *_iface, query_id);
  _engine->initialize(std::move(sink));

  auto& sink_ref = _engine->sirius_physical_plan->Cast<op::sirius_physical_streaming_sink>();

  _session.add_sink(_spec.outputs, sink_ref);
  for (const auto& [id, _] : _spec.inputs) {
    auto* built = catalog->get(id).built;
    if (built == nullptr) {
      // Declared but unread = hang; fail loudly.
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

  try {
    // Shared query window (don't open a second StandaloneQueryScope): a new window resets
    // task_creator / scan manager that build() populated → zero tasks, empty output, no error.
    _engine->execute();
  } catch (...) {
    // Poison every output before unwinding: otherwise the streams are neither closed nor
    // failed, so a peer parked in wait() blocks forever with no error anywhere. fail_output is
    // idempotent (first-failure-wins), so this stays safe even when a caller (e.g.
    // sirius::ffi::Fragment::run()) also poisons the same outputs itself.
    auto const cause = std::current_exception();
    for (auto id : _spec.outputs) {
      try {
        _session.fail_output(id, cause);
      } catch (...) {  // NOLINT(bugprone-empty-catch)
      }
    }
    throw;
  }
}

const duckdb::vector<sirius::logical_type>& streaming_fragment::sink_types() const
{
  if (!_built) {
    throw sirius::invalid_input_exception("streaming_fragment: sink_types() called before build()");
  }
  return _sink_types;
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
