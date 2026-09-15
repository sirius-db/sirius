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

#include "helper/type_conversions.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius/exception.hpp"
#include "sirius_context.hpp"
#include "sirius_engine.hpp"
#include "sirius_interface.hpp"

#include <cudf/types.hpp>

#include <duckdb/main/query_result.hpp>

#include <memory>
#include <string>
#include <string_view>
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

duckdb::shared_ptr<duckdb::PreparedStatementData> synthesize_prepared(
  const duckdb::vector<sirius::logical_type>& types)
{
  auto prepared =
    duckdb::make_shared_ptr<duckdb::PreparedStatementData>(duckdb::StatementType::SELECT_STATEMENT);
  prepared->types = sirius::to_duckdb_vec(types);
  prepared->names.reserve(types.size());
  for (duckdb::idx_t i = 0; i < types.size(); ++i) {
    prepared->names.push_back("col_" + std::to_string(i));
  }
  return prepared;
}

}  // namespace

struct streaming_fragment::query_window {
  duckdb::SiriusContext::StandaloneQueryScope scope;
  query_window(duckdb::SiriusContext& ctx, duckdb::ClientContext& client, std::string_view label)
    : scope(ctx, client, label)
  {
  }
};

streaming_fragment::streaming_fragment(duckdb::ClientContext& context, fragment_spec spec)
  : _context(context), _spec(std::move(spec))
{
  if (!_spec.plan_source) {
    throw sirius::invalid_input_exception("streaming_fragment: a plan source is required");
  }
  if (_spec.partitioning.has_value() && _spec.outputs.size() < 2) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: a partition spec needs at least two output streams; this fragment has " +
      std::to_string(_spec.outputs.size()));
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
  // Drop only the ids this fragment declared. clear() would wipe the whole per-connection
  // catalog, including a peer fragment's declarations; swallow in dtor.
  try {
    auto catalog = catalog_for(_context);
    for (const auto& [id, _] : _spec.inputs) {
      catalog->erase(id);
    }
  } catch (...) {  // NOLINT(bugprone-empty-catch)
  }
  try {
    _lifecycle.reset();
  } catch (...) {  // NOLINT(bugprone-empty-catch)
  }
}

void streaming_fragment::require_built(const char* what) const
{
  if (!_built) {
    throw sirius::invalid_input_exception(std::string("streaming_fragment: ") + what +
                                          " requires build()");
  }
}

void streaming_fragment::open_window()
{
  auto sirius_ctx = _context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (sirius_ctx == nullptr) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: Sirius is not registered on this connection");
  }
  _lifecycle = std::make_unique<query_window>(*sirius_ctx, _context, kFragmentQueryLabel);
}

void streaming_fragment::close_window(bool finish)
{
  if (!_lifecycle) { return; }
  if (finish) { _lifecycle->scope.finish(); }
  _lifecycle.reset();
}

void streaming_fragment::poison_outputs(std::exception_ptr cause) noexcept
{
  for (auto id : _spec.outputs) {
    try {
      _session.fail_output(id, cause);
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }
  }
}

void streaming_fragment::register_sources()
{
  auto catalog = catalog_for(_context);
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
}

void streaming_fragment::build_streaming_sink(
  duckdb::unique_ptr<op::sirius_physical_operator> subtree)
{
  auto types       = subtree->types;
  auto cardinality = subtree->estimated_cardinality;
  _sink_types      = types;

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

  _iface = std::make_unique<sirius::sirius_interface>(
    _context, std::optional<std::string>(kFragmentQueryLabel));
  _engine =
    std::make_unique<sirius::sirius_engine>(_context, *_iface, _lifecycle->scope.query_id());
  _engine->initialize(std::move(sink));

  auto& sink_ref = _engine->sirius_physical_plan->Cast<op::sirius_physical_streaming_sink>();
  _session.add_sink(_spec.outputs, sink_ref);
  register_sources();
}

void streaming_fragment::build_result_collector(
  duckdb::unique_ptr<op::sirius_physical_operator> subtree)
{
  auto prepared = _spec.prepared;
  if (!prepared) { prepared = synthesize_prepared(subtree->types); }
  _sink_types = sirius::from_duckdb_vec(prepared->types);

  _result_plan = duckdb::make_shared_ptr<sirius::sirius_prepared_statement_data>(
    std::move(prepared), std::move(subtree));

  auto collector =
    duckdb::make_uniq_base<op::sirius_physical_result_collector,
                           op::sirius_physical_materialized_collector>(*_result_plan, _context);

  _iface = std::make_unique<sirius::sirius_interface>(
    _context, std::optional<std::string>(kFragmentQueryLabel));
  _engine =
    std::make_unique<sirius::sirius_engine>(_context, *_iface, _lifecycle->scope.query_id());
  _engine->initialize(std::move(collector));
  register_sources();
}

void streaming_fragment::build()
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

  open_window();
  try {
    auto logical_plan = _spec.plan_source(_context);
    if (!logical_plan) {
      throw sirius::invalid_input_exception("streaming_fragment: plan source produced no plan");
    }

    sirius::planner::sirius_physical_plan_generator generator(_context);
    auto subtree = generator.create_plan(std::move(logical_plan));

    if (_spec.outputs.empty()) {
      build_result_collector(std::move(subtree));
    } else {
      build_streaming_sink(std::move(subtree));
    }
    _built = true;
  } catch (...) {
    // Release the slot so a later fragment on this connection can build. The destructor
    // backstop would do the same, but only when *this* is dropped.
    try {
      close_window(false);
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }
    throw;
  }
}

void streaming_fragment::run()
{
  require_built("run()");
  if (_ran) { throw sirius::invalid_input_exception("streaming_fragment: already run"); }

  try {
    // The window opened in build() must stay the one execute() uses: a second
    // StandaloneQueryScope would reset task_creator / scan manager that build() populated
    // → zero tasks, empty output, no error.
    _engine->execute();
    if (is_result()) { _result = _engine->get_result(); }
  } catch (...) {
    // Poison every output before unwinding: otherwise the streams are neither closed nor
    // failed, so a peer parked in wait() blocks forever with no error anywhere. fail_output is
    // idempotent (first-failure-wins), so this stays safe even when a caller (e.g.
    // sirius::ffi::Fragment::run()) also poisons the same outputs itself.
    poison_outputs(std::current_exception());
    try {
      close_window(false);
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }
    throw;
  }
  _ran = true;
  close_window(true);
  if (_result && _result->HasError()) { _result->ThrowError(); }
}

std::size_t streaming_fragment::relay_from(streaming_fragment& source,
                                           stream_id_t source_stream_id,
                                           stream_id_t input_stream_id,
                                           sender_id_t sender_id)
{
  require_built("relay_from()");
  if (!source._built) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: relay_from() requires the source fragment to have been built");
  }
  if (!source._ran) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: relay_from() requires the source fragment to have run — call "
      "source.run() first, otherwise an empty stream is indistinguishable from a finished one "
      "and the input would be closed early");
  }
  if (_ran) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: relay_from() must run before this fragment's run()");
  }
  if (&source._context != &_context) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: relay_from() requires both fragments to share a ClientContext");
  }
  if (source.is_result()) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: relay source has no output streams — a result fragment produces a "
      "QueryResult via take_result(), not a relayable stream");
  }

  auto declared_it = _spec.inputs.find(input_stream_id);
  if (declared_it == _spec.inputs.end()) {
    throw sirius::invalid_input_exception("streaming_fragment: relay target input stream " +
                                          std::to_string(input_stream_id) +
                                          " was never declared on this fragment");
  }
  const auto& declared_senders = declared_it->second.expected_senders;
  if (!declared_senders.empty() && declared_senders.count(sender_id) == 0) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: sender " + std::to_string(sender_id) +
      " is not in the expected set for input stream " + std::to_string(input_stream_id));
  }

  {
    const auto& declared = declared_it->second.types;
    const auto& produced = source.sink_types();
    if (produced.size() != declared.size()) {
      throw sirius::invalid_input_exception(
        "streaming_fragment: relay into stream " + std::to_string(input_stream_id) + " expects " +
        std::to_string(declared.size()) + " declared columns but the source sink produces " +
        std::to_string(produced.size()));
    }
    for (std::size_t i = 0; i < declared.size(); ++i) {
      if (produced[i] != declared[i]) {
        throw sirius::invalid_input_exception(
          "streaming_fragment: relay into stream " + std::to_string(input_stream_id) + " column " +
          std::to_string(i) + " is declared " + declared[i].to_string() +
          " but the source sink produces " + produced[i].to_string());
      }
    }
  }

  std::size_t moved = 0;
  while (auto batch = source._session.pull(source_stream_id)) {
    if (!_session.push(input_stream_id, *batch)) {
      throw sirius::invalid_input_exception("streaming_fragment: input stream " +
                                            std::to_string(input_stream_id) +
                                            " refused a batch; it had already ended");
    }
    ++moved;
  }
  _session.close_input(input_stream_id, sender_id);
  return moved;
}

bool streaming_fragment::push(stream_id_t id, std::shared_ptr<cucascade::data_batch> batch)
{
  require_built("push()");
  return _session.push(id, std::move(batch));
}

void streaming_fragment::close_input(stream_id_t id, sender_id_t sender)
{
  require_built("close_input()");
  _session.close_input(id, sender);
}

std::optional<std::shared_ptr<cucascade::data_batch>> streaming_fragment::pull(stream_id_t id)
{
  require_built("pull()");
  if (!_ran) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: pull() requires run() first, otherwise an empty stream is "
      "indistinguishable from a finished one");
  }
  return _session.pull(id);
}

bool streaming_fragment::drained(stream_id_t id) const
{
  require_built("drained()");
  return _session.drained(id);
}

void streaming_fragment::fail_output(stream_id_t id, std::exception_ptr error)
{
  require_built("fail_output()");
  _session.fail_output(id, std::move(error));
}

duckdb::unique_ptr<duckdb::QueryResult> streaming_fragment::take_result()
{
  if (!is_result()) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: take_result() is only valid on a fragment with no output streams");
  }
  if (!_ran || !_result) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: run() must complete before take_result()");
  }
  return std::move(_result);
}

const duckdb::vector<sirius::logical_type>& streaming_fragment::sink_types() const
{
  require_built("sink_types()");
  return _sink_types;
}

std::size_t streaming_fragment::output_batch_count(stream_id_t id) const
{
  if (is_result()) { return 0; }
  require_built("output_batch_count()");
  auto it = _output_repos.find(id);
  if (it == _output_repos.end()) {
    throw sirius::invalid_input_exception("streaming_fragment: no output stream with id " +
                                          std::to_string(id));
  }
  return it->second->total_size();
}

}  // namespace sirius::exec
