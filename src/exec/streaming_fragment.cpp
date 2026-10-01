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
#include "sirius_interface.hpp"  // sirius_prepared_statement_data

#include <cudf/types.hpp>

#include <duckdb/main/query_result.hpp>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace sirius::exec {

namespace {

constexpr const char* kFragmentQueryLabel = "sirius_streaming_fragment";

// Per-key cuDF cast type so independently planned senders hash the same logical value.
// Planners may bind a column to INT32 in one fragment and INT64 in another, and cuDF murmur3
// hashes bytes, so matching keys would otherwise land in different partitions.
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
void normalize_key_cast_types(op::partition_spec& spec,
                              const duckdb::vector<sirius::logical_type>& output_types)
{
  if (!spec.key_cast_types.empty()) { return; }
  spec.key_cast_types.reserve(spec.key_columns.size());
  for (int key : spec.key_columns) {
    // The sink checks key ranges too, but is constructed after this; a bad key would index
    // output_types first.
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

namespace {

duckdb::SiriusContext& sirius_context_of(duckdb::ClientContext& context)
{
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (sirius_ctx == nullptr) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: Sirius is not registered on this connection");
  }
  return *sirius_ctx;
}

}  // namespace

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

  for (auto id : _spec.outputs) {
    if (_output_repos.count(id) != 0) {
      throw sirius::invalid_input_exception("streaming_fragment: duplicate output stream id " +
                                            std::to_string(id));
    }
    _output_repos[id] = std::make_shared<cucascade::shared_data_repository>();
  }
}

streaming_fragment::~streaming_fragment() = default;

void streaming_fragment::require_built(const char* what) const
{
  auto const current = _phase.load();
  if (current == phase::declared || current == phase::build_failed) {
    throw sirius::invalid_input_exception(std::string("streaming_fragment: ") + what +
                                          " requires build()");
  }
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
      // Declared but unread would hang. Fail here.
      throw sirius::invalid_input_exception("streaming_fragment: input stream " +
                                            std::to_string(id) +
                                            " was declared but the plan does not read it");
    }
    _session.add_source(id, *built);
  }
}

duckdb::unique_ptr<op::sirius_physical_operator> streaming_fragment::make_streaming_sink(
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
  _session.add_sink(_spec.outputs, *sink);
  return sink;
}

duckdb::unique_ptr<op::sirius_physical_operator> streaming_fragment::make_result_collector(
  duckdb::unique_ptr<op::sirius_physical_operator> subtree,
  duckdb::shared_ptr<duckdb::PreparedStatementData> prepared)
{
  if (!prepared) { prepared = synthesize_prepared(subtree->types); }
  // The collector decodes GPU output with prepared->types; a mismatch would misread it.
  _sink_types = sirius::from_duckdb_vec(prepared->types);
  // DuckDB types SUM(BIGINT) as HUGEINT while the planner narrows it to BIGINT; the collector
  // casts the BIGINT column back, so that pair is not a mismatch.
  auto decodes_as = [](const sirius::logical_type& declared, const sirius::logical_type& planned) {
    return declared == planned ||
           (declared.id() == sirius::type_id::HUGEINT && planned.id() == sirius::type_id::BIGINT);
  };
  if (!std::equal(_sink_types.begin(),
                  _sink_types.end(),
                  subtree->types.begin(),
                  subtree->types.end(),
                  decodes_as)) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: prepared result types do not match the physical plan's " +
      std::to_string(subtree->types.size()) + " output column(s)");
  }

  _result_plan = duckdb::make_shared_ptr<sirius::sirius_prepared_statement_data>(
    std::move(prepared), std::move(subtree));
  return duckdb::make_uniq_base<op::sirius_physical_result_collector,
                                op::sirius_physical_materialized_collector>(*_result_plan,
                                                                            _context);
}

void streaming_fragment::build()
{
  if (_phase == phase::build_failed) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: a failed build() cannot be retried; create a new fragment");
  }
  if (_phase != phase::declared) {
    throw sirius::invalid_input_exception("streaming_fragment: already built");
  }

  duckdb::shared_ptr<stream_bind_catalog> catalog;
  // Only bind and create_plan read these ids. Erase them on every exit so no stale binding
  // reaches the next fragment on this connection.
  auto erase_declared = [&]() noexcept {
    if (!catalog) { return; }
    for (const auto& [id, _] : _spec.inputs) {
      try {
        catalog->erase(id);
      } catch (...) {  // NOLINT(bugprone-empty-catch)
      }
    }
  };

  try {
    auto& sirius_ctx = sirius_context_of(_context);
    catalog          = catalog_for(_context);
    // declare() replaces any earlier binding of the id.
    for (const auto& [id, input] : _spec.inputs) {
      catalog->declare(id,
                       stream_input_binding{input.names,
                                            input.types,
                                            std::make_shared<cucascade::shared_data_repository>(),
                                            input.expected_senders,
                                            nullptr});
    }

    auto bound = _spec.plan_source(_context);
    if (!bound.plan) {
      throw sirius::invalid_input_exception("streaming_fragment: plan source produced no plan");
    }

    duckdb::unique_ptr<op::sirius_physical_operator> subtree;
    {
      // create_plan reads the pinned-table registry, which only the slot keeps stable. run()
      // opens the window and compares this epoch to detect a stale plan.
      duckdb::SiriusContext::SlotGuard plan_slot(sirius_ctx, _context);
      _planned_pin_epoch = sirius_ctx.get_scan_manager().pin_registry_epoch();
      subtree            = sirius::planner::sirius_physical_plan_generator(_context).create_plan(
        std::move(bound.plan));
    }

    _plan_root = is_result() ? make_result_collector(std::move(subtree), std::move(bound.prepared))
                             : make_streaming_sink(std::move(subtree));
    register_sources();
  } catch (...) {
    // The session keeps partial registrations, so the fragment is single-shot.
    _phase = phase::build_failed;
    erase_declared();
    throw;
  }
  erase_declared();
  _phase = phase::built;
}

void streaming_fragment::run()
{
  require_built("run()");
  switch (_phase.load()) {
    case phase::built: break;
    case phase::run_failed:
      throw sirius::invalid_input_exception(
        "streaming_fragment: a previous run() failed; create a new fragment");
    case phase::running:
      throw sirius::invalid_input_exception("streaming_fragment: already running");
    default: throw sirius::invalid_input_exception("streaming_fragment: already run");
  }
  // Waiting on an open input would hold the query window, and so every other query on this
  // engine, until another thread closed it.
  for (auto id : _session.input_streams()) {
    if (!_session.input_closed(id)) {
      throw sirius::invalid_input_exception(
        "streaming_fragment: run() needs every input closed first; input stream " +
        std::to_string(id) + " is still open");
    }
  }

  auto& sirius_ctx = sirius_context_of(_context);
  // The switch above reports the state; the CAS catches a concurrent run().
  auto expected = phase::built;
  if (!_phase.compare_exchange_strong(expected, phase::running)) {
    throw sirius::invalid_input_exception("streaming_fragment: already running");
  }
  std::optional<duckdb::SiriusContext::StandaloneQueryScope> window;
  try {
    window.emplace(sirius_ctx, _context, kFragmentQueryLabel);
    if (sirius_ctx.get_scan_manager().pin_registry_epoch() != _planned_pin_epoch) {
      throw sirius::invalid_input_exception(
        "streaming_fragment: a table was pinned or unpinned between build() and run(); "
        "create a new fragment");
    }
    _engine =
      std::make_unique<sirius::sirius_engine>(_context, window->query_id(), kFragmentQueryLabel);
    _engine->initialize(std::move(_plan_root));
    _engine->execute();
    if (is_result()) { _result = _engine->get_result(); }
    window->finish();
  } catch (...) {
    // Fail every output before the window closes. Otherwise a peer in wait() blocks forever.
    poison_outputs(std::current_exception());
    _result.reset();
    _phase = phase::run_failed;
    window.reset();
    throw;
  }
  if (_result && _result->HasError()) {
    _phase      = phase::run_failed;
    auto failed = std::move(_result);
    failed->ThrowError();
  }
  _phase = phase::ran;
}

std::size_t streaming_fragment::relay_from(streaming_fragment& source,
                                           stream_id_t source_stream_id,
                                           stream_id_t input_stream_id,
                                           sender_id_t sender_id)
{
  require_built("relay_from()");
  auto const source_phase = source._phase.load();
  if (source_phase == phase::declared || source_phase == phase::build_failed) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: relay_from() requires the source fragment to have been built");
  }
  if (source_phase == phase::run_failed) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: relay_from() source fragment's run() failed; its output is poisoned");
  }
  if (source_phase != phase::ran) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: relay_from() requires the source fragment to have run — call "
      "source.run() first, otherwise an empty stream is indistinguishable from a finished one "
      "and the input would be closed early");
  }
  if (_phase != phase::built) {
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

void streaming_fragment::close_input(stream_id_t id, sender_id_t sender)
{
  require_built("close_input()");
  _session.close_input(id, sender);
}

std::optional<std::shared_ptr<cucascade::data_batch>> streaming_fragment::pull(stream_id_t id)
{
  require_built("pull()");
  // After a failed run() the outputs are poisoned, so pulling rethrows the cause.
  auto const current = _phase.load();
  if (current != phase::ran && current != phase::run_failed) {
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

duckdb::unique_ptr<duckdb::QueryResult> streaming_fragment::take_result()
{
  if (!is_result()) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: take_result() is only valid on a fragment with no output streams");
  }
  if (_phase != phase::ran) {
    throw sirius::invalid_input_exception(
      "streaming_fragment: run() must complete before take_result()");
  }
  if (!_result) {
    throw sirius::invalid_input_exception("streaming_fragment: the result was already taken");
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
  require_built("output_batch_count()");
  auto it = _output_repos.find(id);
  if (it == _output_repos.end()) {
    throw sirius::invalid_input_exception("streaming_fragment: no output stream with id " +
                                          std::to_string(id));
  }
  return it->second->total_size();
}

}  // namespace sirius::exec
