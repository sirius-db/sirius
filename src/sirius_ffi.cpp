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

// Implementation of sirius/ffi.hpp. This translation unit sees the heavy internal types so
// consumers (e.g. the Rust bindings) never include sirius_context.hpp.

#include "config.hpp"                                      // duckdb::Config::LOG_*
#include "core_functions_extension.hpp"                    // duckdb::CoreFunctionsExtension
#include "data/sirius_converter_registry.hpp"              // sirius::converter_registry
#include "duckdb/common/arrow/result_arrow_wrapper.hpp"    // duckdb::ResultArrowArrayStreamWrapper
#include "duckdb/common/enums/optimizer_type.hpp"          // duckdb::OptimizerType
#include "duckdb/execution/column_binding_resolver.hpp"    // duckdb::ColumnBindingResolver
#include "duckdb/main/client_context.hpp"                  // duckdb::ClientContext
#include "duckdb/main/config.hpp"                          // duckdb::DBConfig
#include "duckdb/main/connection.hpp"                      // duckdb::Connection
#include "duckdb/main/database.hpp"                        // duckdb::DuckDB
#include "duckdb/main/prepared_statement_data.hpp"         // duckdb::PreparedStatementData
#include "duckdb/main/query_result.hpp"                    // duckdb::QueryResult
#include "duckdb/main/relation.hpp"                        // duckdb::Relation
#include "duckdb/optimizer/optimizer.hpp"                  // duckdb::Optimizer
#include "duckdb/parser/statement/relation_statement.hpp"  // duckdb::RelationStatement
#include "duckdb/planner/planner.hpp"                      // duckdb::Planner
#include "exec/stream_bind_catalog.hpp"                    // sirius::exec::stream_bind_catalog
#include "exec/stream_plan_bindings.hpp"  // sirius::exec::register_stream_source_function
#include "exec/streaming_fragment.hpp"    // sirius::exec::streaming_fragment, fragment_spec
#include "from_substrait.hpp"             // duckdb::SubstraitToDuckDB (compiled into libsirius)
#include "helper/type_conversions.hpp"    // sirius::from_duckdb
#include "log/logging.hpp"                // SIRIUS_LOG_INFO
#include "parquet_extension.hpp"          // duckdb::ParquetExtension
#include "sirius/ffi.hpp"
#include "sirius_config.hpp"   // sirius::sirius_config
#include "sirius_context.hpp"  // duckdb::SiriusContext

#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <set>

namespace sirius::ffi {

namespace {

constexpr const char* kSiriusStateKey   = "sirius_state";
constexpr duckdb::idx_t kArrowBatchSize = 1u << 20;

// DuckDB view name a plan uses to read input stream `id`.
std::string stream_view_name_of(std::uint64_t id) { return "sirius_stream_" + std::to_string(id); }

// The embedded DuckDB never loads the Sirius extension, so nothing on this path reads the
// SIRIUS_LOG_{BACKEND,DIR,LEVEL} environment the transparent path honors
// (SiriusContextExtensionCallback) — the engine would run with the noop sink and no log would
// reach the embedder. Install the same sink here when the embedder asks for one; without any
// of the variables the sink is left untouched (noop by default).
void install_log_sink_from_env()
{
  auto const* backend_env = std::getenv("SIRIUS_LOG_BACKEND");
  auto const* log_dir_env = std::getenv("SIRIUS_LOG_DIR");
  auto const* level_env   = std::getenv("SIRIUS_LOG_LEVEL");
  if (!backend_env && !log_dir_env && !level_env) { return; }
  auto previous_backend = duckdb::Config::LOG_BACKEND;
  auto previous_log_dir = duckdb::Config::LOG_DIR;
  auto previous_level   = duckdb::Config::LOG_LEVEL;
  try {
    if (backend_env) { duckdb::Config::LOG_BACKEND = backend_env; }
    if (log_dir_env) { duckdb::Config::LOG_DIR = log_dir_env; }
    if (level_env) { duckdb::Config::LOG_LEVEL = level_env; }
    // With no database, an unknown backend leaves the current sink in place.
    duckdb::install_configured_log_sink(nullptr);
  } catch (...) {
    // Sink construction can throw (for example, if LOG_DIR cannot be created).
    // It happens before set_sink(), so the previous sink is still active.
    duckdb::Config::LOG_BACKEND.swap(previous_backend);
    duckdb::Config::LOG_DIR.swap(previous_log_dir);
    duckdb::Config::LOG_LEVEL.swap(previous_level);
    throw;
  }
}

double elapsed_ms(std::chrono::steady_clock::time_point since)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since)
    .count();
}

// Lower a Substrait plan to a bound+optimized DuckDB LogicalOperator.
sirius::exec::bound_plan lower_substrait(duckdb::Connection& conn,
                                         const std::string& substrait_plan)
{
  auto& client = *conn.context;

  duckdb::SubstraitToDuckDB transformer(conn.context, substrait_plan, /*json=*/false);
  auto relation = transformer.TransformPlan();

  duckdb::Planner planner(client);
  planner.CreatePlan(duckdb::make_uniq<duckdb::RelationStatement>(relation));

  auto prepared =
    duckdb::make_shared_ptr<duckdb::PreparedStatementData>(duckdb::StatementType::SELECT_STATEMENT);
  prepared->names     = planner.names;
  prepared->types     = planner.types;
  prepared->value_map = std::move(planner.value_map);

  auto logical_plan = std::move(planner.plan);
  if (client.config.enable_optimizer) {
    duckdb::Optimizer optimizer(*planner.binder, client);
    logical_plan = optimizer.Optimize(std::move(logical_plan));
  }
  logical_plan->ResolveOperatorTypes();
  duckdb::ColumnBindingResolver resolver;
  duckdb::ColumnBindingResolver::Verify(*logical_plan);
  resolver.VisitOperator(*logical_plan);

  return {std::move(logical_plan), std::move(prepared)};
}

// Run `body` in a transaction on `conn`; a failed rollback does not replace body's error.
// `serial` is held throughout because every Fragment of a Context shares `conn`, and a second
// BEGIN on it fails (and invalidates the open transaction) instead of waiting.
template <typename Body>
void in_transaction(std::mutex& serial, duckdb::Connection& conn, Body&& body)
{
  std::lock_guard<std::mutex> lock(serial);
  conn.BeginTransaction();
  try {
    body();
  } catch (...) {
    try {
      conn.Rollback();
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }
    throw;
  }
  conn.Commit();
}

}  // namespace

// PIMPL: holds the engine + embedded DuckDB using DuckDB's own smart pointers
// (duckdb::shared_ptr, so the SiriusContext can register as a ClientContextState).
struct Context::Impl {
  duckdb::shared_ptr<duckdb::SiriusContext> context;
  duckdb::unique_ptr<duckdb::DuckDB> db;
  duckdb::unique_ptr<duckdb::Connection> conn;
  // Serializes transactions on `conn`; see in_transaction().
  std::mutex conn_mutex;

  void bring_up(sirius::sirius_config& config)
  {
    install_log_sink_from_env();
    sirius::converter_registry::initialize(config.get_downgrade_executor_config().copy_chunk_bytes);
    context = duckdb::make_shared_ptr<duckdb::SiriusContext>();
    context->initialize(config);

    // Substrait lowering uses core functions and resolves local_files reads to parquet_scan.
    db = duckdb::make_uniq<duckdb::DuckDB>(nullptr);
    db->LoadStaticExtension<duckdb::CoreFunctionsExtension>();
    db->LoadStaticExtension<duckdb::ParquetExtension>();
    // Cache parquet footers across binds. The Substrait consumer builds the plan through the
    // Relation API, which re-binds the whole subtree at every level, so a read of one file is
    // bound once per operator above it; with the cache off each bind re-parses the footer and
    // re-derives the column statistics (~0.5 s per bind for a 25 GB, 5k-row-group TPC-H SF100
    // lineitem: Q21 spent 10 s lowering against 4.6 s executing). Set the option at the
    // database level so every connection on this private DuckDB instance inherits it.
    // This DuckDB instance is private to the engine, and the cache validates file mtimes.
    duckdb::DBConfig::GetConfig(*db->instance)
      .SetOptionByName("parquet_metadata_cache", duckdb::Value::BOOLEAN(true));
    conn = duckdb::make_uniq<duckdb::Connection>(*db);
    // Register the engine on the connection and disable DuckDB optimizer rewrites this
    // no-fallback FFI path cannot safely execute. The transparent path only disables
    // IN_CLAUSE and COMPRESSED_MATERIALIZATION here; this path remains more conservative.
    auto& client = *conn->context;
    client.registered_state->Insert(kSiriusStateKey, context);
    // Per-connection Sirius state (guard depths, capture bookkeeping) for the
    // embedded connection, mirroring OnConnectionOpened on the transparent path.
    client.registered_state->Insert("sirius_connection_state",
                                    duckdb::make_shared_ptr<duckdb::SiriusConnectionState>());

    // Fragment bind path: register catalog + sirius_stream_source before any plan binds.
    client.registered_state->Insert(sirius::exec::stream_bind_catalog::kStateKey,
                                    duckdb::make_shared_ptr<sirius::exec::stream_bind_catalog>());
    sirius::exec::register_stream_source_function(*db->instance);
    client.config.enable_optimizer = true;
    auto& disabled = duckdb::DBConfig::GetConfig(client).options.disabled_optimizers;
    disabled.insert(duckdb::OptimizerType::IN_CLAUSE);
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
    // Keep this FFI-only restriction until its no-fallback execution path has
    // dedicated GPU_VALUES coverage.
    disabled.insert(duckdb::OptimizerType::STATISTICS_PROPAGATION);
    disabled.insert(duckdb::OptimizerType::COLUMN_LIFETIME);
    // Rewrites an ORDER BY ... LIMIT parquet scan into a semi-join on virtual
    // file_index/file_row_number columns the GPU scan drops.
    disabled.insert(duckdb::OptimizerType::LATE_MATERIALIZATION);
  }
};

Context::Context() : impl_(std::make_unique<Impl>())
{
  sirius::sirius_config config;
  // Resolve the default memory spaces only after initialize() creates Quent.
  impl_->bring_up(config);
}

Context::Context(const std::string& config_path) : impl_(std::make_unique<Impl>())
{
  sirius::sirius_config config;
  config.parse_from_file(config_path);  // throws on a missing/invalid config file
  impl_->bring_up(config);
}

// Defined here, where the heavy types are complete: destroying `impl_` tears down
// the embedded DuckDB and the initialized engine.
Context::~Context() = default;

void Context::execute_substrait(const std::string& plan, std::uintptr_t out_stream_addr)
{
  auto& client = *impl_->conn->context;

  // Binding (catalog lookups), optimization, and GPU execution all need an active
  // transaction: scans read DuckDB MVCC state through it. GPU execution is eager (the result
  // is materialized), so the transaction can close before the Arrow stream is consumed.
  duckdb::unique_ptr<duckdb::QueryResult> result;
  // Phase timings (lowering / physical planning / GPU execution), logged per query: the embedder
  // only sees the total, and the query window in the telemetry covers execution alone.
  double lower_ms = 0, plan_ms = 0, execute_ms = 0;
  in_transaction(impl_->conn_mutex, *impl_->conn, [&] {
    // The same result path as a zero-output Fragment.
    sirius::exec::fragment_spec spec;
    spec.plan_source = [&](duckdb::ClientContext&) {
      auto const lower_started = std::chrono::steady_clock::now();
      auto bound               = lower_substrait(*impl_->conn, plan);
      lower_ms                 = elapsed_ms(lower_started);
      return bound;
    };
    sirius::exec::streaming_fragment fragment(client, std::move(spec));
    auto const plan_started = std::chrono::steady_clock::now();
    fragment.build();
    plan_ms = elapsed_ms(plan_started) - lower_ms;

    auto const execute_started = std::chrono::steady_clock::now();
    fragment.run();
    result     = fragment.take_result();
    execute_ms = elapsed_ms(execute_started);
  });
  SIRIUS_LOG_INFO(
    "[sirius_ffi] execute_substrait: lower {:.1f} ms, plan {:.1f} ms, execute {:.1f} ms",
    lower_ms,
    plan_ms,
    execute_ms);

  // The stream's release callback deletes the ResultArrowArrayStreamWrapper.
  auto* wrapper = new duckdb::ResultArrowArrayStreamWrapper(std::move(result), kArrowBatchSize);
  *reinterpret_cast<ArrowArrayStream*>(out_stream_addr) = wrapper->stream;
}

std::unique_ptr<Context> make_context() { return std::make_unique<Context>(); }

std::unique_ptr<Context> make_context_from_config(const std::string& config_path)
{
  return std::make_unique<Context>(config_path);
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

// PIMPL around one streaming_fragment.
struct Fragment::Impl {
  explicit Impl(Context::Impl& ctx) : ctx(ctx) {}

  Context::Impl& ctx;

  // An input stream declared before build(). type_names are parsed at build(), inside a
  // transaction, because parsing may need a catalog lookup.
  struct declared_input {
    std::vector<std::string> names;
    std::vector<std::string> type_names;
    std::set<sirius::exec::sender_id_t> expected_senders;
  };

  std::map<sirius::exec::stream_id_t, declared_input> inputs;
  std::vector<sirius::exec::stream_id_t> outputs;
  bool broadcast_outputs{false};
  std::vector<int> hash_key_columns;

  // One streaming_fragment for both terminals. Empty outputs is a RESULT_COLLECTOR.
  std::unique_ptr<sirius::exec::streaming_fragment> fragment;

  void require_not_built(const char* what) const
  {
    if (fragment) {
      throw sirius::invalid_input_exception(std::string("Fragment: ") + what +
                                            " must be called before build()");
    }
  }

  void require_built(const char* what) const
  {
    if (!fragment) {
      throw sirius::invalid_input_exception(std::string("Fragment: build() must run before ") +
                                            what);
    }
  }

  std::map<sirius::exec::stream_id_t, sirius::exec::stream_input_spec> resolve_inputs() const
  {
    std::map<sirius::exec::stream_id_t, sirius::exec::stream_input_spec> resolved;
    for (const auto& [id, declared] : inputs) {
      sirius::exec::stream_input_spec spec;
      spec.names = declared.names;
      spec.types.reserve(declared.type_names.size());
      for (const auto& type_name : declared.type_names) {
        spec.types.push_back(
          sirius::from_duckdb(duckdb::TransformStringToLogicalType(type_name, *ctx.conn->context)));
      }
      spec.expected_senders = declared.expected_senders;
      if (spec.expected_senders.empty()) { spec.expected_senders.insert(0); }
      resolved.emplace(id, std::move(spec));
    }
    return resolved;
  }

  // CREATE OR REPLACE VIEW sirius_stream_<id> AS SELECT * FROM sirius_stream_source(<id>)
  // Needs an open transaction, and the ids declared in the stream_bind_catalog so the view's
  // SELECT binds: call it from the plan source, which streaming_fragment::build() runs after
  // declaring them.
  void create_stream_views()
  {
    for (const auto& [id, _] : inputs) {
      const auto view_name = stream_view_name_of(id);
      const auto sql       = "CREATE OR REPLACE VIEW main." + view_name + " AS SELECT * FROM " +
                       std::string(sirius::exec::kStreamSourceFunctionName) + "(" +
                       std::to_string(id) + ")";
      auto res = ctx.conn->Query(sql);
      if (res->HasError()) { res->ThrowError(); }
    }
  }
};

Fragment::Fragment(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Fragment::~Fragment() = default;

void Fragment::declare_input_column(std::uint64_t stream_id,
                                    const std::string& name,
                                    const std::string& type)
{
  impl_->require_not_built("declare_input_column");
  auto& d = impl_->inputs[stream_id];
  d.names.push_back(name);
  d.type_names.push_back(type);
}

void Fragment::declare_input_sender(std::uint64_t stream_id, std::uint32_t sender_id)
{
  impl_->require_not_built("declare_input_sender");
  impl_->inputs[stream_id].expected_senders.insert(sender_id);
}

void Fragment::declare_output(std::uint64_t stream_id)
{
  impl_->require_not_built("declare_output");
  auto& outs = impl_->outputs;
  if (std::find(outs.begin(), outs.end(), stream_id) != outs.end()) {
    throw sirius::invalid_input_exception("Fragment: duplicate output stream id " +
                                          std::to_string(stream_id));
  }
  outs.push_back(stream_id);
}

void Fragment::declare_output_broadcast()
{
  impl_->require_not_built("declare_output_broadcast");
  if (!impl_->hash_key_columns.empty()) {
    throw sirius::invalid_input_exception(
      "Fragment: broadcast and hash-partitioned output are mutually exclusive");
  }
  impl_->broadcast_outputs = true;
}

void Fragment::declare_output_hash_key(std::uint32_t column_index)
{
  impl_->require_not_built("declare_output_hash_key");
  if (impl_->broadcast_outputs) {
    throw sirius::invalid_input_exception(
      "Fragment: broadcast and hash-partitioned output are mutually exclusive");
  }
  impl_->hash_key_columns.push_back(static_cast<int>(column_index));
}

void Fragment::build(const std::string& substrait_plan)
{
  impl_->require_not_built("build");

  std::unique_ptr<sirius::exec::streaming_fragment> fragment;
  // Type-name parsing, CREATE VIEW, and Substrait lowering all need an active transaction. A
  // failure rolls back the views, so a half-declared fragment leaves nothing behind.
  in_transaction(impl_->ctx.conn_mutex, *impl_->ctx.conn, [&] {
    sirius::exec::fragment_spec spec;
    spec.inputs      = impl_->resolve_inputs();
    spec.outputs     = impl_->outputs;
    spec.plan_source = [this, &substrait_plan](duckdb::ClientContext&) {
      impl_->create_stream_views();
      return lower_substrait(*impl_->ctx.conn, substrait_plan);
    };
    // streaming_fragment rejects a partition mode on fewer than two outputs.
    if (impl_->broadcast_outputs) {
      sirius::op::partition_spec broadcast;
      broadcast.mode    = sirius::op::partition_mode::broadcast;
      spec.partitioning = std::move(broadcast);
    } else if (!impl_->hash_key_columns.empty()) {
      // key_cast_types left empty. streaming_fragment::build() fills them from output types.
      sirius::op::partition_spec hash;
      hash.mode         = sirius::op::partition_mode::hash;
      hash.key_columns  = impl_->hash_key_columns;
      spec.partitioning = std::move(hash);
    }

    fragment = std::make_unique<sirius::exec::streaming_fragment>(*impl_->ctx.conn->context,
                                                                  std::move(spec));
    fragment->build();
  });
  impl_->fragment = std::move(fragment);
}

std::size_t Fragment::relay_from(Fragment& source,
                                 std::uint64_t source_stream_id,
                                 std::uint64_t input_stream_id,
                                 std::uint32_t sender_id)
{
  impl_->require_built("relay_from()");
  if (!source.impl_->fragment) {
    throw sirius::invalid_input_exception(
      "Fragment: relay_from() requires the source fragment to have been built");
  }
  return impl_->fragment->relay_from(
    *source.impl_->fragment, source_stream_id, input_stream_id, sender_id);
}

void Fragment::close_input(std::uint64_t stream_id, std::uint32_t sender_id)
{
  impl_->require_built("close_input()");
  impl_->fragment->close_input(stream_id, sender_id);
}

void Fragment::run()
{
  impl_->require_built("run()");
  // Scans read DuckDB MVCC state through the active transaction.
  in_transaction(impl_->ctx.conn_mutex, *impl_->ctx.conn, [&] { impl_->fragment->run(); });
}

void Fragment::result_to_arrow(std::uintptr_t out_stream_addr)
{
  impl_->require_built("result_to_arrow()");
  auto result   = impl_->fragment->take_result();
  auto* wrapper = new duckdb::ResultArrowArrayStreamWrapper(std::move(result), kArrowBatchSize);
  *reinterpret_cast<ArrowArrayStream*>(out_stream_addr) = wrapper->stream;
}

std::size_t Fragment::output_batch_count(std::uint64_t stream_id) const
{
  if (!impl_->fragment) { return 0; }
  return impl_->fragment->output_batch_count(stream_id);
}

std::unique_ptr<std::vector<std::string>> Fragment::output_types() const
{
  impl_->require_built("output_types()");
  if (impl_->fragment->is_result()) {
    throw sirius::invalid_input_exception(
      "Fragment: output_types() is only valid on an intermediate fragment with output streams");
  }
  auto types = std::make_unique<std::vector<std::string>>();
  types->reserve(impl_->fragment->sink_types().size());
  for (const auto& type : impl_->fragment->sink_types()) {
    types->push_back(type.to_string());
  }
  return types;
}

std::unique_ptr<Fragment> make_fragment(Context& context)
{
  return std::unique_ptr<Fragment>(new Fragment(std::make_unique<Fragment::Impl>(*context.impl_)));
}

std::unique_ptr<std::string> stream_view_name(std::uint64_t stream_id)
{
  return std::make_unique<std::string>(stream_view_name_of(stream_id));
}

}  // namespace sirius::ffi
