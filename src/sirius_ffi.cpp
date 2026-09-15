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
#include "cudf/cudf_utils.hpp"                             // sirius::get_cudf_type
#include "data/data_batch_utils.hpp"                       // sirius::get_cudf_table_view, make_data_batch
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
#include "exec/exchange_staging_arena.hpp"                 // sirius::exec::exchange_staging_arena
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

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace sirius::ffi {

namespace {

constexpr const char* kSiriusStateKey   = "sirius_state";
constexpr duckdb::idx_t kArrowBatchSize = 1u << 20;
/// chunked_pack gather granularity. Every next() span must be exactly this long, so a lease is
/// the payload plus one chunk of slack for the final span. 1 MiB is cudf's minimum.
constexpr std::size_t kPackChunkBytes = 8u << 20;

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

void check_declared_schema(const sirius::exec::stream_input_spec& declared,
                           const cudf::table_view& view,
                           std::uint64_t stream_id,
                           const char* what)
{
  if (static_cast<std::size_t>(view.num_columns()) != declared.types.size()) {
    throw sirius::invalid_input_exception(
      std::string("Fragment: ") + what + " for stream " + std::to_string(stream_id) + " carries " +
      std::to_string(view.num_columns()) + " columns but the stream declares " +
      std::to_string(declared.types.size()));
  }
  for (std::size_t i = 0; i < declared.types.size(); ++i) {
    const auto expected = sirius::get_cudf_type(declared.types[i]);
    const auto actual   = view.column(static_cast<cudf::size_type>(i)).type();
    if (actual != expected) {
      throw sirius::invalid_input_exception(
        std::string("Fragment: ") + what + " for stream " + std::to_string(stream_id) + " column " +
        std::to_string(i) + " (" + declared.names[i] + ") is declared " +
        declared.types[i].to_string() + " (" + cudf::type_to_name(expected) + ") but carries " +
        cudf::type_to_name(actual));
    }
  }
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
  //! Cross-node exchange staging (opt-in via SIRIUS_EXCHANGE_STAGING_BYTES; null otherwise, and
  //! every staging call errors loudly). Plain cudaMalloc by contract — see the arena's header.
  //! `shared_ptr` so a `StagingArena` handle can serve leases from other threads (the arena's
  //! internal mutex makes that safe) and outlive this context; there is still exactly ONE
  //! allocator — the handle shares it, never mirrors it.
  std::shared_ptr<sirius::exec::exchange_staging_arena> staging_arena;

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

    // After engine bring-up so the arena's cudaMalloc comes out of the headroom the operator
    // left beside the pool budget, not out of memory the pool then misses.
    staging_arena                  = sirius::exec::exchange_staging_arena::from_env();
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
  config.apply_defaults();  // populate default GPU/host/disk memory spaces
  impl_->bring_up(config);
}

Context::Context(const std::string& config_path) : impl_(std::make_unique<Impl>())
{
  sirius::sirius_config config;
  config.load_from_file(config_path);  // throws on a missing/invalid config file
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

std::uint64_t Context::staging_lease(std::uint64_t len)
{
  return sirius::exec::exchange_staging_arena::require(impl_->staging_arena.get()).lease(len);
}

void Context::staging_release(std::uint64_t offset)
{
  sirius::exec::exchange_staging_arena::require(impl_->staging_arena.get()).release(offset);
}

std::uintptr_t Context::staging_base() const
{
  return sirius::exec::exchange_staging_arena::require(impl_->staging_arena.get()).base();
}

std::uint64_t Context::staging_capacity() const
{
  return sirius::exec::exchange_staging_arena::require(impl_->staging_arena.get()).capacity();
}

std::unique_ptr<StagingArena> Context::staging_arena_handle() const
{
  if (impl_->staging_arena == nullptr) { return nullptr; }
  return std::make_unique<StagingArena>(impl_->staging_arena);
}

std::unique_ptr<Context> make_context() { return std::make_unique<Context>(); }

std::unique_ptr<Context> make_context_from_config(const std::string& config_path)
{
  return std::make_unique<Context>(config_path);
}

// ---------------------------------------------------------------------------
// StagingArena
// ---------------------------------------------------------------------------

// Thread-safety contract (documented on the class): every method below only touches the arena,
// whose lease/release serialize on its internal std::mutex and make no CUDA calls — so unlike
// the Context methods above, these are callable from any thread.

StagingArena::StagingArena(std::shared_ptr<sirius::exec::exchange_staging_arena> arena)
  : arena_(std::move(arena))
{
}

StagingArena::~StagingArena() = default;

std::uint64_t StagingArena::lease(std::uint64_t len) const { return arena_->lease(len); }

void StagingArena::release(std::uint64_t offset) const { arena_->release(offset); }

std::uintptr_t StagingArena::base() const noexcept { return arena_->base(); }

std::uint64_t StagingArena::capacity() const noexcept { return arena_->capacity(); }

std::size_t StagingArena::outstanding() const { return arena_->outstanding(); }

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

std::unique_ptr<std::vector<std::uint8_t>> Fragment::export_packed(std::uint64_t stream_id,
                                                                   std::uint64_t& offset,
                                                                   std::uint64_t& length,
                                                                   std::uint64_t& rows)
{
  impl_->require_built("export_packed()");
  if (impl_->fragment->is_result()) {
    throw sirius::invalid_input_exception(
      "Fragment: export_packed() requires an intermediate fragment with output streams — a result "
      "fragment produces Arrow via result_to_arrow()");
  }
  auto& arena = sirius::exec::exchange_staging_arena::require(impl_->ctx.staging_arena.get());

  offset = 0;
  length = 0;
  rows   = 0;
  // pull() rejects a fragment that has not run: an empty stream would look finished.
  auto batch = impl_->fragment->pull(stream_id);
  if (!batch) { return nullptr; }

  // The shared lock holds residency and immutability for the whole pack; it releases when this
  // scope ends, after the packing stream has been synchronized and the data lives in the lease.
  auto read_only = (*batch)->to_read_only();
  if (read_only.get_current_tier() != cucascade::memory::Tier::GPU) {
    throw sirius::invalid_input_exception(
      "Fragment: batch on output stream " + std::to_string(stream_id) +
      " is not GPU-resident; exporting a spilled batch is not supported yet");
  }
  auto view   = sirius::get_cudf_table_view(read_only);
  auto* space = read_only.get_memory_space();
  if (space == nullptr) {
    throw sirius::invalid_input_exception("Fragment: batch on output stream " +
                                          std::to_string(stream_id) + " has no memory space");
  }
  rows = static_cast<std::uint64_t>(view.num_rows());

  auto stream = cudf::get_default_stream();
  if (cudaEvent_t writer = read_only.get_writer_event()) {
    if (auto err = cudaStreamWaitEvent(stream.value(), writer, 0); err != cudaSuccess) {
      throw sirius::internal_exception("Fragment: cudaStreamWaitEvent failed: {}",
                                       cudaGetErrorString(err));
    }
  }

  auto packer =
    cudf::chunked_pack::create(view, kPackChunkBytes, stream, space->get_default_allocator());
  const std::uint64_t total = packer->get_total_contiguous_size();

  // A zero-row batch packs to a metadata-only frame: no payload, no lease. offset==0 with
  // length==0 means "no lease exists for this batch".
  if (total == 0) { return packer->build_metadata(); }

  const auto lease_offset = arena.lease(total + kPackChunkBytes);
  std::unique_ptr<std::vector<std::uint8_t>> metadata;
  try {
    auto* lease         = reinterpret_cast<std::uint8_t*>(arena.base()) + lease_offset;
    std::size_t written = 0;
    while (packer->has_next()) {
      written += packer->next(cudf::device_span<std::uint8_t>(lease + written, kPackChunkBytes));
    }
    if (written != total) {
      throw sirius::internal_exception(
        "Fragment: chunked_pack wrote {} of {} bytes for output stream {}",
        written,
        total,
        stream_id);
    }
    metadata = packer->build_metadata();
    stream.synchronize();
  } catch (...) {
    arena.release(lease_offset);
    throw;
  }
  offset = lease_offset;
  length = total;
  return metadata;
}

void Fragment::push_packed(std::uint64_t stream_id,
                           std::uintptr_t metadata_addr,
                           std::size_t metadata_len,
                           std::uint64_t offset,
                           std::uint64_t length)
{
  impl_->require_built("push_packed()");
  auto& arena = sirius::exec::exchange_staging_arena::require(impl_->ctx.staging_arena.get());
  if (metadata_addr == 0 || metadata_len == 0) {
    throw sirius::invalid_input_exception("Fragment: push_packed() requires pack metadata");
  }
  if (offset > arena.capacity() || length > arena.capacity() - offset) {
    throw sirius::invalid_input_exception(
      "Fragment: push_packed() range [{}, +{}) exceeds the staging arena capacity {}",
      offset,
      length,
      arena.capacity());
  }

  const auto& declared = impl_->fragment->input_spec(stream_id);

  const auto* metadata = reinterpret_cast<const std::uint8_t*>(metadata_addr);
  const auto* payload  = reinterpret_cast<const std::uint8_t*>(arena.base()) + offset;
  auto unpacked        = cudf::unpack(metadata, payload);
  check_declared_schema(declared, unpacked, stream_id, "packed batch");

  auto* gpu_space = impl_->ctx.context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, /*device_id=*/0);
  if (gpu_space == nullptr) {
    throw sirius::internal_exception("Fragment: push_packed() found no GPU memory space");
  }

  auto stream = cudf::get_default_stream();
  auto table  = std::make_unique<cudf::table>(unpacked, stream, gpu_space->get_default_allocator());
  stream.synchronize();

  auto data_batch = sirius::make_data_batch(
    std::move(table), *gpu_space, stream, telemetry::batch_telemetry_info{});
  bool const pushed = impl_->fragment->push(stream_id, std::move(data_batch));
  // Study path has no InboundStore: the receiver lease is consumed here. length==0 means the
  // sender never leased, so offset 0 must not be released.
  if (length != 0) { arena.release(offset); }
  if (!pushed) {
    throw sirius::invalid_input_exception("Fragment: input stream " + std::to_string(stream_id) +
                                          " refused a packed batch; it had already ended");
  }
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

bool Fragment::drained(std::uint64_t stream_id)
{
  impl_->require_built("drained()");
  if (impl_->fragment->is_result()) {
    throw sirius::invalid_input_exception(
      "Fragment: drained() is only valid on an intermediate fragment with output streams");
  }
  return impl_->fragment->drained(stream_id);
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
