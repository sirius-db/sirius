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

// Implementation of the public FFI surface (sirius_ffi.hpp). This is the host
// process: Context plus Fragment. Not GPU host memory. This translation unit
// sees the heavy internal types so consumers (e.g. the Rust bindings) never
// include sirius_context.hpp.

#include "sirius_ffi.hpp"

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
#include "parquet_extension.hpp"          // duckdb::ParquetExtension
#include "planner/sirius_physical_plan_generator.hpp"  // sirius::planner::sirius_physical_plan_generator
#include "sirius_config.hpp"                           // sirius::sirius_config
#include "sirius_context.hpp"                          // duckdb::SiriusContext
#include "sirius_interface.hpp"  // sirius::sirius_interface, sirius::sirius_prepared_statement_data

#include <map>
#include <memory>
#include <set>

namespace sirius::ffi {

namespace {

constexpr const char* kSiriusStateKey   = "sirius_state";
constexpr const char* kQueryLabel       = "sirius_ffi";
constexpr duckdb::idx_t kArrowBatchSize = 1u << 20;

// DuckDB view name a plan uses to read input stream `id`.
std::string stream_view_name_of(std::uint64_t id) { return "sirius_stream_" + std::to_string(id); }

// Lower a Substrait plan to a bound+optimized DuckDB LogicalOperator.
struct lowered_plan {
  duckdb::shared_ptr<duckdb::PreparedStatementData> prepared;
  duckdb::unique_ptr<duckdb::LogicalOperator> plan;
};

lowered_plan lower_substrait(duckdb::Connection& conn, const std::string& substrait_plan)
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

  return {std::move(prepared), std::move(logical_plan)};
}

}  // namespace

// PIMPL: holds the engine + embedded DuckDB using DuckDB's own smart pointers
// (duckdb::shared_ptr, so the SiriusContext can register as a ClientContextState).
struct Context::Impl {
  duckdb::shared_ptr<duckdb::SiriusContext> context;
  duckdb::unique_ptr<duckdb::DuckDB> db;
  duckdb::unique_ptr<duckdb::Connection> conn;
  //! Also stored in registered_state. Held here so it outlives registered_state resets.
  duckdb::shared_ptr<sirius::exec::stream_bind_catalog> stream_catalog;

  void bring_up(sirius::sirius_config& config)
  {
    sirius::converter_registry::initialize(config.get_downgrade_executor_config().copy_chunk_bytes);
    context = duckdb::make_shared_ptr<duckdb::SiriusContext>();
    context->initialize(config);

    // Substrait lowering uses core functions and resolves local_files reads to parquet_scan.
    db = duckdb::make_uniq<duckdb::DuckDB>(nullptr);
    db->LoadStaticExtension<duckdb::CoreFunctionsExtension>();
    db->LoadStaticExtension<duckdb::ParquetExtension>();
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
    stream_catalog = duckdb::make_shared_ptr<sirius::exec::stream_bind_catalog>();
    client.registered_state->Insert(sirius::exec::stream_bind_catalog::kStateKey, stream_catalog);
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

  // Binding (catalog lookups), optimization, and GPU execution all require an
  // active transaction. The transparent table-function path inherits one from the
  // enclosing query; this standalone entry has none, so open one explicitly. GPU
  // execution is eager (the result is materialized), so the transaction can close
  // before the Arrow stream is consumed.
  impl_->conn->BeginTransaction();
  duckdb::unique_ptr<duckdb::QueryResult> result;
  try {
    // Lower Substrait to an optimized DuckDB LogicalOperator.
    auto lowered = lower_substrait(*impl_->conn, plan);

    // Lower the LogicalOperator to a Sirius GPU plan and execute it. StandaloneQueryScope
    // begins mutations and takes the lifecycle slot in its constructor, then cleans up in
    // finish(). This path does not go through DuckDB's normal query entry, so nothing else
    // would clean up. The old QueryBegin/QueryEnd pairing could call QueryEnd twice when
    // the first cleanup threw.
    {
      duckdb::SiriusContext::StandaloneQueryScope window(*impl_->context, client, kQueryLabel);
      auto physical_plan = sirius::planner::sirius_physical_plan_generator(client).create_plan(
        std::move(lowered.plan));
      auto gpu_prepared = duckdb::make_shared_ptr<sirius::sirius_prepared_statement_data>(
        std::move(lowered.prepared), std::move(physical_plan));

      sirius::sirius_interface iface(client, std::optional<std::string>(kQueryLabel));
      result = iface.sirius_execute_query(
        client, kQueryLabel, gpu_prepared, duckdb::PendingQueryParameters{}, window.query_id());
      window.finish();
    }
  } catch (...) {
    impl_->conn->Rollback();
    throw;
  }
  impl_->conn->Commit();
  if (result->HasError()) { result->ThrowError(); }

  // Write the result into the caller's ArrowArrayStream at out_stream_addr.
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

// Host-process PIMPL around one streaming_fragment. Not GPU host memory.
struct Fragment::Impl {
  explicit Impl(Context::Impl& ctx) : ctx(ctx) {}

  ~Impl()
  {
    // If the caller dropped a Fragment during an open setup or lowering transaction, roll
    // it back. Destroying `fragment` closes the query window.
    end_lifecycle();
  }

  Context::Impl& ctx;

  // One column declared before build(). type_name is the DuckDB type string parsed at
  // build() time. Parsing may need a catalog lookup, so it must run inside a transaction.
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

  bool built{false};
  bool ran{false};
  bool transaction_open{false};

  [[nodiscard]] bool is_result() const { return outputs.empty(); }

  void require_not_built(const char* what) const
  {
    if (built) {
      throw sirius::invalid_input_exception(std::string("Fragment: ") + what +
                                            " must be called before build()");
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

  // Fill the bind catalog so DuckDB can bind a view of each declared input stream.
  // These repositories are throwaways. streaming_fragment::build() erases the ids and
  // redeclares them on its own repositories. Do not call catalog.clear(). This catalog is
  // shared by every Fragment on the same Context.
  void declare_streams(
    const std::map<sirius::exec::stream_id_t, sirius::exec::stream_input_spec>& resolved)
  {
    auto& catalog = *ctx.stream_catalog;
    for (const auto& [id, spec] : resolved) {
      auto repository = std::make_shared<cucascade::shared_data_repository>();
      catalog.declare(id,
                      sirius::exec::stream_input_binding{
                        spec.names, spec.types, repository, spec.expected_senders, nullptr});
    }
  }

  // CREATE OR REPLACE VIEW sirius_stream_<id> AS SELECT * FROM sirius_stream_source(<id>)
  // Needs an open transaction. Call after declare_streams so bind can resolve the schema.
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

  // Idempotent. Called from build() catch blocks and ~Impl(). streaming_fragment owns the
  // query window, so this only rolls back a still-open DuckDB transaction.
  void end_lifecycle() noexcept
  {
    if (transaction_open) {
      transaction_open = false;
      // Reached only while the transaction is still open, which means setup or lowering
      // failed. build() clears the flag after its own Commit succeeds. Committing here
      // would keep a half-declared fragment, or throw from a noexcept path.
      try {
        ctx.conn->Rollback();
      } catch (...) {  // NOLINT(bugprone-empty-catch)
      }
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

  // Type-name parsing and CREATE VIEW need a transaction. Commit it before
  // streaming_fragment::build() opens StandaloneQueryScope, or QueryBeginStandalone waits
  // on a slot this connection still holds.
  impl_->ctx.conn->BeginTransaction();
  impl_->transaction_open = true;
  std::map<sirius::exec::stream_id_t, sirius::exec::stream_input_spec> resolved;
  try {
    resolved = impl_->resolve_inputs();
    impl_->declare_streams(resolved);
    impl_->create_stream_views();
    impl_->ctx.conn->Commit();
    impl_->transaction_open = false;
  } catch (...) {
    impl_->end_lifecycle();
    throw;
  }

  try {
    // Broadcast or hash with 0 or 1 outputs does not change where rows go. Reject it so a
    // result fragment with a hash key is not treated as routed.
    if (impl_->outputs.size() <= 1 &&
        (impl_->broadcast_outputs || !impl_->hash_key_columns.empty())) {
      throw sirius::invalid_input_exception(
        "Fragment: a partition mode was declared but the fragment has " +
        std::to_string(impl_->outputs.size()) +
        " output stream(s); routing needs at least two destinations");
    }

    auto& client = *impl_->ctx.conn->context;

    // Substrait lowering binds parquet_scan and views, so it needs an active transaction,
    // same as Context::execute_substrait. Commit setup first, then open this short
    // transaction. streaming_fragment::build() opens the query window while it is still open.
    impl_->ctx.conn->BeginTransaction();
    impl_->transaction_open = true;

    auto lowered = lower_substrait(*impl_->ctx.conn, substrait_plan);
    auto plan_holder =
      std::make_shared<duckdb::unique_ptr<duckdb::LogicalOperator>>(std::move(lowered.plan));

    sirius::exec::fragment_spec spec;
    spec.plan_source = [plan_holder](duckdb::ClientContext&) {
      if (!*plan_holder) {
        throw sirius::invalid_input_exception("Fragment: plan source invoked twice");
      }
      return std::move(*plan_holder);
    };
    spec.inputs   = std::move(resolved);
    spec.outputs  = impl_->outputs;
    spec.prepared = std::move(lowered.prepared);

    if (impl_->broadcast_outputs && impl_->outputs.size() > 1) {
      sirius::op::partition_spec broadcast;
      broadcast.mode    = sirius::op::partition_mode::broadcast;
      spec.partitioning = std::move(broadcast);
    } else if (!impl_->hash_key_columns.empty() && impl_->outputs.size() > 1) {
      // key_cast_types left empty. streaming_fragment::build() fills them from output types.
      sirius::op::partition_spec hash;
      hash.mode         = sirius::op::partition_mode::hash;
      hash.key_columns  = impl_->hash_key_columns;
      spec.partitioning = std::move(hash);
    }

    impl_->fragment = std::make_unique<sirius::exec::streaming_fragment>(client, std::move(spec));
    impl_->fragment->build();

    impl_->ctx.conn->Commit();
    impl_->transaction_open = false;
    impl_->built            = true;
  } catch (...) {
    impl_->end_lifecycle();
    throw;
  }
}

std::size_t Fragment::relay_from(Fragment& source,
                                 std::uint64_t source_stream_id,
                                 std::uint64_t input_stream_id,
                                 std::uint32_t sender_id)
{
  if (!impl_->built || !impl_->fragment) {
    throw sirius::invalid_input_exception("Fragment: build() must run before relay_from()");
  }
  if (!source.impl_->built || !source.impl_->fragment) {
    throw sirius::invalid_input_exception(
      "Fragment: relay_from() requires the source fragment to have been built");
  }
  // Keep this layer so the public error says "Fragment:" rather than "streaming_fragment:".
  if (!source.impl_->ran) {
    throw sirius::invalid_input_exception(
      "Fragment: relay_from() requires the source fragment to have run — call source.run() first, "
      "otherwise an empty stream is indistinguishable from a finished one and the input would be "
      "closed early");
  }
  if (source.impl_->is_result()) {
    throw sirius::invalid_input_exception(
      "Fragment: relay source has no output streams — a result fragment produces Arrow via "
      "result_to_arrow(), not a relayable stream");
  }

  return impl_->fragment->relay_from(
    *source.impl_->fragment, source_stream_id, input_stream_id, sender_id);
}

void Fragment::close_input(std::uint64_t stream_id, std::uint32_t sender_id)
{
  if (!impl_->built || !impl_->fragment) {
    throw sirius::invalid_input_exception("Fragment: build() must run before close_input()");
  }
  impl_->fragment->close_input(stream_id, sender_id);
}

void Fragment::run()
{
  if (!impl_->built || !impl_->fragment) {
    throw sirius::invalid_input_exception("Fragment: build() must run before run()");
  }
  if (impl_->ran) { throw sirius::invalid_input_exception("Fragment: already run"); }

  try {
    impl_->fragment->run();
    impl_->ran = true;
  } catch (...) {
    // Fail every output before unwinding. Otherwise a peer in wait() blocks forever.
    // streaming_fragment::run() already poisons and closes the window. Repeat here so
    // ffi::Fragment::run still fails peers if that path changes. First failure wins. A
    // result fragment has no outputs, so the loop is a no-op there.
    auto const cause = std::current_exception();
    for (auto id : impl_->outputs) {
      try {
        impl_->fragment->fail_output(id, cause);
      } catch (...) {  // NOLINT(bugprone-empty-catch)
      }
    }
    throw;
  }
}

void Fragment::result_to_arrow(std::uintptr_t out_stream_addr)
{
  if (!impl_->is_result()) {
    throw sirius::invalid_input_exception(
      "Fragment: result_to_arrow() is only valid on a fragment with no output streams");
  }
  if (!impl_->ran || !impl_->fragment) {
    throw sirius::invalid_input_exception("Fragment: run() must complete before result_to_arrow()");
  }
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
  if (!impl_->built || !impl_->fragment) {
    throw sirius::invalid_input_exception("Fragment: build() must run before output_types()");
  }
  if (impl_->is_result()) {
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
