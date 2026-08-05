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

// Implementation of the public FFI surface (sirius_ffi.hpp). This is the one
// translation unit that sees the heavy internal types, so consumers (e.g. the
// Rust bindings) never include sirius_context.hpp.

#include "sirius_ffi.hpp"

#include "data/data_batch_utils.hpp"                       // sirius::make_data_batch
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
#include "exec/streaming_fragment.hpp"    // sirius::exec::streaming_fragment
#include "from_substrait.hpp"             // duckdb::SubstraitToDuckDB (compiled into libsirius)
#include "helper/type_conversions.hpp"    // sirius::from_duckdb_vec
#include "planner/sirius_physical_plan_generator.hpp"  // sirius::planner::sirius_physical_plan_generator
#include "sirius_config.hpp"                           // sirius::sirius_config
#include "sirius_context.hpp"                          // duckdb::SiriusContext
#include "sirius_interface.hpp"  // sirius::sirius_interface, sirius::sirius_prepared_statement_data

#include <cudf/contiguous_split.hpp>          // cudf::chunked_pack, cudf::unpack
#include <cudf/table/table.hpp>               // cudf::table
#include <cudf/utilities/default_stream.hpp>  // cudf::get_default_stream
#include <cudf/utilities/span.hpp>            // cudf::device_span

#include <cuda_runtime_api.h>  // cudaStreamWaitEvent

#include <algorithm>  // std::find
#include <cstdlib>    // std::getenv
#include <map>
#include <vector>

namespace sirius::ffi {

namespace {
// ClientContextState key the GPU engine resolves its SiriusContext under.
constexpr const char* kSiriusStateKey = "sirius_state";
constexpr const char* kQueryLabel     = "sirius_ffi";
// Arrow record-batch size for the exported stream; the consumer re-batches as needed.
constexpr duckdb::idx_t kArrowBatchSize = 1u << 20;

/// A Substrait plan lowered to the pair the Sirius planner needs: a bound, optimized DuckDB
/// logical plan and the statement metadata that travels with it.
struct lowered_plan {
  duckdb::unique_ptr<duckdb::LogicalOperator> plan;
  duckdb::shared_ptr<duckdb::PreparedStatementData> prepared;
};

/// Substrait bytes -> bound + optimized DuckDB `LogicalOperator`. DuckDB is used only for this
/// lowering; execution runs on the Sirius engine. Shared by the single-shot path and by
/// `Fragment`, which needs the same lowering with its input streams declared first.
lowered_plan lower_substrait(duckdb::Connection& conn, const std::string& plan_bytes)
{
  auto& client = *conn.context;

  duckdb::SubstraitToDuckDB transformer(conn.context, plan_bytes, /*json=*/false);
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

  return lowered_plan{std::move(logical_plan), std::move(prepared)};
}
}  // namespace

// PIMPL: holds the engine + embedded DuckDB using DuckDB's own smart pointers
// (duckdb::shared_ptr, so the SiriusContext can register as a ClientContextState).
struct detail::context_state {
  duckdb::shared_ptr<duckdb::SiriusContext> context;
  duckdb::unique_ptr<duckdb::DuckDB> db;
  duckdb::unique_ptr<duckdb::Connection> conn;
  //! Input streams declared for the fragment currently being planned. Held here as well as on the
  //! connection so a fragment can populate it without re-resolving it out of registered_state.
  duckdb::shared_ptr<sirius::exec::stream_bind_catalog> stream_catalog;
  //! Cross-node exchange staging (opt-in via SIRIUS_EXCHANGE_STAGING_BYTES; null otherwise, and
  //! every staging call errors loudly). Plain cudaMalloc by contract — see the arena's header.
  std::unique_ptr<sirius::exec::exchange_staging_arena> staging_arena;

  void bring_up(sirius::sirius_config& config)
  {
    context = duckdb::make_shared_ptr<duckdb::SiriusContext>();
    context->initialize(config);
    // Register the builtin + parquet representation converters the GPU scan/result
    // path needs. Idempotent; the transparent path does this at extension load.
    sirius::converter_registry::initialize();

    // The embedded DuckDB needs parquet for local_files reads and core_functions for
    // Substrait scalar/aggregate bindings. Only explicitly configured local extensions opt into
    // unsigned loading; absent both variables, the default trust boundary is unchanged.
    duckdb::DBConfig db_config;
    const char* parquet_ext        = std::getenv("SIRIUS_DUCKDB_PARQUET_EXTENSION");
    const char* core_functions_ext = std::getenv("SIRIUS_DUCKDB_CORE_FUNCTIONS_EXTENSION");
    if (parquet_ext != nullptr || core_functions_ext != nullptr) {
      db_config.SetOptionByName("allow_unsigned_extensions", duckdb::Value::BOOLEAN(true));
    }
    db   = duckdb::make_uniq<duckdb::DuckDB>(nullptr, &db_config);
    conn = duckdb::make_uniq<duckdb::Connection>(*db);
    for (const char* extension : {core_functions_ext, parquet_ext}) {
      if (extension == nullptr) { continue; }
      // Escape single quotes so the path can't break out of the SQL string literal.
      std::string escaped(extension);
      for (std::size_t pos = escaped.find('\''); pos != std::string::npos;
           pos             = escaped.find('\'', pos + 2)) {
        escaped.replace(pos, 1, "''");
      }
      auto load = conn->Query("LOAD '" + escaped + "'");
      if (load->HasError()) { load->ThrowError(); }
    }
    // Register the engine on the connection and disable DuckDB optimizer rewrites this
    // no-fallback FFI path cannot safely execute. The transparent path only disables
    // IN_CLAUSE and COMPRESSED_MATERIALIZATION here; this path remains more conservative.
    auto& client = *conn->context;
    client.registered_state->Insert(kSiriusStateKey, context);
    // Per-connection Sirius state (guard depths, capture bookkeeping) for the
    // embedded connection, mirroring OnConnectionOpened on the transparent path.
    client.registered_state->Insert("sirius_connection_state",
                                    duckdb::make_shared_ptr<duckdb::SiriusConnectionState>());

    // A fragment reads each of its exchange inputs through sirius_stream_source(id). The
    // function has to exist on this DuckDB before any fragment plan binds, and its bind needs
    // somewhere to look up a schema for a stream that has no file behind it — the catalog,
    // reached the same way the engine reaches its SiriusContext.
    stream_catalog = duckdb::make_shared_ptr<sirius::exec::stream_bind_catalog>();
    client.registered_state->Insert(sirius::exec::stream_bind_catalog::kStateKey, stream_catalog);
    sirius::exec::register_stream_source_function(*db->instance);

    // After engine bring-up so the arena's cudaMalloc comes out of the headroom the operator
    // left beside the pool budget (PLAN-PATH-B D-B4), not out of memory the pool then misses.
    staging_arena = sirius::exec::exchange_staging_arena::from_env();

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

Context::Context() : impl_(std::make_unique<detail::context_state>())
{
  sirius::sirius_config config;
  config.apply_defaults();  // populate default GPU/host/disk memory spaces
  impl_->bring_up(config);
}

Context::Context(const std::string& config_path) : impl_(std::make_unique<detail::context_state>())
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
    // 1+2. Substrait → optimized DuckDB LogicalOperator.
    auto lowered = lower_substrait(*impl_->conn, plan);

    // 3. DuckDB LogicalOperator -> Sirius GPU physical plan -> execute directly
    // on the engine, inside an execution window: begin mutations and slot
    // acquire in the constructor, mandatory cleanup and release in finish().
    // This standalone path bypasses DuckDB's normal query entry point, so
    // nothing else would clean up for it. (The old manual QueryBegin/QueryEnd
    // pairing could call QueryEnd twice when the first cleanup threw.)
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

  // 4. Hand the result to the caller as a self-owning Arrow C Data Interface stream,
  //    written into the caller's ArrowArrayStream (addressed by `out_stream_addr`);
  //    its `release` callback deletes the heap wrapper (ResultArrowArrayStreamWrapper).
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

namespace {
std::string stream_view_name_of(std::uint64_t stream_id)
{
  return "sirius_stream_" + std::to_string(stream_id);
}
}  // namespace

std::unique_ptr<std::string> stream_view_name(std::uint64_t stream_id)
{
  return std::make_unique<std::string>(stream_view_name_of(stream_id));
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

struct Fragment::Impl {
  explicit Impl(detail::context_state& ctx) : ctx(ctx) {}

  ~Impl()
  {
    // A fragment dropped between build() and run() still holds the context's query-lifecycle
    // mutex. Releasing it here keeps a failed fragment from wedging every later statement on
    // this connection — the failure mode is a silent deadlock, so it must not depend on the
    // caller remembering to run().
    end_lifecycle();
  }

  detail::context_state& ctx;

  /// One input stream as the caller declared it. The column types are kept as DuckDB type
  /// *names* and parsed in build(): parsing needs an active transaction (it may resolve a
  /// user-defined type through the catalog), and a declaration happens before there is one.
  struct declared_input {
    std::vector<std::string> names;
    std::vector<std::string> type_names;
    std::set<sirius::exec::sender_id_t> expected_senders;
  };

  //! Declared before build(); the map is what the bind catalog is populated from.
  std::map<sirius::exec::stream_id_t, declared_input> inputs;
  std::vector<sirius::exec::stream_id_t> outputs;

  //! Intermediate fragment: a streaming sink root, owning its own repositories and session.
  std::unique_ptr<sirius::exec::streaming_fragment> fragment;

  //! Result fragment: a RESULT_COLLECTOR root. The input repositories, the session borrowing the
  //! built sources out of the plan, and the plan itself all have to outlive build().
  std::map<sirius::exec::stream_id_t, std::shared_ptr<cucascade::shared_data_repository>>
    result_input_repos;
  sirius::exec::stream_session result_session;
  duckdb::shared_ptr<sirius::sirius_prepared_statement_data> result_plan;
  duckdb::unique_ptr<duckdb::QueryResult> result;

  bool built{false};
  bool ran{false};
  bool transaction_open{false};

  // Heap-allocated because StandaloneQueryScope is non-movable. Opened in build(), closed in
  // run() / ~Impl(). QueryBeginStandalone is gone on this base — the scope is the lifecycle.
  std::unique_ptr<duckdb::SiriusContext::StandaloneQueryScope> lifecycle;

  [[nodiscard]] bool is_result() const { return outputs.empty(); }

  sirius::exec::stream_session& session()
  {
    return fragment ? fragment->session() : result_session;
  }

  //! Idempotent, and never throws: it runs from run(), from the error path, and from ~Impl.
  void end_lifecycle() noexcept
  {
    if (lifecycle) {
      try {
        // Destructor backstop cleans up if finish() never ran.
        lifecycle.reset();
      } catch (...) {  // NOLINT(bugprone-empty-catch)
      }
    }
    if (transaction_open) {
      transaction_open = false;
      try {
        ctx.conn->Commit();
      } catch (...) {  // NOLINT(bugprone-empty-catch)
      }
    }
  }

  void require_not_built(const char* what) const
  {
    if (built) {
      throw sirius::invalid_input_exception(std::string("Fragment: ") + what +
                                            " must be called before build()");
    }
  }

  //! Resolves each declared stream to a `stream_input_spec`, parsing its column type names.
  //! Requires an open transaction.
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
      // A stream with no declared sender expects the single sender 0, which is the gather case.
      if (spec.expected_senders.empty()) { spec.expected_senders.insert(0); }
      resolved.emplace(id, std::move(spec));
    }
    return resolved;
  }

  //! Creates the view each declared stream is read through. Runs *before* the query lifecycle is
  //! opened, because creating a view is an ordinary DuckDB statement and an ordinary statement
  //! takes the lifecycle mutex that StandaloneQueryScope holds — and *after* the streams are
  //! declared, because DuckDB binds a view's body at CREATE time, which resolves the stream's
  //! schema out of the bind catalog.
  void create_stream_views()
  {
    for (const auto& [id, _] : inputs) {
      auto view   = stream_view_name_of(id);
      auto create = ctx.conn->Query("CREATE OR REPLACE VIEW main." + view + " AS SELECT * FROM " +
                                    std::string(sirius::exec::kStreamSourceFunctionName) + "(" +
                                    std::to_string(id) + ")");
      if (create->HasError()) { create->ThrowError(); }
    }
  }

  //! Declares every input stream on the connection's bind catalog, with a repository this
  //! fragment owns.
  //!
  //! Both paths need the declaration before the view is created. The result path then keeps these
  //! repositories, because nothing else creates them; on the streaming path
  //! streaming_fragment::build() clears the catalog and redeclares the same schemas against its
  //! own repositories, and these are dropped unused.
  void declare_streams(
    const std::map<sirius::exec::stream_id_t, sirius::exec::stream_input_spec>& resolved)
  {
    auto& catalog = *ctx.stream_catalog;
    catalog.clear();
    for (const auto& [id, spec] : resolved) {
      auto repository = std::make_shared<cucascade::shared_data_repository>();
      if (is_result()) { result_input_repos[id] = repository; }
      catalog.declare(id,
                      sirius::exec::stream_input_binding{
                        spec.names, spec.types, repository, spec.expected_senders, nullptr});
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
  auto& declared = impl_->inputs[stream_id];
  declared.names.push_back(name);
  declared.type_names.push_back(type);
}

void Fragment::declare_input_sender(std::uint64_t stream_id, std::uint32_t sender_id)
{
  impl_->require_not_built("declare_input_sender");
  impl_->inputs[stream_id].expected_senders.insert(sender_id);
}

void Fragment::declare_output(std::uint64_t stream_id)
{
  impl_->require_not_built("declare_output");
  auto& outputs = impl_->outputs;
  if (std::find(outputs.begin(), outputs.end(), stream_id) != outputs.end()) {
    throw sirius::invalid_input_exception("Fragment: duplicate output stream id " +
                                          std::to_string(stream_id));
  }
  outputs.push_back(stream_id);
}

void Fragment::build(const std::string& substrait_plan)
{
  impl_->require_not_built("build");

  // Ordering: type-name parsing and CREATE VIEW need a transaction and must precede the
  // StandaloneQueryScope (ordinary statements take the lifecycle mutex the scope holds). The
  // window then spans planning and execution, as streaming_fragment::run() requires.
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

  auto& client     = *impl_->ctx.conn->context;
  impl_->lifecycle = std::make_unique<duckdb::SiriusContext::StandaloneQueryScope>(
    *impl_->ctx.context, client, kQueryLabel);

  try {
    if (impl_->is_result()) {
      // A result fragment takes the ordinary single-shot path; the only difference is that its
      // leaves may be streaming sources, which the plan generator built from the bind catalog.
      auto lowered       = lower_substrait(*impl_->ctx.conn, substrait_plan);
      auto physical_plan = sirius::planner::sirius_physical_plan_generator(client).create_plan(
        std::move(lowered.plan));
      impl_->result_plan = duckdb::make_shared_ptr<sirius::sirius_prepared_statement_data>(
        std::move(lowered.prepared), std::move(physical_plan));

      for (const auto& [id, _] : impl_->inputs) {
        auto* built = impl_->ctx.stream_catalog->get(id).built;
        if (built == nullptr) {
          throw sirius::invalid_input_exception("Fragment: input stream " + std::to_string(id) +
                                                " was declared but the plan does not read it");
        }
        impl_->result_session.add_source(id, *built);
      }
    } else {
      sirius::exec::fragment_spec spec;
      spec.plan_source = [plan = substrait_plan, conn = impl_->ctx.conn.get()](
                           duckdb::ClientContext&) { return lower_substrait(*conn, plan).plan; };
      spec.inputs     = std::move(resolved);
      spec.outputs    = impl_->outputs;
      impl_->fragment = std::make_unique<sirius::exec::streaming_fragment>(client, std::move(spec));
      // streaming_fragment declares its own inputs on the catalog from the spec; the views
      // created above are what the plan reads them through.
      impl_->fragment->build(impl_->lifecycle->query_id());
    }
    impl_->built = true;
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
  if (!impl_->built) {
    throw sirius::invalid_input_exception("Fragment: build() must run before relay_from()");
  }
  std::size_t moved = 0;
  while (auto batch = source.impl_->session().pull(source_stream_id)) {
    if (!impl_->session().push(input_stream_id, *batch)) {
      throw sirius::invalid_input_exception("Fragment: input stream " +
                                            std::to_string(input_stream_id) +
                                            " refused a batch; it had already ended");
    }
    ++moved;
  }
  impl_->session().close_input(input_stream_id, sender_id);
  return moved;
}

namespace {
/// chunked_pack gather granularity. Every `next()` span must be exactly this long, so a lease is
/// the payload plus one chunk of slack for the final span. 1 MiB is cudf's minimum.
constexpr std::size_t kPackChunkBytes = 8u << 20;
}  // namespace

std::unique_ptr<std::vector<std::uint8_t>> Fragment::export_packed(std::uint64_t stream_id,
                                                                   std::uint64_t& offset,
                                                                   std::uint64_t& length)
{
  if (!impl_->built) {
    throw sirius::invalid_input_exception("Fragment: build() must run before export_packed()");
  }
  auto& arena = sirius::exec::exchange_staging_arena::require(impl_->ctx.staging_arena.get());

  offset     = 0;
  length     = 0;
  auto batch = impl_->session().pull(stream_id);
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

  auto stream = cudf::get_default_stream();
  // STREAM-LINEAGE: order the pack's gather after the batch's writer.
  if (cudaEvent_t writer = read_only.get_writer_event()) {
    if (auto err = cudaStreamWaitEvent(stream.value(), writer, 0); err != cudaSuccess) {
      throw sirius::internal_exception("Fragment: cudaStreamWaitEvent failed: {}",
                                       cudaGetErrorString(err));
    }
  }

  auto packer =
    cudf::chunked_pack::create(view, kPackChunkBytes, stream, space->get_default_allocator());
  const std::uint64_t total = packer->get_total_contiguous_size();

  // Each next() span is a full chunk long and starts where the previous copy ended, so the
  // final span can reach up to one chunk past the payload — hence the slack.
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
    // The caller transmits from the lease the moment this returns.
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
  if (!impl_->built) {
    throw sirius::invalid_input_exception("Fragment: build() must run before push_packed()");
  }
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

  const auto* metadata = reinterpret_cast<const std::uint8_t*>(metadata_addr);
  const auto* payload  = reinterpret_cast<const std::uint8_t*>(arena.base()) + offset;
  // Allocates no device memory: the view aliases the lease until the deep copy below.
  auto unpacked = cudf::unpack(metadata, payload);

  auto* gpu_space = impl_->ctx.context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, /*device_id=*/0);
  if (gpu_space == nullptr) {
    throw sirius::internal_exception("Fragment: push_packed() found no GPU memory space");
  }

  // Copy-out-on-arrival (PLAN-PATH-B D-B5): the batch the engine keeps lives in ordinary pool
  // memory, so the lease is reusable the moment this call returns and the batch is fully
  // accounted and spillable like any other.
  auto stream = cudf::get_default_stream();
  auto table  = std::make_unique<cudf::table>(unpacked, stream, gpu_space->get_default_allocator());
  stream.synchronize();

  // A wire batch has no local producing operator, so there is no telemetry lineage to thread.
  auto data_batch = sirius::make_data_batch(
    std::move(table), *gpu_space, stream, telemetry::batch_telemetry_info{});
  if (!impl_->session().push(stream_id, std::move(data_batch))) {
    throw sirius::invalid_input_exception("Fragment: input stream " + std::to_string(stream_id) +
                                          " refused a packed batch; it had already ended");
  }
}

void Fragment::close_input(std::uint64_t stream_id, std::uint32_t sender_id)
{
  if (!impl_->built) {
    throw sirius::invalid_input_exception("Fragment: build() must run before close_input()");
  }
  impl_->session().close_input(stream_id, sender_id);
}

void Fragment::run()
{
  if (!impl_->built) {
    throw sirius::invalid_input_exception("Fragment: build() must run before run()");
  }
  if (impl_->ran) { throw sirius::invalid_input_exception("Fragment: already run"); }

  try {
    if (impl_->is_result()) {
      auto& client = *impl_->ctx.conn->context;
      sirius::sirius_interface iface(client, std::optional<std::string>(kQueryLabel));
      impl_->result = iface.sirius_execute_query(client,
                                                 kQueryLabel,
                                                 impl_->result_plan,
                                                 duckdb::PendingQueryParameters{},
                                                 impl_->lifecycle->query_id());
    } else {
      impl_->fragment->run();
    }
    impl_->ran = true;
  } catch (...) {
    impl_->end_lifecycle();
    throw;
  }
  impl_->lifecycle->finish();
  impl_->lifecycle.reset();
  if (impl_->result && impl_->result->HasError()) { impl_->result->ThrowError(); }
}

void Fragment::result_to_arrow(std::uintptr_t out_stream_addr)
{
  if (!impl_->is_result()) {
    throw sirius::invalid_input_exception(
      "Fragment: result_to_arrow() is only valid on a fragment with no output stream");
  }
  if (!impl_->ran || !impl_->result) {
    throw sirius::invalid_input_exception("Fragment: run() must complete before result_to_arrow()");
  }
  auto* wrapper =
    new duckdb::ResultArrowArrayStreamWrapper(std::move(impl_->result), kArrowBatchSize);
  *reinterpret_cast<ArrowArrayStream*>(out_stream_addr) = wrapper->stream;
}

std::size_t Fragment::output_batch_count(std::uint64_t stream_id) const
{
  if (!impl_->fragment) { return 0; }
  return impl_->fragment->output_repository(stream_id)->total_size();
}

std::unique_ptr<Context> make_context() { return std::make_unique<Context>(); }

std::unique_ptr<Context> make_context_from_config(const std::string& config_path)
{
  return std::make_unique<Context>(config_path);
}

std::unique_ptr<Fragment> make_fragment(Context& context)
{
  return std::unique_ptr<Fragment>(new Fragment(std::make_unique<Fragment::Impl>(*context.impl_)));
}

}  // namespace sirius::ffi
