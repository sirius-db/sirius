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

#include "config.hpp"                                    // duckdb::Config::LOG_*
#include "core_functions_extension.hpp"                  // duckdb::CoreFunctionsExtension
#include "data/sirius_converter_registry.hpp"            // sirius::converter_registry
#include "downgrade/downgrade_executor.hpp"              // sirius::parallel::downgrade_executor
#include "duckdb/common/enums/optimizer_type.hpp"        // duckdb::OptimizerType
#include "duckdb/execution/column_binding_resolver.hpp"  // duckdb::ColumnBindingResolver
#include "duckdb/main/client_context.hpp"                // duckdb::ClientContext
#include "duckdb/main/config.hpp"                        // duckdb::DBConfig
#include "duckdb/main/connection.hpp"                    // duckdb::Connection
#include "duckdb/main/database.hpp"                      // duckdb::DuckDB
#include "duckdb/main/prepared_statement_data.hpp"       // duckdb::PreparedStatementData
#include "duckdb/main/query_result.hpp"                  // duckdb::QueryResult
#include "duckdb/main/relation.hpp"                      // duckdb::Relation
#include "duckdb/main/settings.hpp"
#include "duckdb/optimizer/optimizer.hpp"                  // duckdb::Optimizer
#include "duckdb/parser/statement/relation_statement.hpp"  // duckdb::RelationStatement
#include "duckdb/planner/planner.hpp"                      // duckdb::Planner
#include "exec/arrow_result_stream.hpp"
#include "exec/exchange_inbox.hpp"        // sirius::exec::exchange_inbox
#include "exec/exchange_ring_ends.hpp"    // sirius::exec::ring_exchange
#include "exec/stream_bind_catalog.hpp"   // sirius::exec::stream_bind_catalog
#include "exec/stream_plan_bindings.hpp"  // sirius::exec::register_stream_source_function
#include "exec/streaming_fragment.hpp"    // sirius::exec::streaming_fragment, fragment_spec
#include "from_substrait.hpp"             // duckdb::SubstraitToDuckDB (compiled into libsirius)
#include "helper/type_conversions.hpp"    // sirius::from_duckdb
#include "log/logging.hpp"                // SIRIUS_LOG_INFO
#include "parquet_extension.hpp"          // duckdb::ParquetExtension
#include "planner/unsupported_optimizers.hpp"
#include "sirius/ffi.hpp"
#include "sirius_config.hpp"   // sirius::sirius_config
#include "sirius_context.hpp"  // duckdb::SiriusContext

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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
  // Serialize the complete settings update, including rollback if sink creation fails.
  std::lock_guard log_lock(duckdb::Config::logging_mutex);
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
  if (duckdb::Settings::Get<duckdb::EnableOptimizerSetting>(client)) {
    duckdb::Optimizer optimizer(*planner.binder, client);
    logical_plan = optimizer.Optimize(std::move(logical_plan));
  }
  logical_plan->ResolveOperatorTypes();
  duckdb::ColumnBindingResolver resolver;
  duckdb::ColumnBindingResolver verifier(true);
  verifier.VisitOperator(*logical_plan);
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
  // Shared with every ExchangeInbox handle, which may outlive this Context.
  std::shared_ptr<sirius::exec::exchange_inbox> exchange;
  // The ring exchange, when the context was made with one; shared with RingExchange handles.
  std::shared_ptr<sirius::exec::ring_exchange> ring;

  // Frees the exchange's buffers while the memory space they come from still exists.
  ~Impl()
  {
    if (ring) { ring->close(); }
    if (exchange) { exchange->close(); }
  }

  /// Points the ring's landings at the GPU space, the host space nearest the GPU and the first
  /// disk space.
  void attach_ring(Context::RingOptions const& opts)
  {
    if (!exchange) {
      throw sirius::invalid_input_exception(
        "Context: the ring exchange needs exactly one GPU memory space");
    }
    auto& gpu = exchange->space();
    if (gpu.get_device_id() != ring->device()) {
      throw sirius::invalid_input_exception(
        "Context: the ring exchange is on device {} but the GPU memory space on device {}",
        ring->device(),
        gpu.get_device_id());
    }
    auto& memory  = context->get_memory_manager();
    int host_numa = -1;
    if (cudaDeviceGetAttribute(&host_numa, cudaDevAttrHostNumaId, ring->device()) != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      host_numa = -1;
    }
    sirius::exec::ring_exchange::landing_tiers tiers;
    for (auto const* host : memory.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST)) {
      if (tiers.host == nullptr || host->get_device_id() == host_numa) {
        tiers.host = const_cast<cucascade::memory::memory_space*>(host);
      }
    }
    if (auto const disks = memory.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK);
        !disks.empty()) {
      tiers.disk = const_cast<cucascade::memory::memory_space*>(disks.front());
    }
    tiers.gpu_bytes  = static_cast<std::size_t>(opts.gpu_landing_fraction *
                                               static_cast<double>(gpu.get_max_memory()));
    tiers.host_bytes = tiers.host ? opts.host_landing_bytes : 0;
    ring->attach(exchange, tiers);
  }

  sirius::exec::exchange_inbox& require_exchange() const
  {
    if (!exchange) {
      throw sirius::invalid_input_exception(
        "Fragment: exchange inbox requires a single GPU memory space");
    }
    return *exchange;
  }

  void bring_up(const sirius::parsed_sirius_config& config,
                std::optional<Context::RingOptions> ring_options = std::nullopt)
  {
    install_log_sink_from_env();
    if (ring_options) {
      // Before the engine's pool, so the rings are not carved out of its budget.
      int device = 0;
      if (cudaGetDevice(&device) != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        throw sirius::internal_exception("Context: no current CUDA device for the ring exchange");
      }
      sirius::exec::ring_exchange::options opts;
      opts.frame_bytes = ring_options->frame_bytes;
      opts.send_bytes  = ring_options->send_bytes;
      opts.recv_bytes  = ring_options->recv_bytes;
      ring             = std::make_shared<sirius::exec::ring_exchange>(device, opts);
    }
    context = duckdb::make_shared_ptr<duckdb::SiriusContext>();
    context->initialize(config);
    auto& memory    = context->get_memory_manager();
    auto const gpus = memory.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    if (gpus.size() == 1) {
      auto& gpu =
        *memory.get_memory_space(cucascade::memory::Tier::GPU, gpus.front()->get_device_id());
      exchange = std::make_shared<sirius::exec::exchange_inbox>(gpu);
      // Received batches can wait a long time for their receivers. Let them spill to host under
      // pressure rather than fail the query when a shuffle outgrows the pool.
      exchange->enable_spill(
        context->get_exchange_staging(), [ctx = context.get(), space = &gpu](std::size_t bytes) {
          // The spill stops once this reservation is made, and keeps what it
          // freed for it.
          std::mutex mutex;
          std::unique_ptr<cucascade::memory::reservation> made;
          auto const reserve = [&] {
            std::lock_guard const lock{mutex};
            if (!made) { made = space->make_reservation_or_null(bytes); }
            return made != nullptr;
          };
          ctx->get_downgrade_executor(space->get_id()).request_free_memory_and_wait(bytes, reserve);
          reserve();
          return std::move(made);
        });
    }
    if (ring) { attach_ring(*ring_options); }

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
    // no-fallback FFI path cannot safely execute.
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
    duckdb::Settings::Set<duckdb::EnableOptimizerSetting>(
      client, duckdb::SetScope::LOCAL, duckdb::Value::BOOLEAN(true));
    auto& disabled = duckdb::DBConfig::GetConfig(client).options.disabled_optimizers;
    disabled.insert(duckdb::OptimizerType::IN_CLAUSE);
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
    disabled.insert(sirius::planner::unsupported_optimizers.begin(),
                    sirius::planner::unsupported_optimizers.end());
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
  sirius::parsed_sirius_config config;
  impl_->bring_up(config);
}

Context::Context(const std::string& config_path) : impl_(std::make_unique<Impl>())
{
  auto config = sirius::parsed_sirius_config::from_file(config_path);
  impl_->bring_up(config);
}

Context::Context(const std::string& config_path, RingOptions const& ring)
  : impl_(std::make_unique<Impl>())
{
  auto config = config_path.empty() ? sirius::parsed_sirius_config{}
                                    : sirius::parsed_sirius_config::from_file(config_path);
  impl_->bring_up(config, ring);
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

  sirius::exec::export_arrow_result(
    std::move(result), *reinterpret_cast<ArrowArrayStream*>(out_stream_addr), kArrowBatchSize);
}

std::unique_ptr<ExchangeInbox> Context::exchange_inbox() const
{
  if (!impl_->exchange) { return nullptr; }
  // Not make_unique: the constructor is private to Context.
  return std::unique_ptr<ExchangeInbox>(new ExchangeInbox(impl_->exchange));
}

std::unique_ptr<RingExchange> Context::ring_exchange() const
{
  if (!impl_->ring) { return nullptr; }
  return std::unique_ptr<RingExchange>(new RingExchange(impl_->ring));
}

RingExchange::RingExchange(std::shared_ptr<sirius::exec::ring_exchange> ring)
  : ring_(std::move(ring))
{
}

int RingExchange::device() const noexcept { return ring_->device(); }
std::uintptr_t RingExchange::send_base() const noexcept { return ring_->send_ring().base(); }
std::uint64_t RingExchange::send_len() const noexcept { return ring_->send_ring().bytes(); }
std::uintptr_t RingExchange::recv_base() const noexcept { return ring_->recv_ring().base(); }
std::uint64_t RingExchange::recv_len() const noexcept { return ring_->recv_ring().bytes(); }
std::uint64_t RingExchange::frame_bytes() const noexcept { return ring_->frame_bytes(); }

std::int64_t RingExchange::lease_send() const
{
  auto const frame = ring_->send_ring().lease();
  return frame ? static_cast<std::int64_t>(*frame) : -1;
}

std::int64_t RingExchange::lease_recv() const
{
  auto const frame = ring_->recv_ring().lease();
  return frame ? static_cast<std::int64_t>(*frame) : -1;
}

void RingExchange::release_send(std::uint32_t frame) const { ring_->send_ring().release(frame); }

void RingExchange::release_recv(std::uint32_t frame) const { ring_->recv_ring().release(frame); }

std::uint64_t RingExchange::stage(std::uint64_t token, std::uint32_t k, std::uint32_t frame) const
{
  return ring_->stage(token, k, frame);
}

void RingExchange::release_job(std::uint64_t token) const { ring_->release_job(token); }

std::uint64_t RingExchange::begin(std::uintptr_t header_addr,
                                  std::size_t header_len,
                                  std::uint8_t& tier,
                                  std::uint64_t& rows,
                                  std::uint64_t& stream_bytes,
                                  std::uint32_t& frames) const
{
  auto const landing =
    ring_->begin({reinterpret_cast<std::uint8_t const*>(header_addr), header_len});
  tier         = landing.tier == cucascade::memory::Tier::GPU    ? 0
                 : landing.tier == cucascade::memory::Tier::HOST ? 1
                                                                 : 2;
  rows         = landing.rows;
  stream_bytes = landing.stream_bytes;
  frames       = landing.frames;
  return landing.id;
}

std::uint64_t RingExchange::copy_out(std::uint64_t landing,
                                     std::uint32_t k,
                                     std::uint32_t frame,
                                     std::uint64_t len) const
{
  return ring_->copy_out(landing, k, frame, len);
}

std::uint64_t RingExchange::seal(std::uint64_t landing) const { return ring_->seal(landing); }

void RingExchange::abort(std::uint64_t landing) const { ring_->abort(landing); }

bool RingExchange::done(std::uint64_t ticket) const { return ring_->done(ticket); }

std::unique_ptr<std::string> RingExchange::stats_line() const
{
  return std::make_unique<std::string>(ring_->stats_line());
}

ExchangeInbox::ExchangeInbox(std::shared_ptr<sirius::exec::exchange_inbox> exchange)
  : exchange_(std::move(exchange))
{
}

int ExchangeInbox::device() const noexcept { return exchange_->device(); }

void ExchangeInbox::release(std::uint64_t token) const { exchange_->release(token); }

std::size_t ExchangeInbox::outstanding() const { return exchange_->outstanding(); }

OutputDrain::OutputDrain(std::shared_ptr<sirius::exec::batch_stream> stream,
                         std::shared_ptr<sirius::exec::exchange_inbox> exchange,
                         std::shared_ptr<sirius::exec::ring_exchange> ring)
  : stream_(std::move(stream)), exchange_(std::move(exchange)), ring_(std::move(ring))
{
}

std::unique_ptr<std::vector<std::uint8_t>> OutputDrain::export_ring_next(
  std::uint32_t timeout_ms,
  bool& ended,
  std::uint64_t& token,
  std::uint8_t& kind,
  std::uint64_t& rows,
  std::uint64_t& stream_bytes) const
{
  if (!ring_) {
    throw sirius::invalid_input_exception("OutputDrain: this context has no ring exchange");
  }
  using availability  = sirius::exec::batch_stream::availability;
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  ended               = false;
  for (;;) {
    // A batch whose export ran out of GPU memory last time goes first, whatever the stream says.
    std::shared_ptr<cucascade::data_batch> batch = std::move(ring_retry_);
    ring_retry_.reset();
    if (!batch) {
      switch (stream_->classify()) {
        case availability::END_OF_STREAM: ended = true; return nullptr;
        case availability::HAS_DATA:
          // Rethrows the fragment's error; nullptr only if another consumer took the batch.
          batch = stream_->try_pull();
          break;
        case availability::WAITING: {
          auto const left = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
          if (left.count() <= 0) { return nullptr; }
          static_cast<void>(stream_->wait_for(left));
          continue;
        }
      }
      if (!batch) { continue; }
    }
    std::optional<sirius::exec::ring_exchange::send_job> job;
    try {
      job = ring_->export_batch(batch);
    } catch (rmm::out_of_memory const&) {
      // Kept for the next call: the caller may wait and ask again.
      ring_retry_ = std::move(batch);
      throw;
    }
    if (job) {
      token        = job->token;
      kind         = static_cast<std::uint8_t>(job->kind);
      rows         = job->rows;
      stream_bytes = job->stream_bytes;
      return std::make_unique<std::vector<std::uint8_t>>(std::move(job->header));
    }
    // A batch without rows is not sent.
  }
}

std::unique_ptr<Context> make_context() { return std::make_unique<Context>(); }

std::unique_ptr<Context> make_context_from_config(const std::string& config_path)
{
  return std::make_unique<Context>(config_path);
}

std::unique_ptr<Context> make_context_with_ring(const std::string& config_path,
                                                std::uint64_t frame_bytes,
                                                std::uint64_t send_bytes,
                                                std::uint64_t recv_bytes,
                                                double gpu_landing_fraction,
                                                std::uint64_t host_landing_bytes)
{
  return std::make_unique<Context>(
    config_path,
    Context::RingOptions{
      frame_bytes, send_bytes, recv_bytes, gpu_landing_fraction, host_landing_bytes});
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
    std::optional<std::uint64_t> estimated_rows;
  };

  std::map<sirius::exec::stream_id_t, declared_input> inputs;
  std::vector<sirius::exec::stream_id_t> outputs;
  bool broadcast_outputs{false};
  std::vector<int> hash_key_columns;

  // One streaming_fragment for both terminals. Empty outputs is a RESULT_COLLECTOR.
  std::unique_ptr<sirius::exec::streaming_fragment> fragment;

  // Per output stream, a batch pulled for export_ring() whose export ran out of GPU memory
  // (bringing a spilled batch back): it is exported first on the next call.
  std::map<std::uint64_t, std::shared_ptr<cucascade::data_batch>> ring_pending;

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
      spec.estimated_rows = declared.estimated_rows;
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

void Fragment::declare_input_cardinality(std::uint64_t stream_id, std::uint64_t rows)
{
  impl_->require_not_built("declare_input_cardinality");
  impl_->inputs[stream_id].estimated_rows = rows;
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

std::unique_ptr<std::vector<std::uint8_t>> Fragment::export_ring(std::uint64_t stream_id,
                                                                 std::uint64_t& token,
                                                                 std::uint8_t& kind,
                                                                 std::uint64_t& rows,
                                                                 std::uint64_t& stream_bytes)
{
  impl_->require_built("export_ring()");
  if (!impl_->ctx.ring) {
    throw sirius::invalid_input_exception("Fragment: this context has no ring exchange");
  }
  for (;;) {
    std::shared_ptr<cucascade::data_batch> batch;
    if (auto pending = impl_->ring_pending.find(stream_id); pending != impl_->ring_pending.end()) {
      batch = std::move(pending->second);
      impl_->ring_pending.erase(pending);
    } else if (auto pulled = impl_->fragment->pull(stream_id)) {
      batch = std::move(*pulled);
    } else {
      return nullptr;
    }
    std::optional<sirius::exec::ring_exchange::send_job> job;
    try {
      job = impl_->ctx.ring->export_batch(batch);
    } catch (rmm::out_of_memory const&) {
      // Kept for the next call: the caller may wait and ask again.
      impl_->ring_pending[stream_id] = std::move(batch);
      throw;
    }
    if (job) {
      token        = job->token;
      kind         = static_cast<std::uint8_t>(job->kind);
      rows         = job->rows;
      stream_bytes = job->stream_bytes;
      return std::make_unique<std::vector<std::uint8_t>>(std::move(job->header));
    }
  }
}

std::unique_ptr<OutputDrain> Fragment::output_drain(std::uint64_t stream_id) const
{
  impl_->require_built("output_drain()");
  if (!impl_->ctx.ring) {
    throw sirius::invalid_input_exception("Fragment: output_drain() needs a ring exchange");
  }
  return std::make_unique<OutputDrain>(
    impl_->fragment->output_stream(stream_id), impl_->ctx.exchange, impl_->ctx.ring);
}

void Fragment::push_received(std::uint64_t stream_id, std::uint64_t token)
{
  impl_->require_built("push_received()");
  auto& exchange = impl_->ctx.require_exchange();
  // A batch dropped by a refused push is freed on the device its buffers live on.
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{exchange.device()}};
  // Refuses an undeclared input or a running fragment before the token is consumed.
  impl_->fragment->require_pushable(stream_id);
  // push() checks the columns of a batch on the GPU. A batch that spilled to host goes in
  // unchecked, and nothing checks it when a task brings it back: its columns are the ones its
  // transport landed, and the spill keeps them.
  auto batch = exchange.take_batch(token);
  if (!impl_->fragment->push(stream_id, std::move(batch))) {
    throw sirius::invalid_input_exception("Fragment: input stream " + std::to_string(stream_id) +
                                          " refused a received batch; it had already ended");
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
  auto result = impl_->fragment->take_result();
  sirius::exec::export_arrow_result(
    std::move(result), *reinterpret_cast<ArrowArrayStream*>(out_stream_addr), kArrowBatchSize);
}

std::size_t Fragment::output_batch_count(std::uint64_t stream_id) const
{
  if (!impl_->fragment) { return 0; }
  return impl_->fragment->output_batch_count(stream_id);
}

std::uint64_t Fragment::output_row_count(std::uint64_t stream_id) const
{
  impl_->require_built("output_row_count()");
  return impl_->fragment->output_row_count(stream_id);
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
