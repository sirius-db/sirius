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
#include "exec/stream_bind_catalog.hpp"       // sirius::exec::stream_bind_catalog
#include "exec/stream_plan_bindings.hpp"      // sirius::exec::register_stream_source_function
#include "exec/streaming_fragment.hpp"        // sirius::exec::streaming_fragment, fragment_spec
#include "from_substrait.hpp"                 // duckdb::SubstraitToDuckDB (compiled into libsirius)
#include "helper/type_conversions.hpp"        // sirius::from_duckdb
#include "io/io_errors.hpp"                   // sirius::io::credential_error
#include "io/rest/s3/endpoint.hpp"            // sirius::io::rest::s3::parse_endpoint
#include "log/logging.hpp"                    // SIRIUS_LOG_INFO
#include "parquet_extension.hpp"              // duckdb::ParquetExtension
#include "planner/substrait_scan_ranges.hpp"  // sirius::planner::scan_byte_ranges_state
#include "planner/unsupported_optimizers.hpp"
#include "scan_manager/config.hpp"  // sirius::scan_manager::scan_manager_config
#include "sirius/ffi.hpp"
#include "sirius_config.hpp"   // sirius::sirius_config
#include "sirius_context.hpp"  // duckdb::SiriusContext
#include "sirius_ffi_testing.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>

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

// --- Test-only capture pause seam --------------------------------------------------------------

// The installed callable is held through an owning handle so that the mutex only ever guards
// a pointer copy/swap: constructing, copying, invoking and destroying the callable (all of which
// may run user code that installs or clears hooks) happen outside the lock.
std::mutex capture_pause_mutex;
std::shared_ptr<const std::function<void()>> capture_pause_hook;

// Runs the installed hook (if any) outside the hook mutex, so the hook may itself install or
// clear hooks or touch the Context's thread-safe members.
void run_capture_pause_hook()
{
  std::shared_ptr<const std::function<void()>> hook;
  {
    std::lock_guard lock(capture_pause_mutex);
    hook = capture_pause_hook;
  }
  if (hook) { (*hook)(); }
}

// --- ObjectStoreConfig checks ------------------------------------------------------------------
//
// Run at set_object_store() and at the object-store factory without object-store network
// access or backend-resource creation. CA readability may access the local filesystem.
// Errors name the field and a fixed reason, never its value. Authentication, TLS and
// network failures are reported when a snapshot is used.

constexpr std::array<std::string_view, 3> kTransports{"auto", "http", "rdma"};
constexpr std::array<std::string_view, 2> kSigningModes{"presigned", "header"};

[[noreturn]] void invalid_field(const char* field, const char* reason)
{
  throw std::invalid_argument(std::string("ObjectStoreConfig.") + field + ": " + reason);
}

bool one_of(std::string_view value, auto const& allowed)
{
  return std::find(allowed.begin(), allowed.end(), value) != allowed.end();
}

// Shape of an S3 service endpoint: value-free checks first, then the SigV4 authorizer's own
// parser, so the two can never disagree on what is accepted.
void check_endpoint(const std::string& endpoint)
{
  if (endpoint.empty()) { invalid_field("endpoint", "required"); }
  auto const sep = endpoint.find("://");
  if (sep == std::string::npos) {
    invalid_field("endpoint", "missing scheme (expected http:// or https://)");
  }
  std::string scheme = endpoint.substr(0, sep);
  for (auto& c : scheme) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (scheme != "http" && scheme != "https") {
    invalid_field("endpoint", "scheme must be http or https");
  }
  auto const remainder = std::string_view{endpoint}.substr(sep + 3);
  // A port-only authority ("https://:443") has no host either.
  if (remainder.empty() || remainder.front() == ':') { invalid_field("endpoint", "missing host"); }
  if (remainder.find_first_of("/?#") != std::string_view::npos) {
    invalid_field("endpoint", "must be scheme://host[:port] with no path, query or fragment");
  }
  if (remainder.find('@') != std::string_view::npos) {
    invalid_field("endpoint", "must not contain user-info");
  }
  try {
    (void)sirius::io::rest::s3::parse_endpoint(endpoint);
  } catch (sirius::io::credential_error const&) {
    invalid_field("endpoint", "malformed");
  }
}

void check_intrinsic(const ObjectStoreConfig& cfg)
{
  check_endpoint(cfg.endpoint);
  if (cfg.region.empty()) { invalid_field("region", "required"); }
  if (cfg.access_key.empty()) { invalid_field("access_key", "required"); }
  if (cfg.secret_key.empty()) { invalid_field("secret_key", "required"); }
  // The public value sets are exact: the engine's parser also knows an internal "https" alias
  // for the HTTP transport, which is deliberately not accepted here.
  if (!one_of(cfg.transport, kTransports)) {
    invalid_field("transport", "must be one of auto, http, rdma");
  }
  if (!one_of(cfg.signing_mode, kSigningModes)) {
    invalid_field("signing_mode", "must be one of presigned, header");
  }
  if (!cfg.ca_bundle_path.empty()) {
    std::ifstream bundle(cfg.ca_bundle_path);
    if (!bundle.good()) { invalid_field("ca_bundle_path", "file is not readable"); }
  }
  if (!cfg.uri_prefix.empty()) {
    // "s3://bucket/" or "s3://bucket/prefix/": scheme s3, a bucket, a trailing slash, no query.
    constexpr std::string_view kScheme = "s3://";
    std::string_view prefix{cfg.uri_prefix};
    if (prefix.substr(0, kScheme.size()) != kScheme || prefix.size() == kScheme.size() ||
        prefix[kScheme.size()] == '/') {
      invalid_field("uri_prefix", "must be s3://bucket/ or s3://bucket/prefix/");
    }
    if (prefix.back() != '/') { invalid_field("uri_prefix", "must end with '/'"); }
    if (prefix.find_first_of("?#") != std::string_view::npos) {
      invalid_field("uri_prefix", "must not contain a query or fragment");
    }
  }
}

// Scheme-only canonicalization for prefix matching: bucket and key bytes compare exactly.
std::string lowercase_scheme(std::string_view uri)
{
  std::string out{uri};
  auto const sep = out.find("://");
  if (sep == std::string::npos) { return out; }
  for (std::size_t i = 0; i < sep; ++i) {
    out[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(out[i])));
  }
  return out;
}

// With a namespace in force, an object key is matched byte-for-byte against the prefix, so a key
// the transport or store could normalize to a different object (empty, `.` or `..` segments) or
// decode (`%xx`) is refused rather than normalized: libcurl squashes dot segments after this
// check, and S3 fronts differ on how they alias them.
bool key_defeats_prefix_match(std::string_view path)
{
  auto const sep = path.find("://");
  auto const bucket_end =
    sep == std::string_view::npos ? std::string_view::npos : path.find('/', sep + 3);
  if (bucket_end == std::string_view::npos) { return false; }
  auto key = path.substr(bucket_end + 1);
  if (key.find('%') != std::string_view::npos) { return true; }
  while (!key.empty()) {
    auto const slash   = key.find('/');
    auto const segment = key.substr(0, slash);
    if (segment.empty() || segment == "." || segment == "..") { return true; }
    if (slash == std::string_view::npos) { break; }
    key.remove_prefix(slash + 1);
  }
  return false;
}

// Identify an offending plan item in an error without echoing authentication material a URI can
// carry: user-info, query (e.g. presigned X-Amz-* parameters) and fragment are dropped.
std::string redacted_uri(std::string_view uri)
{
  auto const cut = uri.find_first_of("?#");
  std::string out{uri.substr(0, cut)};
  auto const scheme_end = out.find("://");
  if (scheme_end != std::string::npos) {
    auto const authority_begin = scheme_end + 3;
    auto const authority_end   = out.find('/', authority_begin);
    auto const authority       = std::string_view{out}.substr(
      authority_begin,
      (authority_end == std::string::npos ? out.size() : authority_end) - authority_begin);
    auto const at = authority.rfind('@');
    if (at != std::string_view::npos) { out.erase(authority_begin, at + 1); }
  }
  return out;
}

// A plan that reads from an object store is refused here, before DuckDB's consumer can bind the
// read: without a snapshot there is nothing to serve it with; outside the snapshot's namespace
// the read is out of scope; and until the native object-store route lands, a configured context
// refuses too rather than letting the read reach a legacy path/default route. The offending item
// is named in redacted form; no secret lookup and no network access happen. Each message starts
// with a stable reason token.
void refuse_remote_reads(const std::vector<std::string>& remote_paths,
                         const std::shared_ptr<ObjectStoreConfig>& snapshot)
{
  if (remote_paths.empty()) { return; }
  if (!snapshot) {
    throw sirius::invalid_input_exception(
      "object_store_unconfigured: the plan reads '{}' from an object store, but this Context "
      "has no object-store snapshot (Context::set_object_store)",
      redacted_uri(remote_paths.front()));
  }
  if (!snapshot->uri_prefix.empty()) {
    for (const auto& path : remote_paths) {
      if (!lowercase_scheme(path).starts_with(snapshot->uri_prefix) ||
          key_defeats_prefix_match(path)) {
        throw sirius::invalid_input_exception(
          "object_store_scope: the plan reads '{}', outside the snapshot's uri_prefix (or with a "
          "key segment that cannot be matched against it)",
          redacted_uri(path));
      }
    }
  }
  throw sirius::invalid_input_exception(
    "object_store_route_unavailable: the plan reads '{}' from an object store; native "
    "object-store reads are not yet served on the FFI path",
    redacted_uri(remote_paths.front()));
}

// Options the configured IO backend cannot honor are refused rather than ignored. At this tree
// the registered backends are the Sirius stack (uring + REST) and kvikIO: no S3-over-RDMA
// backend exists, and kvikIO's remote handle consumes endpoint/region/credentials only.
void check_backend_support(const ObjectStoreConfig& cfg,
                           const sirius::scan_manager::scan_manager_config& scan_manager)
{
  if (cfg.transport == "rdma") {
    invalid_field("transport", "no S3-over-RDMA backend is registered in this build");
  }
  if (scan_manager.backend == sirius::scan_manager::io_backend::kvikio) {
    if (cfg.signing_mode != "presigned") {
      invalid_field("signing_mode", "the configured kvikio backend does not honor a signing mode");
    }
    if (!cfg.ca_bundle_path.empty()) {
      invalid_field("ca_bundle_path", "the configured kvikio backend does not honor a CA bundle");
    }
  }
}

void check_object_store(const ObjectStoreConfig& cfg,
                        const sirius::scan_manager::scan_manager_config& scan_manager)
{
  check_intrinsic(cfg);
  check_backend_support(cfg, scan_manager);
}

// Lower a Substrait plan to a bound+optimized DuckDB LogicalOperator. `snapshot` is the
// object-store snapshot the execution captured when its binding began (null = none).
sirius::exec::bound_plan lower_substrait(duckdb::Connection& conn,
                                         const std::string& substrait_plan,
                                         const std::shared_ptr<ObjectStoreConfig>& snapshot)
{
  auto& client = *conn.context;

  // Byte-ranged parquet splits ride the plan's LocalFiles items, but DuckDB's consumer and
  // parquet binding cannot carry them — extract into a per-plan state the physical plan
  // generator consumes. Always replaced (and removed when this plan has none), so a stale
  // registry can never leak a previous plan's ranges into this one. The same parse lists the
  // plan's remote reads, which are refused or scoped before anything binds them.
  client.registered_state->Remove(sirius::planner::scan_byte_ranges_state::kStateKey);
  auto inputs = sirius::planner::inspect_scan_plan_inputs(substrait_plan);
  refuse_remote_reads(inputs.remote_paths, snapshot);
  if (!inputs.byte_ranges.empty()) {
    client.registered_state->Insert(
      sirius::planner::scan_byte_ranges_state::kStateKey,
      duckdb::make_shared_ptr<sirius::planner::scan_byte_ranges_state>(
        std::move(inputs.byte_ranges)));
  }

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
  // Current configuration, atomically replaced under its own mutex. Executions already
  // capture an owning copy before binding; object-store I/O routing is not implemented here.
  mutable std::mutex object_store_mutex;
  std::shared_ptr<ObjectStoreConfig> object_store;

  // The snapshot an execution captures when its binding begins: one owning copy per
  // execution, unaffected by later set_object_store() calls.
  std::shared_ptr<ObjectStoreConfig> capture_object_store() const
  {
    std::lock_guard lock(object_store_mutex);
    return object_store;
  }

  void bring_up(const sirius::parsed_sirius_config& config)
  {
    install_log_sink_from_env();
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

Context::Context(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

std::shared_ptr<ObjectStoreConfig> Context::object_store_config() const
{
  std::lock_guard lock(impl_->object_store_mutex);
  return impl_->object_store;
}

void Context::set_object_store(std::shared_ptr<ObjectStoreConfig> cfg) const
{
  if (cfg) { check_object_store(*cfg, impl_->context->get_config().get_scan_manager_config()); }
  bool const published = cfg != nullptr;
  {
    std::lock_guard lock(impl_->object_store_mutex);
    // swap, not assign: the retired snapshot (now in `cfg`) is released after the lock, so a
    // caller-supplied deleter can never run under the mutex the getter takes.
    impl_->object_store.swap(cfg);
  }
  SIRIUS_LOG_INFO("[sirius_ffi] object store snapshot {}", published ? "published" : "cleared");
}

std::shared_ptr<ObjectStoreConfig> ObjectStoreConfig::with_credentials(
  const std::string& new_access_key,
  const std::string& new_secret_key,
  const std::string& new_session_token) const
{
  return std::make_shared<ObjectStoreConfig>(ObjectStoreConfig{.endpoint       = endpoint,
                                                               .region         = region,
                                                               .access_key     = new_access_key,
                                                               .secret_key     = new_secret_key,
                                                               .session_token  = new_session_token,
                                                               .transport      = transport,
                                                               .signing_mode   = signing_mode,
                                                               .ca_bundle_path = ca_bundle_path,
                                                               .uri_prefix     = uri_prefix});
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
  auto const snapshot = impl_->capture_object_store();
  run_capture_pause_hook();
  in_transaction(impl_->conn_mutex, *impl_->conn, [&] {
    // The same result path as a zero-output Fragment.
    sirius::exec::fragment_spec spec;
    spec.plan_source = [&](duckdb::ClientContext&) {
      auto const lower_started = std::chrono::steady_clock::now();
      auto bound               = lower_substrait(*impl_->conn, plan, snapshot);
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

std::unique_ptr<Context> make_context() { return std::make_unique<Context>(); }

std::unique_ptr<Context> make_context_from_config(const std::string& config_path)
{
  return std::make_unique<Context>(config_path);
}

std::unique_ptr<Context> make_context_from_config(const std::string& config_path,
                                                  std::shared_ptr<ObjectStoreConfig> cfg)
{
  // The snapshot is checked against the parsed configuration before bring_up(), so a rejected
  // snapshot never creates engine state.
  auto config = config_path.empty() ? sirius::parsed_sirius_config{}
                                    : sirius::parsed_sirius_config::from_file(config_path);
  if (cfg) { check_object_store(*cfg, config.get_scan_manager_config()); }
  auto impl = std::make_unique<Context::Impl>();
  impl->bring_up(config);
  impl->object_store = std::move(cfg);
  return std::unique_ptr<Context>(new Context(std::move(impl)));
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
  // Object-store snapshot captured at build() for this execution; null = none.
  std::shared_ptr<ObjectStoreConfig> object_store;

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
  impl_->object_store = impl_->ctx.capture_object_store();
  run_capture_pause_hook();
  // Type-name parsing, CREATE VIEW, and Substrait lowering all need an active transaction. A
  // failure rolls back the views, so a half-declared fragment leaves nothing behind.
  in_transaction(impl_->ctx.conn_mutex, *impl_->ctx.conn, [&] {
    sirius::exec::fragment_spec spec;
    spec.inputs      = impl_->resolve_inputs();
    spec.outputs     = impl_->outputs;
    spec.plan_source = [this, &substrait_plan](duckdb::ClientContext&) {
      impl_->create_stream_views();
      return lower_substrait(*impl_->ctx.conn, substrait_plan, impl_->object_store);
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

namespace testing {

void set_capture_pause_hook(std::function<void()> hook)
{
  std::shared_ptr<const std::function<void()>> replacement;
  if (hook) { replacement = std::make_shared<const std::function<void()>>(std::move(hook)); }
  {
    std::lock_guard lock(capture_pause_mutex);
    capture_pause_hook.swap(replacement);
  }
  // `replacement` now owns the retired callable and releases it here, after the lock.
}

}  // namespace testing

}  // namespace sirius::ffi
