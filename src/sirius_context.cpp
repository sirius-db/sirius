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

#include "sirius_context.hpp"

#include "config.hpp"
#include "cucascade/memory/memory_reservation_manager.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/planner.hpp"
#include "log/duckdb_sink.hpp"
#include "log/logging.hpp"
#include "log/noop_sink.hpp"
#include "log/spdlog_owning_sink.hpp"
#include "memory/numa_small_pinned_mr.hpp"
#include "memory/resource_ref_utils.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "memory/topology_index.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_sql_rewrite.hpp"
#include "telemetry/batch_telemetry.hpp"
#include "transparent/physical_sirius_execution.hpp"
#include "transparent/sirius_optimizer_extension.hpp"
#include "util/duckdb_error_message.hpp"

#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>
#include <duckdb/common/allocator.hpp>
#include <duckdb/execution/operator/persistent/physical_insert.hpp>
#include <duckdb/execution/operator/persistent/physical_merge_into.hpp>
#include <duckdb/execution/operator/persistent/physical_update.hpp>
#include <duckdb/execution/physical_plan_generator.hpp>
#include <io/types.hpp>
#include <io/uring/uring_ioctx.hpp>
#include <sys/resource.h>
#include <unistd.h>  // for isatty/fileno

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdio>   // for fprintf/fileno (fallback banner)
#include <cstdlib>  // for std::getenv
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>

namespace duckdb {

namespace {

static constexpr std::string_view CONFIG_FILE_NAME        = "sirius.yaml";
static constexpr std::string_view LEGACY_CONFIG_FILE_NAME = "sirius.cfg";
static constexpr std::string_view CONFIG_FILE_DIR         = ".sirius";
static constexpr std::string_view CONFIG_FILE_ENV_NAME    = "SIRIUS_CONFIG_FILE";

optional_ptr<TableCatalogEntry const> find_update_target(PhysicalOperator const& op)
{
  switch (op.type) {
    case PhysicalOperatorType::UPDATE: return &op.Cast<PhysicalUpdate>().tableref;
    case PhysicalOperatorType::INSERT: {
      auto const& insert = op.Cast<PhysicalInsert>();
      if (insert.action_type == OnConflictAction::UPDATE && insert.insert_table) {
        return &*insert.insert_table;
      }
      break;
    }
    case PhysicalOperatorType::MERGE_INTO: {
      auto const& merge = op.Cast<PhysicalMergeInto>();
      for (auto const& action : merge.actions) {
        if (action->action_type == MergeActionType::MERGE_UPDATE && action->op) {
          return find_update_target(*action->op);
        }
      }
      break;
    }
    default: break;
  }
  for (auto const& child : op.GetChildren()) {
    if (auto target = find_update_target(child.get())) { return target; }
  }
  return nullptr;
}

std::optional<std::string> pinned_name_for_table(
  sirius::scan_manager::sirius_scan_manager const& scan_manager, TableCatalogEntry const& table)
{
  std::optional<std::string> result;
  scan_manager.visit_pinned_entries([&](std::string_view name, auto const& entry) {
    if (!entry.cache_info.matches_duckdb_table(
          table.ParentCatalog().GetName(), table.ParentSchema().name, table.name)) {
      return true;
    }
    result = name;
    return false;
  });
  return result;
}

void reject_update_to_pinned_table(SiriusContext& sirius_context,
                                   ClientContext& context,
                                   PreparedStatementData& prepared)
{
  switch (prepared.statement_type) {
    case StatementType::UPDATE_STATEMENT:
    case StatementType::INSERT_STATEMENT:
    case StatementType::MERGE_INTO_STATEMENT: break;
    default: return;
  }
  if (!prepared.physical_plan) { return; }
  auto target = find_update_target(prepared.physical_plan->Root());
  if (!target) { return; }

  auto connection_state = get_sirius_connection_state(context);
  if (!connection_state) {
    throw InternalException("Sirius connection state is unavailable while guarding UPDATE");
  }

  std::shared_lock<std::shared_mutex> update_guard;
  if (!connection_state->has_pinned_update_guard()) {
    update_guard = sirius_context.lock_pinned_table_updates();
  }
  {
    SiriusContext::SlotGuard pin_registry_guard(sirius_context, context);
    auto pinned_name = pinned_name_for_table(sirius_context.get_scan_manager(), *target);
    if (pinned_name) {
      throw InvalidInputException(
        "Sirius does not support UPDATE on pinned DuckDB table '%s'. Run CALL "
        "unpin_table('%s') before updating it",
        target->name,
        *pinned_name);
    }
  }

  if (update_guard.owns_lock()) {
    connection_state->set_pinned_update_guard(std::move(update_guard));
  }
}

/// Resolve the config file path. Search order:
///   1. SIRIUS_CONFIG_FILE environment variable (explicit path)
///   2. ./sirius.yaml in the current working directory
///   3. ~/.sirius/sirius.yaml in the user's home directory
/// Returns std::nullopt if none of the candidates exist.
std::optional<std::string> get_config_file_path()
{
  // 1. Explicit env var — return as-is (caller checks existence)
  const char* env = std::getenv(std::string(CONFIG_FILE_ENV_NAME).c_str());
  if (env != nullptr) { return std::string(env); }

  // 2. Current working directory
  auto cwd_path = std::filesystem::current_path() / std::string(CONFIG_FILE_NAME);
  if (std::filesystem::exists(cwd_path)) { return cwd_path.string(); }

  // 3. Home directory
  const char* home_dir = std::getenv("HOME");
  if (home_dir != nullptr) {
    auto home_path = std::filesystem::path(home_dir) / std::string(CONFIG_FILE_DIR) /
                     std::string(CONFIG_FILE_NAME);
    if (std::filesystem::exists(home_path)) { return home_path.string(); }
  }

  return std::nullopt;
}

/// Check whether a legacy sirius.cfg file exists in any of the search locations.
/// Returns the path if found, std::nullopt otherwise.
std::optional<std::string> find_legacy_config_file()
{
  // Current working directory
  auto cwd_path = std::filesystem::current_path() / std::string(LEGACY_CONFIG_FILE_NAME);
  if (std::filesystem::exists(cwd_path)) { return cwd_path.string(); }

  // Home directory
  const char* home_dir = std::getenv("HOME");
  if (home_dir != nullptr) {
    auto home_path = std::filesystem::path(home_dir) / std::string(CONFIG_FILE_DIR) /
                     std::string(LEGACY_CONFIG_FILE_NAME);
    if (std::filesystem::exists(home_path)) { return home_path.string(); }
  }

  return std::nullopt;
}

}  // namespace

// ================= sirius_context ================= //

SiriusContext::SiriusContext() = default;

SiriusContext::~SiriusContext() noexcept
{
  if (is_initialized_) { terminate(); }
}

// Log host and GPU memory pool stats at a labeled point.
// Lets us verify that allocated bytes return to baseline at the end of each
// query — the leak signature is "QueryEnd allocated != QueryBegin allocated".
void SiriusContext::log_pool_stats(std::string_view tag) const
{
  if (!memory_manager_) { return; }

  // Host pinned pool (fixed_size_host_memory_resource).
  for (auto const* space :
       memory_manager_->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST)) {
    auto* fs_mr =
      space->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
    if (fs_mr) {
      SIRIUS_LOG_INFO("[host_pool] HOST:{} {} allocated={} bytes peak={} bytes free_blocks={}",
                      space->get_id().device_id,
                      tag,
                      fs_mr->get_total_allocated_bytes(),
                      fs_mr->get_peak_total_allocated_bytes(),
                      fs_mr->get_free_blocks());
    }
  }

  // GPU device memory pools (reservation_aware_resource_adaptor), one line per GPU.
  for (auto const* space :
       memory_manager_->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)) {
    auto* ra_mr =
      space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    if (!ra_mr) { continue; }
    SIRIUS_LOG_INFO("[gpu_pool] GPU:{} {} allocated={} bytes peak={} bytes reserved={} bytes",
                    space->get_device_id(),
                    tag,
                    ra_mr->get_total_allocated_bytes(),
                    ra_mr->get_peak_total_allocated_bytes(),
                    ra_mr->get_total_reserved_bytes());
  }
}

void SiriusContext::QueryBegin(ClientContext& context)
{
  // Suppress logging for internal connections (e.g. internal metadata lookups).
  if (is_internal_query_active(context)) { return; }

  // Advance the per-connection query ordinal FIRST, before any best-effort
  // observation: even if the SQL capture or logging below fails, the next
  // execution window must see a FRESH ordinal — a stale one would correlate
  // that window to the PREVIOUS query's SQL line instead of standing alone as
  // a window without SQL. next_query_ordinal() is noexcept.
  shared_ptr<SiriusConnectionState> conn_state;
  uint64_t query_ordinal = 0;
  try {
    conn_state = get_sirius_connection_state(context);
  } catch (...) {  // resolution failure: fall through to keyless logging
  }
  if (conn_state) { query_ordinal = conn_state->next_query_ordinal(); }

  // QueryBegin holds no lock and mutates no shared state: slot ownership is
  // scope-bound (StandaloneQueryScope/SlotGuard) and the begin mutations run
  // inside the execution window. Everything below is observation only and
  // best-effort; a logging or allocation failure must not fail the query it
  // decorates.
  try {
    auto query = context.GetCurrentQuery();
    // Collapse every run of whitespace (incl. newlines/tabs) to a single space,
    // trim leading/trailing whitespace, and log the full normalized query so
    // log_analysis tools can correlate by SQL text without truncation.
    std::string normalized_query;
    normalized_query.reserve(query.size());
    bool in_ws = true;  // skip leading whitespace
    for (char c : query) {
      bool is_ws = std::isspace(static_cast<unsigned char>(c)) != 0;
      if (is_ws) {
        if (!in_ws) { normalized_query.push_back(' '); }
        in_ws = true;
      } else {
        normalized_query.push_back(c);
        in_ws = false;
      }
    }
    if (!normalized_query.empty() && normalized_query.back() == ' ') {
      normalized_query.pop_back();
    }
    // ONE line carrying both the correlation key and the SQL (two separate lines
    // could interleave with another connection's logging); execution windows and
    // pool lines carry the same instance/connection key. The trailing
    // "SQL: <sql>" form is preserved for log-analyzer text correlation.
    if (conn_state) {
      SIRIUS_LOG_INFO("QueryBegin: instance={} connection={} query={} SQL: {}",
                      static_cast<const void*>(this),
                      conn_state->connection_id(),
                      query_ordinal,
                      normalized_query);
    } else {
      SIRIUS_LOG_INFO("QueryBegin: SQL: {}", normalized_query);
    }
  } catch (...) {  // best-effort observability
  }
}

void SiriusContext::QueryEnd()
{
  // The DuckDB query-end callback releases nothing: slot ownership is
  // scope-bound and the mandatory cleanup runs inside the execution window
  // (StandaloneQueryScope::finish), before the result is exposed. Kept as the
  // ClientContextState interface hook.
}

void SiriusContext::QueryEnd(ClientContext& context)
{
  if (is_internal_query_active(context)) { return; }
  QueryEnd();
}

void SiriusContext::QueryEnd(ClientContext& context, optional_ptr<ErrorData> error)
{
  QueryEnd(context);
}

bool SiriusContext::is_internal_query_active(ClientContext& context) noexcept
{
  auto conn_state = get_sirius_connection_state(context);
  return conn_state && conn_state->is_internal_query_active();
}

void SiriusContext::throw_runtime_unavailable() const
{
  // IS-A ExecutorException: deliberately non-invalidating (INTERNAL/FATAL
  // would invalidate the whole DatabaseInstance and defeat "CPU queries
  // continue"). Typed so entry points can classify it without text matching.
  throw SiriusRuntimeUnavailableException(
    "Sirius GPU runtime is unavailable after a mandatory cleanup failure; "
    "CPU execution continues. Restart the process to restore GPU execution.");
}

void SiriusContext::begin_execution_window(ClientContext& context,
                                           sirius::query_id_t query_id,
                                           std::string_view window_label,
                                           std::string_view pool_tag)
{
  // Runs inside the held slot, after acquire and the health check and before
  // the final create_plan.
  // Logging around the mutations is best-effort: a logging failure must never
  // leave the runtime half-begun (the mutations themselves are the only
  // throwing steps that matter; a throw here is handled by the scope ctor's
  // backstop-then-release path).
  try {
    log_pool_stats(pool_tag);
    SIRIUS_LOG_INFO("QueryBegin: {}", window_label);
  } catch (...) {  // best-effort observability
  }
  // Register this query's repository manager up front
  data_repository_registry_.create_for_query(query_id);
  task_creator_->reset();
  task_creator_->set_client_context(context);
  // GPU admission runs later, in sirius_engine::initialize_internal().
}

void SiriusContext::run_mandatory_cleanup(sirius::query_id_t query_id, std::string_view end_tag)
{
  // Observability inside the cleanup is best-effort: only the mandatory steps
  // (query/drain/repository/scan/task resets) may throw out of this function
  // and thereby poison the runtime — a logging or telemetry failure must not.
  try {
    SIRIUS_LOG_INFO("QueryEnd");
  } catch (...) {
  }

  query_.reset();

  // Drain all downgrade executors before clearing repositories — ensures no downgrade
  // tasks hold shared_ptr<data_batch> references to batches we're about to destroy.
  for (auto& executor : downgrade_executors_) {
    executor->drain();
  }

  // Close out batch placements still alive (un-consumed repo contents,
  // result-collector outputs) before their repositories are cleared.
  // Best-effort: telemetry failure must not abort the remaining mandatory
  // steps or poison the runtime.
  try {
    sirius::telemetry::batch_telemetry_registry::instance().on_query_end();
  } catch (std::exception& e) {
    try {
      SIRIUS_LOG_WARN("batch telemetry on_query_end failed (ignored): {}", e.what());
    } catch (...) {
    }
  } catch (...) {
  }

  // Drop THIS query's data repositories, leaving any other in-flight query's untouched.
  // Any batches still present are leaked — operators should have popped everything.
  // Safe to clear here because the downgrade executors were drained above, so nothing still
  // holds a raw data_repository* borrowed from this query's manager.
  {
    auto leaked = data_repository_registry_.erase(query_id);
    try {
      for (auto const& info : leaked) {
        SIRIUS_LOG_WARN(
          "SiriusContext::run_mandatory_cleanup: query {} operator {} port '{}' still had {} "
          "un-consumed data batch(es) (memory leak).",
          query_id,
          info.operator_id,
          info.port_id,
          info.count);
      }
    } catch (...) {  // best-effort observability
    }
  }

  // Drop scan-manager providers for this query. Repositories are already
  // cleared above, so downstream data_batches that referenced sliced
  // host_data_representation are gone before the providers go away.
  if (scan_manager_) { scan_manager_->reset(); }

  // Drop per-query global states held by task_creator. These include
  // duckdb_scan_task_global_state, which transitively owns a
  // duckdb::DuckTableScanState referencing BufferManager-owned BlockHandles.
  // If we leave this state alive past the window, ~task_creator at
  // SiriusContext teardown ends up releasing those BlockHandles after parts of
  // DuckDB's DatabaseInstance have already been torn down (~DBConfig fires
  // ~SiriusContext mid-DB destruction), which SIGSEGVs in ~BlockMemory.
  if (task_creator_) { task_creator_->reset(); }

  try {
    log_pool_stats(end_tag);
  } catch (...) {  // best-effort observability
  }
}

void SiriusContext::run_mandatory_cleanup_backstop(sirius::query_id_t query_id,
                                                   std::string_view end_tag) noexcept
{
  try {
    run_mandatory_cleanup(query_id, end_tag);
  } catch (std::exception& e) {
    mark_runtime_unavailable();
    drop_task_creator_state_best_effort();
    try {
      SIRIUS_LOG_ERROR(
        "Mandatory per-query cleanup failed during unwind; marking the Sirius runtime "
        "unavailable: {}",
        e.what());
    } catch (...) {
    }
  } catch (...) {
    mark_runtime_unavailable();
    drop_task_creator_state_best_effort();
    try {
      SIRIUS_LOG_ERROR(
        "Mandatory per-query cleanup failed during unwind (unknown exception); marking the "
        "Sirius runtime unavailable");
    } catch (...) {
    }
  }
}

void SiriusContext::drop_task_creator_state_best_effort() noexcept
{
  // Once the runtime is latched unavailable no later window will run the
  // task_creator reset, so the failed query's per-query global state (and the
  // buffer handles it retains) would survive until ~task_creator during DB
  // teardown — the exact shutdown-order crash the in-window reset prevents.
  try {
    if (task_creator_) { task_creator_->reset(); }
  } catch (...) {
  }
}

SiriusContext::SlotGuard::SlotGuard(SiriusContext& ctx, ClientContext& context) : ctx_(ctx)
{
  ctx_.acquire_query_lifecycle_slot(&context);
}

SiriusContext::SlotGuard::~SlotGuard() noexcept { ctx_.release_query_lifecycle_slot(); }

void SiriusContext::StandaloneQueryScope::log_window_event(char const* event,
                                                           char const* outcome) const noexcept
{
  try {
    SIRIUS_LOG_INFO("[window] {} instance={} connection={} window={} query={} outcome={}",
                    event,
                    static_cast<const void*>(&ctx_),
                    connection_id_,
                    window_id_,
                    query_ordinal_,
                    outcome);
  } catch (...) {  // logging is best-effort and must never retain the slot
  }
}

SiriusContext::StandaloneQueryScope::StandaloneQueryScope(SiriusContext& ctx,
                                                          ClientContext& context,
                                                          std::string_view window_label)
  : ctx_(ctx), window_id_(sirius::make_query_id(0)), connection_id_(0), query_ordinal_(0)
{
  if (auto conn_state = get_sirius_connection_state(context)) {
    connection_id_ = conn_state->connection_id();
    query_ordinal_ = conn_state->current_query_ordinal();
  }
  // Window id + keyed tags are prepared BEFORE the slot is acquired: after
  // acquire, no statement on any path allocates, so release cannot be skipped
  // (and the noexcept destructor cannot terminate) for an allocation reason.
  window_id_ =
    sirius::make_query_id(ctx_.next_window_id_.fetch_add(1, std::memory_order_relaxed) + 1);
  std::snprintf(begin_tag_,
                sizeof(begin_tag_),
                "QueryBegin instance=%p connection=%llu window=%llu query=%llu",
                static_cast<const void*>(&ctx_),
                static_cast<unsigned long long>(connection_id_),
                static_cast<unsigned long long>(sirius::value_of(window_id_)),
                static_cast<unsigned long long>(query_ordinal_));
  std::snprintf(end_tag_,
                sizeof(end_tag_),
                "QueryEnd instance=%p connection=%llu window=%llu query=%llu",
                static_cast<const void*>(&ctx_),
                static_cast<unsigned long long>(connection_id_),
                static_cast<unsigned long long>(sirius::value_of(window_id_)),
                static_cast<unsigned long long>(query_ordinal_));

  ctx_.acquire_query_lifecycle_slot(&context);
  log_window_event("begin", "-");
  try {
    ctx_.begin_execution_window(context, window_id_, window_label, begin_tag_);
  } catch (std::exception& e) {
    // A failed begin may have left the shared runtime part-mutated. This must
    // NEVER be classified as an ordinary GPU failure (which entry points would
    // CPU-fall-back on): latch unavailability, attempt the backstop cleanup,
    // release, and throw the distinguishable begin-failure error. Entry-point
    // catch blocks rethrow it as-is instead of falling back.
    state_ = scope_state::FAILED;
    ctx_.mark_runtime_unavailable();
    ctx_.run_mandatory_cleanup_backstop(window_id_, end_tag_);
    log_window_event("end", "begin_failed");
    ctx_.release_query_lifecycle_slot();
    throw SiriusBeginWindowFailureException(
      string("Sirius execution-window initialization failed (runtime marked unavailable): ") +
      e.what());
  } catch (...) {
    state_ = scope_state::FAILED;
    ctx_.mark_runtime_unavailable();
    ctx_.run_mandatory_cleanup_backstop(window_id_, end_tag_);
    log_window_event("end", "begin_failed");
    ctx_.release_query_lifecycle_slot();
    throw SiriusBeginWindowFailureException(
      "Sirius execution-window initialization failed (runtime marked unavailable): "
      "unknown exception");
  }
}

void SiriusContext::StandaloneQueryScope::finish()
{
  if (state_ != scope_state::ACTIVE) { return; }
  // Release is guaranteed on EVERY path out of this function — including a
  // logging throw — by this non-throwing releaser; nothing below can retain
  // the slot.
  struct slot_releaser {
    SiriusContext& ctx;
    ~slot_releaser() noexcept { ctx.release_query_lifecycle_slot(); }
  } releaser{ctx_};

  try {
    ctx_.run_mandatory_cleanup(window_id_, end_tag_);
  } catch (...) {
    // A mandatory-cleanup failure means the shared runtime can no longer be
    // trusted; the destructor must NOT run a second pass over half-cleaned
    // state. Latch unavailability, drop task_creator's per-query state (the
    // failed cleanup may have thrown before that step, and no later window
    // will run it once the latch is set — retained buffer handles must not
    // survive to DB teardown), release (via the releaser), and let the query
    // error.
    state_ = scope_state::FAILED;
    ctx_.mark_runtime_unavailable();
    ctx_.drop_task_creator_state_best_effort();
    log_window_event("end", "cleanup_failed");
    throw;
  }
  state_ = scope_state::FINISHED;
  log_window_event("end", "ok");
}

SiriusContext::StandaloneQueryScope::~StandaloneQueryScope() noexcept
{
  if (state_ != scope_state::ACTIVE) { return; }
  // Unwind path: finish() never ran (an exception escaped the window body).
  // One backstop cleanup attempt; on failure the runtime is latched
  // unavailable. The slot is released exactly once either way; logging is
  // noexcept-wrapped so the destructor can never terminate.
  ctx_.run_mandatory_cleanup_backstop(window_id_, end_tag_);
  log_window_event("end", "unwind");
  ctx_.release_query_lifecycle_slot();
  state_ = scope_state::FAILED;
}

void SiriusContext::initialize(const sirius::sirius_config& config)
{
  if (is_initialized_) { throw std::runtime_error("Sirius context is already initialized."); }

  config_ = config;
  // Validate the cached topology before any downstream construction so a stub
  // topology fails loudly rather than producing zero-GPU executors silently.
  // get_hw_topology() is the only authorised source of physical GPU/NUMA discovery — never call
  // raw CUDA/NUMA device-enumeration APIs directly elsewhere. Configured execution GPU ids come
  // from the memory manager built below.
  auto const& topo = config_.get_hw_topology();
  if (topo.num_gpus == 0) {
    throw std::runtime_error(
      "SiriusContext::initialize: cucascade::topology_discovery reported 0 GPUs — "
      "refusing to initialize on stub topology.");
  }
  SIRIUS_LOG_INFO("SiriusContext: topology summary — {} GPU(s), {} NUMA node(s), host='{}'",
                  topo.num_gpus,
                  topo.num_numa_nodes,
                  topo.hostname);
  for (auto const& gpu : topo.gpus) {
    SIRIUS_LOG_INFO(
      "  GPU {}: {} (numa={}, pci={})", gpu.id, gpu.name, gpu.numa_node, gpu.pci_bus_id);
  }

  memory_manager_ = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(
    config_.get_memory_space_configs());

  // Declare one telemetry device group per GPU so thread/queue telemetry can
  // nest under its device instead of piling up flat under the engine. The GPU
  // memory manager already reflects the configured topology.num_gpus / topology.gpu_ids
  // selection, unlike the raw hardware topology.
  std::vector<int> active_gpu_ids;
  for (auto const* gpu_space :
       memory_manager_->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)) {
    if (gpu_space != nullptr) { active_gpu_ids.push_back(gpu_space->get_device_id()); }
  }
  std::sort(active_gpu_ids.begin(), active_gpu_ids.end());
  active_gpu_ids.erase(std::unique(active_gpu_ids.begin(), active_gpu_ids.end()),
                       active_gpu_ids.end());
  telemetry_context_ = sirius::telemetry::telemetry_context::create(
    config_.get_telemetry_config(), memory_manager_.get(), active_gpu_ids);

  if (config_.get_telemetry_config().enable_quent &&
      config_.get_telemetry_config().enable_batch_events) {
    sirius::telemetry::batch_telemetry_registry::instance().install(telemetry_context_,
                                                                    *memory_manager_);
  }

  {
    auto disk_spaces = memory_manager_->get_memory_spaces_for_tier(cucascade::memory::Tier::DISK);
    if (disk_spaces.empty()) {
      SIRIUS_LOG_WARN(
        "SiriusContext: disk memory space is not configured; spilling GPU or HOST data to disk is "
        "disabled. If queries run out of GPU/HOST memory, downgrades to disk will not be "
        "possible.");
    }
  }

  // Verify cucascade built one host memory space per NUMA node. Warn rather
  // than throw — non-NUMA CI hosts legitimately report num_numa_nodes == 0.
  {
    auto const mgpu05_host_spaces =
      memory_manager_->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    SIRIUS_LOG_INFO("SiriusContext: {} host memory space(s) created for {} NUMA node(s)",
                    mgpu05_host_spaces.size(),
                    topo.num_numa_nodes);
    if (topo.num_numa_nodes > 0 &&
        mgpu05_host_spaces.size() != static_cast<size_t>(topo.num_numa_nodes)) {
      SIRIUS_LOG_WARN(
        "SiriusContext: host space count ({}) != NUMA node count ({}) — "
        "expected one host space per NUMA domain. Check "
        "sirius_config apply_defaults (.use_numa_id_as_host_id()) or YAML host "
        "configuration.",
        mgpu05_host_spaces.size(),
        topo.num_numa_nodes);
    }
  }

  // Build the single GPU<->NUMA topology index now that the memory manager's
  // GPU/HOST spaces exist. Every NUMA-aware component below (small-pinned
  // allocator, downgrade executors, task_creator, scan_manager) shares this one
  // index by shared_ptr copy instead of rebuilding its own device<->NUMA map.
  topology_index_ = std::make_shared<const sirius::memory::topology_index>(
    config_.get_hw_topology(), *memory_manager_);

  // Enable P2P peer access for every configured GPU pair.
  // cucascade::convert_gpu_to_gpu calls cudaMemcpyPeerAsync on every GPU->GPU
  // conversion. For that call to bypass host staging, peer access must be
  // enabled ONCE at init for every (src, dst) pair the host supports.
  // Non-fatal failure mode: SIRIUS_LOG_ERROR and continue — host-staged fallback
  // in cucascade's converter is a correct alternate path.
  {
    if (active_gpu_ids.size() >= 2) {
      peer_access_enabled_pairs_.reserve(active_gpu_ids.size() * (active_gpu_ids.size() - 1));
      for (int source_device : active_gpu_ids) {
        rmm::cuda_set_device_raii guard_i{rmm::cuda_device_id{source_device}};
        for (int target_device : active_gpu_ids) {
          if (source_device == target_device) continue;
          int can_access = 0;
          cudaError_t probe_err =
            cudaDeviceCanAccessPeer(&can_access, source_device, target_device);
          if (probe_err != cudaSuccess) {
            SIRIUS_LOG_ERROR("SiriusContext: cudaDeviceCanAccessPeer({},{}) failed: {}",
                             source_device,
                             target_device,
                             cudaGetErrorString(probe_err));
            continue;
          }
          if (can_access == 0) {
            SIRIUS_LOG_INFO("SiriusContext: no P2P access {} -> {} -- falling back to host staging",
                            source_device,
                            target_device);
            continue;
          }
          cudaError_t enable_err = cudaDeviceEnablePeerAccess(target_device, 0);
          // Always consume sticky error state — cudaErrorPeerAccessAlreadyEnabled
          // (and any other non-fatal condition) persists in the runtime until
          // cudaGetLastError() is called, which would make the NEXT CUDA API
          // call fail spuriously in unrelated code (e.g., thrust::exclusive_scan).
          (void)cudaGetLastError();
          if (enable_err == cudaSuccess || enable_err == cudaErrorPeerAccessAlreadyEnabled) {
            peer_access_enabled_pairs_.emplace(source_device, target_device);
            SIRIUS_LOG_INFO("SiriusContext: P2P enabled {} -> {}", source_device, target_device);
          } else {
            SIRIUS_LOG_ERROR("SiriusContext: cudaDeviceEnablePeerAccess({}) from ctx {} failed: {}",
                             target_device,
                             source_device,
                             cudaGetErrorString(enable_err));
          }
        }
      }
    } else {
      SIRIUS_LOG_INFO(
        "SiriusContext: skipping peer-access enable loop (num_gpus={}); "
        "configured GPU set has no pairs to enable",
        active_gpu_ids.size());
    }
  }

  // Configure cuDF to use our pinned slab allocator for small internal host buffers
  // (e.g. column_device_view metadata arrays in cudf::concatenate).  This eliminates
  // the pageable H2D transfers that cuDF issues by default.
  {
    auto host_spaces = memory_manager_->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (!host_spaces.empty()) {
      std::unordered_map<int, std::unique_ptr<cucascade::memory::small_pinned_host_memory_resource>>
        per_node_pools;
      int fallback_node = -1;
      for (auto* host_space : host_spaces) {
        auto* fsmr =
          host_space->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
        if (fsmr == nullptr) { continue; }
        // Key each pool by the host space's NUMA node verbatim (-1 is the
        // "unknown" sentinel), matching topology_index::numa_node_of() so the
        // allocator's device->node lookups resolve to the right pool.
        int node = host_space->get_device_id();
        if (fallback_node < 0) { fallback_node = node; }
        per_node_pools.emplace(
          node, std::make_unique<cucascade::memory::small_pinned_host_memory_resource>(*fsmr));
      }
      if (!per_node_pools.empty()) {
        if (fallback_node < 0) { fallback_node = per_node_pools.begin()->first; }
        small_pinned_allocator_ = std::make_unique<sirius::memory::numa_small_pinned_mr>(
          std::move(per_node_pools), topology_index_, fallback_node);
        small_pinned_allocator_view_.emplace(
          sirius::memory::make_host_device_resource_view_checked(small_pinned_allocator_.get()));
        prev_pinned_threshold_ = cudf::get_allocate_host_as_pinned_threshold();
        prev_pinned_mr_        = cudf::set_pinned_memory_resource(
          rmm::host_device_async_resource_ref{*small_pinned_allocator_view_});
        cudf::set_allocate_host_as_pinned_threshold(
          cucascade::memory::small_pinned_host_memory_resource::MAX_SLAB_SIZE);
        SIRIUS_LOG_INFO(
          "SiriusContext: cuDF pinned memory resource configured (NUMA-aware, max slab {} B, "
          "fallback node={})",
          cucascade::memory::small_pinned_host_memory_resource::MAX_SLAB_SIZE,
          fallback_node);
      }
    } else {
      throw std::runtime_error(
        "SiriusContext: no HOST memory space configured; pinned memory for small cuDF buffers is "
        "disabled. This may cause performance degradation due to increased pageable H2D transfers. "
        "To fix this, add a HOST memory space to the Sirius config.");
    }
  }

  // Managers are created per execution window (begin_execution_window), not here; the registry
  // starts empty and only ever holds entries for in-flight queries.

  // Create one downgrade executor per GPU memory space BEFORE task_scheduler,
  // so pointers are available for injection into gpu_pipeline_executors.
  // HOST->DISK downgrade is not yet implemented, so we skip HOST tier for now.
  //
  // Per-GPU NUMA-aware downgrade (re-authored from v1.0 dd86dd0 onto dev PR #579 shape):
  // each GPU's downgrade_executor gets its own copy of downgrade_executor_config with
  // preferred_numa_node populated from hw_topology().gpus[device_id].numa_node. The config
  // copy flows into downgrade_task via processing_loop so GPU->HOST dispatch prefers the
  // NUMA-local host memory_space via cucascade's
  // any_memory_space_in_tier_with_preference strategy.
  auto create_executors_for_tier = [&](cucascade::memory::Tier tier) {
    auto spaces          = memory_manager_->get_memory_spaces_for_tier(tier);
    auto const& base_cfg = config_.get_downgrade_executor_config();
    for (auto* space : spaces) {
      // Copy the base downgrade_executor_config so we can attach a per-GPU NUMA preference
      // without mutating the shared config owned by sirius_config.
      sirius::exec::downgrade_executor_config dg_cfg = base_cfg;
      if (tier == cucascade::memory::Tier::GPU) {
        // NUMA-local host space to prefer for GPU->HOST downgrade. -1 ("unknown",
        // e.g. single-NUMA hosts) selects the host space with device_id -1.
        dg_cfg.preferred_numa_node = topology_index_->numa_node_of(space->get_device_id());
      }
      auto executor = std::make_unique<sirius::parallel::downgrade_executor>(
        dg_cfg,
        data_repository_registry_,
        space->get_id(),
        const_cast<cucascade::memory::memory_space*>(space),
        *memory_manager_);
      // NOTE: do not call executor->start() here -- deferred until after
      // task_scheduler_ and task_creator_ are constructed.
      downgrade_executors_.push_back(std::move(executor));
    }
  };
  create_executors_for_tier(cucascade::memory::Tier::GPU);
  create_executors_for_tier(cucascade::memory::Tier::HOST);

  task_scheduler_ =
    std::make_unique<sirius::pipeline::task_scheduler>(config_.get_gpu_pipeline_executor_config(),
                                                       *memory_manager_,
                                                       telemetry_context_,
                                                       &config_.get_hw_topology(),
                                                       &downgrade_executors_);

  task_creator_ = std::make_unique<sirius::creator::task_creator>(
    config_.get_task_creator_config(), *memory_manager_, topology_index_);
  task_creator_->set_task_scheduler(*task_scheduler_);
  task_scheduler_->set_task_creator(*task_creator_);

  scan_manager_ = std::make_unique<sirius::scan_manager::sirius_scan_manager>(
    config_.get_scan_manager_config(), *memory_manager_, topology_index_);

  // Wire the pipeline task queue into downgrade executors now that task_scheduler_
  // has been constructed.
  for (auto& executor : downgrade_executors_) {
    executor->set_pipeline_task_queue(task_scheduler_->get_pipeline_task_queue());
  }

  // Start everything -- downgrade executors deferred until now
  for (auto& executor : downgrade_executors_) {
    executor->start();
  }
  task_creator_->start_thread_pool();
  scan_manager_->start();
  task_scheduler_->start();

  is_initialized_ = true;
}

void SiriusContext::terminate()
{
  throw_if_not_initialized();

  task_scheduler_->stop();
  task_scheduler_.reset();
  if (scan_manager_) { scan_manager_->stop(); }
  task_creator_->stop_thread_pool();
  task_creator_.reset();
  for (auto& executor : downgrade_executors_) {
    executor->stop();
  }
  downgrade_executors_.clear();
  sirius::telemetry::batch_telemetry_registry::instance().uninstall();
  telemetry_context_.reset();

  peer_access_enabled_pairs_.clear();

  if (memory_manager_) {
    auto gpu_spaces = memory_manager_->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    for (auto const* space : gpu_spaces) {
      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{space->get_device_id()}};
      cudaDeviceSynchronize();
    }
  }
  cudaDeviceSynchronize();

  scan_manager_.reset();

  // Drop any remaining per-query repositories while the memory manager is still alive. The
  // downgrade executors (the only other holders of a manager reference) were stopped above, so
  // no borrower can outlive this.
  data_repository_registry_.clear();

  // Restore the previous cuDF pinned memory resource and threshold before destroying the
  // slab allocator — cuDF holds a non-owning reference and would dangle after reset().
  if (prev_pinned_mr_.has_value()) {
    cudf::set_pinned_memory_resource(*prev_pinned_mr_);
    cudf::set_allocate_host_as_pinned_threshold(prev_pinned_threshold_);
    prev_pinned_mr_.reset();
  }

  // Release the slab allocator before tearing down the memory manager, since
  // its owned_allocations_ will return blocks back to the fixed_size_host_memory_resource.
  small_pinned_allocator_view_.reset();
  small_pinned_allocator_.reset();

  memory_manager_->shutdown();
  memory_manager_.reset();

  // Owns only a topology copy (no device resources); drop after the components
  // that held shared_ptr copies of it are gone.
  topology_index_.reset();

  is_initialized_ = false;
}

sirius::memory::sirius_memory_reservation_manager& SiriusContext::get_memory_manager()
{
  throw_if_not_initialized();
  return *memory_manager_;
}

const sirius::memory::sirius_memory_reservation_manager& SiriusContext::get_memory_manager() const
{
  throw_if_not_initialized();
  return *memory_manager_;
}

sirius::data::data_repository_manager_registry::manager_ptr
SiriusContext::get_data_repository_manager(sirius::query_id_t query_id) const
{
  throw_if_not_initialized();
  return data_repository_registry_.get(query_id);
}

std::vector<sirius::data::data_repository_manager_registry::manager_ptr>
SiriusContext::get_data_repository_managers() const
{
  throw_if_not_initialized();
  return data_repository_registry_.get_all();
}

sirius::data::data_repository_manager_registry& SiriusContext::get_data_repository_registry()
{
  throw_if_not_initialized();
  return data_repository_registry_;
}

sirius::pipeline::task_scheduler& SiriusContext::get_task_scheduler()
{
  throw_if_not_initialized();
  return *task_scheduler_;
}

const sirius::pipeline::task_scheduler& SiriusContext::get_task_scheduler() const
{
  throw_if_not_initialized();
  return *task_scheduler_;
}

sirius::parallel::downgrade_executor& SiriusContext::get_downgrade_executor(
  cucascade::memory::memory_space_id space_id)
{
  throw_if_not_initialized();
  for (auto& executor : downgrade_executors_) {
    if (executor->get_space_id() == space_id) { return *executor; }
  }
  throw std::runtime_error("No downgrade executor for the requested memory space");
}

const sirius::parallel::downgrade_executor& SiriusContext::get_downgrade_executor(
  cucascade::memory::memory_space_id space_id) const
{
  throw_if_not_initialized();
  for (auto& executor : downgrade_executors_) {
    if (executor->get_space_id() == space_id) { return *executor; }
  }
  throw std::runtime_error("No downgrade executor for the requested memory space");
}

const std::vector<std::unique_ptr<sirius::parallel::downgrade_executor>>&
SiriusContext::get_downgrade_executors() const
{
  throw_if_not_initialized();
  return downgrade_executors_;
}

sirius::creator::task_creator& SiriusContext::get_task_creator()
{
  throw_if_not_initialized();
  return *task_creator_;
}

const sirius::creator::task_creator& SiriusContext::get_task_creator() const
{
  throw_if_not_initialized();
  return *task_creator_;
}

sirius::scan_manager::sirius_scan_manager& SiriusContext::get_scan_manager()
{
  throw_if_not_initialized();
  return *scan_manager_;
}

const sirius::scan_manager::sirius_scan_manager& SiriusContext::get_scan_manager() const
{
  throw_if_not_initialized();
  return *scan_manager_;
}

std::shared_lock<std::shared_mutex> SiriusContext::lock_pinned_table_updates()
{
  return std::shared_lock(pinned_table_update_mutex_);
}

std::unique_lock<std::shared_mutex> SiriusContext::lock_pinned_table_registry()
{
  return std::unique_lock(pinned_table_update_mutex_);
}

std::shared_ptr<const sirius::telemetry::telemetry_context> SiriusContext::get_telemetry_context()
  const
{
  throw_if_not_initialized();
  return telemetry_context_;
}

void SiriusContext::create_query(
  duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>> pipelines,
  sirius::query_id_t query_id,
  sirius::telemetry::query_telemetry_info telemetry_info)
{
  throw_if_not_initialized();
  query_ = duckdb::make_shared_ptr<sirius::planner::query>(
    std::move(pipelines), telemetry_context_->context(), query_id, telemetry_info);
  task_scheduler_->prepare_for_query(query_);
  task_creator_->prepare_for_query(*query_);
  // Reads the admitted subset back off task_creator, so this must run after
  // initialize_internal has set it — otherwise scan_manager gets the full topology list.
  scan_manager_->prepare_for_query(*query_,
                                   config_.get_operator_params().enable_pinned_zone_map_pruning,
                                   task_creator_->get_active_gpu_ids());
}

duckdb::shared_ptr<sirius::planner::query> SiriusContext::get_query()
{
  throw_if_not_initialized();
  return query_;
}

duckdb::shared_ptr<const sirius::planner::query> SiriusContext::get_query() const
{
  throw_if_not_initialized();
  return query_;
}

bool SiriusContext::is_query_lifecycle_active() const noexcept
{
  return query_lifecycle_held_.load(std::memory_order_acquire);
}

SiriusConnectionState::SiriusConnectionState()
{
  static std::atomic<uint64_t> next_connection_id{0};
  connection_id_ = next_connection_id.fetch_add(1, std::memory_order_relaxed) + 1;
}

shared_ptr<SiriusConnectionState> get_sirius_connection_state(ClientContext& context)
{
  // Callers include noexcept per-query paths (QueryBegin/QueryEnd, the guard
  // constructors); a lookup failure must degrade to "no per-connection state"
  // (plain CPU behavior), never escape into a noexcept frame.
  try {
    static const string key = "sirius_connection_state";
    return context.registered_state->Get<SiriusConnectionState>(key);
  } catch (...) {
    return nullptr;
  }
}

SiriusContext::transparent_execution_stats SiriusContext::get_transparent_execution_stats()
  const noexcept
{
  return transparent_execution_stats{
    .successful_rebinds = transparent_rebind_success_count_.load(std::memory_order_relaxed),
    .fallbacks          = transparent_fallback_count_.load(std::memory_order_relaxed),
    .executions         = transparent_execution_count_.load(std::memory_order_relaxed),
    .runtime_fallbacks  = transparent_runtime_fallback_count_.load(std::memory_order_relaxed),
  };
}

void SiriusContext::record_transparent_rebind_success() noexcept
{
  transparent_rebind_success_count_.fetch_add(1, std::memory_order_relaxed);
}

void SiriusContext::record_transparent_fallback() noexcept
{
  transparent_fallback_count_.fetch_add(1, std::memory_order_relaxed);
}

void SiriusContext::record_transparent_execution() noexcept
{
  transparent_execution_count_.fetch_add(1, std::memory_order_relaxed);
}

void SiriusContext::record_transparent_runtime_fallback() noexcept
{
  transparent_runtime_fallback_count_.fetch_add(1, std::memory_order_relaxed);
}

SiriusContext::compressed_materialization_stats
SiriusContext::get_compressed_materialization_stats() const noexcept
{
  return compressed_materialization_stats{
    .scan_columns_narrowed =
      compressed_materialization_scan_columns_narrowed_count_.load(std::memory_order_relaxed),
    .scan_columns_restored =
      compressed_materialization_scan_columns_restored_count_.load(std::memory_order_relaxed),
    .pin_columns_narrowed =
      compressed_materialization_pin_columns_narrowed_count_.load(std::memory_order_relaxed),
    .scan_sidecars_installed =
      compressed_materialization_scan_sidecars_installed_count_.load(std::memory_order_relaxed),
    .partition_narrow_columns =
      compressed_materialization_partition_narrow_columns_count_.load(std::memory_order_relaxed),
    .scan_narrow_targets_retracted =
      compressed_materialization_scan_narrow_targets_retracted_count_.load(
        std::memory_order_relaxed),
  };
}

void SiriusContext::record_compressed_materialization_scan_columns_narrowed(uint64_t count) noexcept
{
  compressed_materialization_scan_columns_narrowed_count_.fetch_add(count,
                                                                    std::memory_order_relaxed);
}

void SiriusContext::record_compressed_materialization_scan_columns_restored(uint64_t count) noexcept
{
  compressed_materialization_scan_columns_restored_count_.fetch_add(count,
                                                                    std::memory_order_relaxed);
}

void SiriusContext::record_compressed_materialization_pin_columns_narrowed(uint64_t count) noexcept
{
  compressed_materialization_pin_columns_narrowed_count_.fetch_add(count,
                                                                   std::memory_order_relaxed);
}

void SiriusContext::record_compressed_materialization_scan_sidecar_installed() noexcept
{
  compressed_materialization_scan_sidecars_installed_count_.fetch_add(1, std::memory_order_relaxed);
}

void SiriusContext::record_compressed_materialization_partition_narrow_columns(
  uint64_t count) noexcept
{
  compressed_materialization_partition_narrow_columns_count_.fetch_add(count,
                                                                       std::memory_order_relaxed);
}

void SiriusContext::record_compressed_materialization_scan_narrow_targets_retracted(
  uint64_t count) noexcept
{
  compressed_materialization_scan_narrow_targets_retracted_count_.fetch_add(
    count, std::memory_order_relaxed);
}

namespace {

bool logical_plan_reads_s3(duckdb::LogicalOperator const& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto const& get = op.Cast<duckdb::LogicalGet>();
    if (auto const* mf = dynamic_cast<duckdb::MultiFileBindData const*>(get.bind_data.get())) {
      if (mf->file_list) {
        for (auto const& file : mf->file_list->GetAllFiles()) {
          auto const& p = file.path;
          if (p.size() > 5 && (p[0] == 's' || p[0] == 'S') && p[1] == '3' && p[2] == ':' &&
              p[3] == '/' && p[4] == '/') {
            return true;
          }
        }
      }
    }
  }
  for (auto const& child : op.children) {
    if (logical_plan_reads_s3(*child)) { return true; }
  }
  return false;
}

// S3 is GPU-only. When a transparent query that reads s3:// fails on the GPU
// path, refuse to fall back to CPU: DuckDB's CPU read_parquet would re-read the
// s3:// data through the bind-only Sirius FileSystem, which is exactly the CPU
// fallback S3 does not support. Detection is plan-based (the SQL text is not
// reliably available during prepare, and view bodies hide the s3:// literal);
// references_sirius_owned_s3_parquet on the query text is a secondary signal.
// Non-s3 (local) queries fall through to the normal CPU fallback. Surfaces the
// same clear error the gpu_execution(...) path raises.
void throw_if_s3_no_cpu_fallback(bool plan_reads_s3,
                                 std::string const& query_sql,
                                 std::string const& gpu_error)
{
  if (plan_reads_s3 || sirius::references_sirius_owned_s3_parquet(query_sql)) {
    throw std::runtime_error(
      "S3 CPU fallback is not supported: this query reads s3:// data, GPU execution failed, and "
      "Sirius has no CPU fallback for S3 data sources. Underlying GPU error: " +
      gpu_error);
  }
}

// With enable_duckdb_fallback off, surface a GPU plan-generation failure as a
// normal query error. Sanitize INTERNAL/FATAL (which would invalidate the whole
// database) to ExecutorException; keep other exception types as-is.
[[noreturn]] void rethrow_gpu_error_no_fallback(std::exception& e, const std::string& prefix)
{
  duckdb::ErrorData err(e);
  if (err.Type() == duckdb::ExceptionType::INTERNAL || err.Type() == duckdb::ExceptionType::FATAL) {
    throw duckdb::ExecutorException(prefix + err.RawMessage());
  }
  err.Throw(prefix);
}

}  // namespace

bool duckdb_fallback_enabled(ClientContext& context)
{
  Value setting;
  if (context.TryGetCurrentSetting("enable_duckdb_fallback", setting) && !setting.IsNull()) {
    return setting.GetValue<bool>();
  }
  return true;
}

bool compressed_materialization_enabled(ClientContext& context)
{
  Value setting;
  if (context.TryGetCurrentSetting("enable_compressed_materialization", setting) &&
      !setting.IsNull()) {
    return setting.GetValue<bool>();
  }
  // Reached only when the extension option is not registered, which is every caller that runs
  // without a loaded Sirius extension.
  return sirius::operator_params{}.enable_compressed_materialization;
}

void print_cpu_fallback_banner()
{
  const bool tty  = ::isatty(::fileno(stdout)) != 0;
  const char* red = tty ? "\033[1;31m" : "";
  const char* off = tty ? "\033[0m" : "";
  std::fprintf(stdout,
               "%s=============================================\n"
               "Error in Sirius GPU execution, fallback to DuckDB\n"
               "=============================================%s\n",
               red,
               off);
  std::fflush(stdout);
}

RebindQueryInfo SiriusContext::OnFinalizePrepare(ClientContext& context,
                                                 PreparedStatementData& prepared,
                                                 PreparedStatementMode mode)
{
  if (!is_internal_query_active(context)) {
    reject_update_to_pinned_table(*this, context, prepared);
  }
  if (is_internal_query_active(context)) { return RebindQueryInfo::DO_NOT_REBIND; }
  auto conn_state = get_sirius_connection_state(context);
  // Mirror the optimizer hook's gpu_execution gate: when transparent execution
  // is disabled (e.g. compare_gpu_vs_cpu's CPU run after SET gpu_execution=false),
  // never rewrite the physical plan even if we could.
  {
    duckdb::Value setting;
    auto have_setting = context.TryGetCurrentSetting("gpu_execution", setting);
    if (!have_setting || setting.IsNull() || !setting.GetValue<bool>()) {
      if (conn_state) { conn_state->clear_captured_plan(); }
      return RebindQueryInfo::DO_NOT_REBIND;
    }
  }
  if (!is_initialized_) {
    if (conn_state) { conn_state->clear_captured_plan(); }
    return RebindQueryInfo::DO_NOT_REBIND;
  }

  // Only intercept SELECT statements.
  if (prepared.statement_type != StatementType::SELECT_STATEMENT) {
    if (conn_state) { conn_state->clear_captured_plan(); }
    return RebindQueryInfo::DO_NOT_REBIND;
  }

  // If the optimizer hook captured a plan FOR THIS planning attempt, use it.
  // A generation mismatch (e.g. a leftover from Connection::ExtractPlan, which
  // never reaches finalize) is dropped rather than consumed — a stale plan can
  // bind different objects than the current statement (search path / catalog
  // drift) and must never be executed. Otherwise (no capture, or a LogicalGet
  // whose bind_data isn't serializable so plan->Copy() failed), re-plan from
  // the unbound SQL statement — this is what gpu_execution(...) does
  // internally and it works even when LogicalGet::Copy can't.
  unique_ptr<LogicalOperator> logical_plan;
  if (conn_state) { logical_plan = conn_state->take_captured_plan_if_current(); }
  // Try to capture the SQL string while the active query context is alive —
  // PreparedStatementData::unbound_statement isn't populated until *after*
  // OnFinalizePrepare returns (see ClientContext::PrepareInternal in DuckDB).
  // ClientContext::GetCurrentQuery() unconditionally derefs active_query;
  // outside a query lifecycle (e.g. plain Prepare()) it would throw, so guard
  // it. When the SQL is unavailable we still proceed with the captured plan
  // (which covers the common cases including prepared statements).
  std::string current_query_sql;
  try {
    current_query_sql = context.GetCurrentQuery();
  } catch (std::exception&) {
    current_query_sql.clear();
  }
  if (!logical_plan) {
    if (current_query_sql.empty()) { return RebindQueryInfo::DO_NOT_REBIND; }
    try {
      InternalQueryGuard guard(context);  // suppress recursive optimizer hooks
      Parser parser(context.GetParserOptions());
      parser.ParseQuery(current_query_sql);
      if (parser.statements.size() != 1) { return RebindQueryInfo::DO_NOT_REBIND; }
      Planner planner(context);
      planner.CreatePlan(std::move(parser.statements[0]));
      Optimizer optimizer(*planner.binder, context);
      logical_plan = optimizer.Optimize(std::move(planner.plan));
    } catch (InterruptException&) {
      // Cancellation is never a fallback candidate — propagate as-is.
      throw;
    } catch (SiriusRuntimeUnavailableException& e) {
      // Stable typed unavailable error: S3 keeps it as-is (no rewrite); a
      // LOCAL query falls back to the retained CPU plan when allowed.
      if (sirius::references_sirius_owned_s3_parquet(current_query_sql)) { throw; }
      if (!duckdb_fallback_enabled(context)) { throw; }
      record_transparent_fallback();
      SIRIUS_LOG_INFO("Transparent execution fallback (runtime unavailable): {}",
                      sirius::sanitized_message(e));
      return RebindQueryInfo::DO_NOT_REBIND;
    } catch (NotImplementedException& e) {
      // No captured plan to inspect on the replan path; guard on the SQL text so
      // a direct read_parquet('s3://') that fails to re-plan does not CPU-fall-back.
      throw_if_s3_no_cpu_fallback(false, current_query_sql, sirius::sanitized_message(e));
      if (!duckdb_fallback_enabled(context)) {
        rethrow_gpu_error_no_fallback(e, "GPU plan generation failed: ");
      }
      record_transparent_fallback();
      SIRIUS_LOG_INFO("Transparent execution fallback (replan unsupported): {}",
                      sirius::sanitized_message(e));
      return RebindQueryInfo::DO_NOT_REBIND;
    } catch (std::exception& e) {
      throw_if_s3_no_cpu_fallback(false, current_query_sql, sirius::sanitized_message(e));
      if (!duckdb_fallback_enabled(context)) {
        rethrow_gpu_error_no_fallback(e, "GPU plan generation failed: ");
      }
      record_transparent_fallback();
      SIRIUS_LOG_INFO("Transparent execution fallback (replan failed): {}",
                      sirius::sanitized_message(e));
      return RebindQueryInfo::DO_NOT_REBIND;
    }
    if (!logical_plan) { return RebindQueryInfo::DO_NOT_REBIND; }
  }

  // Detect an s3:// read from the plan now, while logical_plan is still intact
  // (create_plan below consumes it). S3 is GPU-only: if GPU translation fails we
  // must NOT fall back to CPU for s3:// (see throw_if_s3_no_cpu_fallback).
  bool const plan_reads_s3 = logical_plan_reads_s3(*logical_plan);

  try {
    // Plan-generation window: create_plan below reads the scan manager's pin
    // registry, which requires single-flight discipline, so the validation is
    // serialized against execution windows. Placed after the replan block so
    // its nested bind never re-enters the slot, and inside this try so a
    // runtime-unavailable error takes the existing fallback split below.
    SlotGuard plan_window(*this, context);
    // Validate that the captured logical plan is GPU-translatable before we
    // install a reusable transparent execution operator for prepared statements.
    //
    // For plans whose LogicalGet does not implement Copy (bind_data has no
    // serializer), validation runs against `logical_plan`
    // directly and consumes it; we then re-plan from `unbound_statement` for
    // the actual execution path. PhysicalSiriusExecution falls back to
    // re-planning per execute when its `logical_plan_` is null.
    sirius::planner::sirius_physical_plan_generator planner(context);
    duckdb::unique_ptr<duckdb::LogicalOperator> validation_plan;
    bool plan_is_copyable = true;
    try {
      validation_plan = sirius::transparent::copy_logical_plan(*logical_plan, context);
    } catch (NotImplementedException&) {
      plan_is_copyable = false;
    }
    if (plan_is_copyable) {
      planner.create_plan(std::move(validation_plan));
    } else {
      // Validate by consuming the freshly re-planned logical_plan; the
      // PhysicalSiriusExecution operator will re-plan again at execute time
      // using the SQL string we cached above.
      planner.create_plan(std::move(logical_plan));
      logical_plan.reset();  // signal PhysicalSiriusExecution to use the SQL replan path
    }

    SIRIUS_LOG_INFO("Transparent execution: Sirius physical plan generated successfully");

    // Stash DuckDB's CPU plan before overwriting it, wrapped in a minimal
    // PreparedStatementData. On a runtime GPU failure PhysicalSiriusExecution runs
    // it on a private Executor in the same transaction. It's collector-free here —
    // exactly what GetResultCollector expects at execute time.
    auto cpu_fallback   = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
    cpu_fallback->types = prepared.types;
    cpu_fallback->names = prepared.names;
    cpu_fallback->properties    = prepared.properties;
    cpu_fallback->physical_plan = std::move(prepared.physical_plan);

    // Create a new DuckDB PhysicalPlan containing our custom operator.
    auto new_physical_plan = make_uniq<PhysicalPlan>(Allocator::Get(context));
    auto& sirius_op =
      new_physical_plan->Make<sirius::transparent::PhysicalSiriusExecution>(std::move(logical_plan),
                                                                            current_query_sql,
                                                                            prepared.types,
                                                                            prepared.names,
                                                                            std::move(cpu_fallback),
                                                                            plan_reads_s3,
                                                                            0);
    new_physical_plan->SetRoot(sirius_op);

    // Replace the DuckDB CPU physical plan.
    prepared.physical_plan = std::move(new_physical_plan);
    record_transparent_rebind_success();

    SIRIUS_LOG_INFO("Transparent execution: physical plan replaced with GPU operator");
  } catch (InterruptException&) {
    // Cancellation is never a fallback candidate — propagate as-is.
    throw;
  } catch (SiriusRuntimeUnavailableException& e) {
    // Stable typed unavailable error: S3 keeps it as-is; LOCAL falls back to
    // the retained CPU plan when allowed.
    if (plan_reads_s3) { throw; }
    if (!duckdb_fallback_enabled(context)) { throw; }
    record_transparent_fallback();
    SIRIUS_LOG_INFO("Transparent execution fallback (runtime unavailable): {}",
                    sirius::sanitized_message(e));
    return RebindQueryInfo::DO_NOT_REBIND;
  } catch (NotImplementedException& e) {
    throw_if_s3_no_cpu_fallback(plan_reads_s3, current_query_sql, sirius::sanitized_message(e));
    if (!duckdb_fallback_enabled(context)) {
      rethrow_gpu_error_no_fallback(e, "GPU plan generation failed: ");
    }
    record_transparent_fallback();
    SIRIUS_LOG_INFO("Transparent execution fallback (unsupported): {}",
                    sirius::sanitized_message(e));
  } catch (std::exception& e) {
    throw_if_s3_no_cpu_fallback(plan_reads_s3, current_query_sql, sirius::sanitized_message(e));
    if (!duckdb_fallback_enabled(context)) {
      rethrow_gpu_error_no_fallback(e, "GPU plan generation failed: ");
    }
    record_transparent_fallback();
    SIRIUS_LOG_INFO("Transparent execution fallback: {}", sirius::sanitized_message(e));
  }

  return RebindQueryInfo::DO_NOT_REBIND;
}

RebindQueryInfo SiriusContext::OnExecutePrepared(ClientContext& context,
                                                 PreparedStatementCallbackInfo& info,
                                                 RebindQueryInfo current_rebind)
{
  auto& prepared = info.prepared_statement;
  reject_update_to_pinned_table(*this, context, prepared);

  // GPU eligibility can drift with data alone (e.g. an insert pushes a varchar past
  // the overflow-string limit) and data changes never trigger DuckDB's own rebind.
  // By execute time the CPU plan has been discarded, so a stale
  // PhysicalSiriusExecution would error with no fallback. Rebind instead:
  // OnFinalizePrepare re-decides against current stats and keeps the fresh CPU plan
  // when create_plan now refuses.
  if (!prepared.unbound_statement || !prepared.physical_plan) { return current_rebind; }
  auto& root = prepared.physical_plan->Root();
  if (root.type == sirius::transparent::PhysicalSiriusExecution::TYPE &&
      dynamic_cast<sirius::transparent::PhysicalSiriusExecution*>(&root) != nullptr) {
    return RebindQueryInfo::ATTEMPT_TO_REBIND;
  }
  return current_rebind;
}

void SiriusContext::throw_if_not_initialized() const
{
  if (!is_initialized_) { throw std::runtime_error("Sirius context is not initialized."); }
}

void SiriusContext::acquire_query_lifecycle_slot(ClientContext* context)
{
  // `| 1` keeps the sentinel (0 = free) unreachable; a cross-thread hash
  // collision could only cause a spurious diagnosable error, never a missed
  // release (the check is advisory — correctness rests on the scope-bound
  // release).
  auto const my_hash = std::hash<std::thread::id>{}(std::this_thread::get_id()) | 1;
  // Same-thread reacquire would be a silent permanent wait (plain std::mutex).
  // Only the CURRENT holder can observe its own hash here, so a match is a
  // definite programming error — surface it as a diagnosable error instead.
  if (holder_thread_hash_.load(std::memory_order_relaxed) == my_hash) {
    throw std::runtime_error(
      "Sirius internal error: query-lifecycle slot re-acquired on the holding thread "
      "(nested execution window)");
  }
  // Fast-fail before waiting: no point queuing on an unavailable runtime or
  // for an already-cancelled query.
  if (runtime_unavailable_.load(std::memory_order_acquire)) { throw_runtime_unavailable(); }
  if (context && context->IsInterrupted()) { throw InterruptException(); }

  query_lifecycle_mutex_.lock();
  holder_thread_hash_.store(my_hash, std::memory_order_relaxed);
  query_lifecycle_held_.store(true, std::memory_order_release);

  // Re-check AFTER acquiring, BEFORE any shared mutation: the previous holder
  // may have latched unavailability, and this waiter may have been cancelled,
  // while it was blocked. A cancelled waiter must never late-enter the window.
  if (runtime_unavailable_.load(std::memory_order_acquire)) {
    release_query_lifecycle_slot();
    throw_runtime_unavailable();
  }
  if (context && context->IsInterrupted()) {
    release_query_lifecycle_slot();
    throw InterruptException();
  }
}

void SiriusContext::release_query_lifecycle_slot() noexcept
{
  holder_thread_hash_.store(0, std::memory_order_relaxed);
  query_lifecycle_held_.store(false, std::memory_order_release);
  query_lifecycle_mutex_.unlock();
}

// ================= Free Functions ================= //

void install_configured_log_sink(DatabaseInstance* db)
{
  auto parsed_level = sirius::log::string_to_enum(Config::LOG_LEVEL);
  auto lvl          = parsed_level.value_or(sirius::log::level::info);

  const std::string& backend = Config::LOG_BACKEND;
  if (backend == "spdlog") {
    auto flush =
      Config::LOG_FLUSH_SECONDS <= 0
        ? std::nullopt
        : std::optional<std::chrono::milliseconds>{std::chrono::seconds{Config::LOG_FLUSH_SECONDS}};
    auto sink = sirius::log::make_spdlog_owning_sink({Config::LOG_DIR, flush});
    sink->set_level(lvl);
    sirius::log::set_sink(std::move(sink));
    // Warn only once the sink is installed, so the message actually reaches it.
    if (!parsed_level) {
      SIRIUS_LOG_WARN("Unknown log level '{}', defaulting to info", Config::LOG_LEVEL);
    }
  } else if (backend == "noop") {
    sirius::log::set_sink(sirius::log::make_noop_sink());
  } else if (backend == "duckdb") {
    // Needs a DatabaseInstance; with none, defer and leave the current sink.
    if (db) { sirius::log::set_sink(sirius::log::make_duckdb_sink(*db)); }
  } else if (db) {
    // Only report a bad backend on the db path; the db-less call is best-effort
    // and must not throw.
    throw InvalidInputException("Unknown sirius_log_backend '%s' (expected: duckdb, spdlog, noop)",
                                backend);
  }
}

SiriusContextExtensionCallback::SiriusContextExtensionCallback()
{
  if (auto* env = std::getenv("SIRIUS_LOG_BACKEND")) { Config::LOG_BACKEND = env; }
  if (auto* env = std::getenv("SIRIUS_LOG_DIR")) { Config::LOG_DIR = env; }
  if (auto* env = std::getenv("SIRIUS_LOG_LEVEL")) { Config::LOG_LEVEL = env; }
  // Install now (no db yet) so spdlog/noop capture the logs emitted by
  // read_config_file_if_exists() below; the duckdb backend needs a db (installed
  // later).
  install_configured_log_sink(nullptr);
  read_config_file_if_exists();
}

void SiriusContextExtensionCallback::OnConnectionOpened(ClientContext& context)
{
  SIRIUS_LOG_INFO("Connection opened.");
  if (context_) {
    context.registered_state->Insert("sirius_state", context_);
    // Each connection gets its OWN per-connection state (planning generation,
    // capture, label, guard depths) — unlike the shared SiriusContext above.
    context.registered_state->Insert("sirius_connection_state",
                                     duckdb::make_shared_ptr<SiriusConnectionState>());
  }
}

void SiriusContextExtensionCallback::OnConnectionClosed(ClientContext& context)
{
  SIRIUS_LOG_INFO("Connection closed.");
  // remove the context from the registered state
  context.registered_state->Remove("sirius_state");
  context.registered_state->Remove("sirius_connection_state");
}

void SiriusContextExtensionCallback::OnExtensionLoaded(DatabaseInstance& db, const string& name)
{
  SIRIUS_LOG_INFO("Extension loaded: {}", name);
}

void SiriusContextExtensionCallback::OnBeginExtensionLoad(DatabaseInstance& db, const string& name)
{
  SIRIUS_LOG_INFO("Beginning to load extension: {}", name);
}

void SiriusContextExtensionCallback::OnExtensionLoadFail(DatabaseInstance& db,
                                                         const string& name,
                                                         const ErrorData& error)
{
  SIRIUS_LOG_ERROR("Failed to load extension: {}. Error: {}", name, error.RawMessage());
}

void SiriusContextExtensionCallback::read_config_file_if_exists()
{
  // Check for explicit disable (used by benchmarks/tests that need pure CPU execution)
  if (auto* val = std::getenv("SIRIUS_DISABLE"); val != nullptr && std::string(val) != "0") {
    SIRIUS_LOG_INFO("Sirius disabled via SIRIUS_DISABLE environment variable.");
    return;
  }

  auto config_path = get_config_file_path();
  if (config_path && std::filesystem::exists(*config_path)) {
    config_.load_from_file(*config_path);
    SIRIUS_LOG_INFO("Loaded Sirius configuration from file: {}", *config_path);
  } else if (config_path) {
    // SIRIUS_CONFIG_FILE was explicitly set but points to a non-existent file — error
    auto msg = "SIRIUS_CONFIG_FILE points to non-existent file: " + *config_path;
    SIRIUS_LOG_ERROR("{}", msg);
    throw std::runtime_error(msg);
  } else {
    // Check if the user has a legacy .cfg file they may need to migrate
    if (auto legacy_path = find_legacy_config_file()) {
      SIRIUS_LOG_WARN(
        "Found legacy config file '{}'. Sirius now uses YAML configuration "
        "(sirius.yaml). Please migrate your settings to the new format. "
        "See docs/super-sirius/configuration.md for details.",
        *legacy_path);
    }
    SIRIUS_LOG_INFO(
      "No sirius.yaml found (checked $SIRIUS_CONFIG_FILE, ./sirius.yaml, "
      "~/.sirius/sirius.yaml). Using defaults.");
    SIRIUS_LOG_WARN(
      "Super Sirius will allocate most GPU and pinned host memory on startup. "
      "If you are using the legacy code path (gpu_buffer_init / gpu_processing), "
      "set SIRIUS_DISABLE=1 to prevent this.");
    config_.apply_defaults();
  }

  context_ = duckdb::make_shared_ptr<SiriusContext>();
  context_->initialize(config_);
}

}  // namespace duckdb
