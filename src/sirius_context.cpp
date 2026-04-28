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
#include "duckdb/common/helper.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "log/logging.hpp"
#include "memory/resource_ref_utils.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/duckdb_scan_executor.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "transparent/physical_sirius_execution.hpp"

#include <cudf/utilities/pinned_memory.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>
#include <duckdb/common/allocator.hpp>
#include <duckdb/execution/physical_plan_generator.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <cstdlib>  // for std::getenv
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace duckdb {

namespace {

static constexpr std::string_view CONFIG_FILE_NAME        = "sirius.yaml";
static constexpr std::string_view LEGACY_CONFIG_FILE_NAME = "sirius.cfg";
static constexpr std::string_view CONFIG_FILE_DIR         = ".sirius";
static constexpr std::string_view CONFIG_FILE_ENV_NAME    = "SIRIUS_CONFIG_FILE";

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

void SiriusContext::QueryBegin(ClientContext& context)
{
  // Suppress all state mutations for internal connections (e.g. iceberg metadata lookups).
  if (is_internal_query_active()) { return; }

  acquire_query_lifecycle_slot();

  try {
    // Clear any stale captured plan from a previous query.
    captured_logical_plan_.reset();

    // Reset operator ID counter so each query starts from 0
    sirius::op::sirius_physical_operator::next_operator_id.store(0);

    auto query = context.GetCurrentQuery();
    spdlog::info("QueryBegin: {}", query.substr(0, std::min(query.size(), size_t(120))));
    bool query_cache_hit = false;
    if (config_.is_scan_caching_enabled()) {
      query_cache_hit = task_scheduler_->get_scan_executor().cache_scan_results_for_query(query);
    }
    task_scheduler_->set_scan_caching_config(config_.get_cache_level());

    task_creator_->reset(query_cache_hit);
    task_creator_->set_client_context(context);
  } catch (...) {
    release_query_lifecycle_slot();
    throw;
  }
}

void SiriusContext::QueryEnd()
{
  // Suppress state mutations triggered by internal connections (e.g. iceberg metadata lookups).
  if (is_internal_query_active()) { return; }

  try {
    spdlog::info("QueryEnd");
    captured_logical_plan_.reset();
    query_.reset();

    // Drain all downgrade executors before clearing repositories — ensures no downgrade
    // tasks hold shared_ptr<data_batch> references to batches we're about to destroy.
    for (auto& executor : downgrade_executors_) {
      executor->drain();
    }

    // Clear all data repositories between queries.
    // Any batches still present are leaked — operators should have popped everything.
    if (data_repository_manager_) {
      auto leaked = data_repository_manager_->clear_all_repositories();
      for (auto const& info : leaked) {
        spdlog::warn(
          "SiriusContext::QueryEnd: operator {} port '{}' still had {} un-consumed "
          "data batch(es) (memory leak).",
          info.operator_id,
          info.port_id,
          info.count);
      }
    }
  } catch (...) {
    release_query_lifecycle_slot();
    throw;
  }

  release_query_lifecycle_slot();
}

void SiriusContext::QueryEnd(ClientContext& context)
{
  if (is_internal_query_active()) { return; }
  restore_transparent_disabled_optimizers(context);
  QueryEnd();
}

void SiriusContext::QueryEnd(ClientContext& context, optional_ptr<ErrorData> error)
{
  QueryEnd(context);
}

void SiriusContext::initialize(const sirius::sirius_config& config)
{
  if (is_initialized_) { throw std::runtime_error("Sirius context is already initialized."); }

  config_ = config;

  memory_manager_ = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(
    config_.get_memory_space_configs());

  // Configure cuDF to use our pinned slab allocator for small internal host buffers
  // (e.g. column_device_view metadata arrays in cudf::concatenate).  This eliminates
  // the pageable H2D transfers that cuDF issues by default.
  {
    auto host_spaces = memory_manager_->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (!host_spaces.empty()) {
      auto* fsmr = host_spaces[0]
                     ->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
      if (fsmr != nullptr) {
        small_pinned_allocator_ =
          std::make_unique<cucascade::memory::small_pinned_host_memory_resource>(*fsmr);
        small_pinned_allocator_view_.emplace(
          sirius::memory::make_host_device_resource_view_checked(small_pinned_allocator_.get()));
        prev_pinned_threshold_ = cudf::get_allocate_host_as_pinned_threshold();
        prev_pinned_mr_        = cudf::set_pinned_memory_resource(
          rmm::host_device_async_resource_ref{*small_pinned_allocator_view_});
        cudf::set_allocate_host_as_pinned_threshold(
          cucascade::memory::small_pinned_host_memory_resource::MAX_SLAB_SIZE);
        spdlog::info("SiriusContext: cuDF pinned memory resource configured (max slab {} B)",
                     cucascade::memory::small_pinned_host_memory_resource::MAX_SLAB_SIZE);
      }
    }
  }

  data_repository_manager_ = std::make_unique<cucascade::shared_data_repository_manager>();

  // Create one downgrade executor per GPU memory space BEFORE task_scheduler,
  // so pointers are available for injection into gpu_pipeline_executors.
  // HOST->DISK downgrade is not yet implemented, so we skip HOST tier for now.
  auto create_executors_for_tier = [&](cucascade::memory::Tier tier) {
    auto spaces        = memory_manager_->get_memory_spaces_for_tier(tier);
    auto const& dg_cfg = config_.get_downgrade_executor_config();
    for (auto* space : spaces) {
      auto executor = std::make_unique<sirius::parallel::downgrade_executor>(
        dg_cfg,
        *data_repository_manager_,
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
                                                       config_.get_duckdb_scan_executor_config(),
                                                       *memory_manager_,
                                                       &config_.get_hw_topology(),
                                                       &downgrade_executors_);

  task_creator_ = std::make_unique<sirius::creator::task_creator>(config_.get_task_creator_config(),
                                                                  *memory_manager_);
  task_creator_->set_task_scheduler(*task_scheduler_);
  task_scheduler_->set_task_creator(*task_creator_);

  scan_manager_ =
    std::make_unique<sirius::scan_manager::sirius_scan_manager>(config_.get_scan_manager_config());

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

  // Configure scan caching based on config
  task_scheduler_->set_scan_caching_config(config_.get_cache_level());

  is_initialized_ = true;
}

void SiriusContext::terminate()
{
  throw_if_not_initialized();

  task_scheduler_->stop();
  task_scheduler_.reset();
  task_creator_->stop_thread_pool();
  task_creator_.reset();
  if (scan_manager_) {
    scan_manager_->stop();
    scan_manager_->reset();
  }
  for (auto& executor : downgrade_executors_) {
    executor->stop();
  }
  downgrade_executors_.clear();

  // Ensure all CUDA operations (including async copies from downgrade tasks)
  // are complete before destroying pinned memory pools.  cudaStreamDestroy
  // returns immediately even when copies are still in-flight; without this
  // sync, the subsequent cudaFreeHost inside the memory manager destructor
  // can deadlock against a new cudaHostAlloc from the next SiriusContext.
  cudaDeviceSynchronize();

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

cucascade::shared_data_repository_manager& SiriusContext::get_data_repository_manager()
{
  throw_if_not_initialized();
  return *data_repository_manager_;
}

const cucascade::shared_data_repository_manager& SiriusContext::get_data_repository_manager() const
{
  throw_if_not_initialized();
  return *data_repository_manager_;
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

void SiriusContext::create_query(
  duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>> pipelines)
{
  throw_if_not_initialized();
  query_ = duckdb::make_shared_ptr<sirius::planner::query>(std::move(pipelines));
  task_scheduler_->prepare_for_query(query_);
  task_creator_->prepare_for_query(*query_);
  scan_manager_->prepare_for_query(*query_);
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
  std::lock_guard lock(query_lifecycle_mutex_);
  return active_query_depth_ > 0;
}

void SiriusContext::set_captured_logical_plan(unique_ptr<LogicalOperator> plan)
{
  captured_logical_plan_ = std::move(plan);
}

unique_ptr<LogicalOperator> SiriusContext::take_captured_logical_plan()
{
  return std::move(captured_logical_plan_);
}

void SiriusContext::set_transparent_original_disabled_optimizers(std::set<OptimizerType> disabled)
{
  std::lock_guard lock(mutex_);
  transparent_original_disabled_optimizers_ = std::move(disabled);
}

void SiriusContext::restore_transparent_disabled_optimizers(ClientContext& context)
{
  std::optional<std::set<OptimizerType>> original_disabled_optimizers;
  {
    std::lock_guard lock(mutex_);
    original_disabled_optimizers = std::move(transparent_original_disabled_optimizers_);
    transparent_original_disabled_optimizers_.reset();
  }

  if (original_disabled_optimizers) {
    DBConfig::GetConfig(context).options.disabled_optimizers =
      std::move(*original_disabled_optimizers);
  }
}

SiriusContext::transparent_execution_stats SiriusContext::get_transparent_execution_stats()
  const noexcept
{
  return transparent_execution_stats{
    .successful_rebinds = transparent_rebind_success_count_.load(std::memory_order_relaxed),
    .fallbacks          = transparent_fallback_count_.load(std::memory_order_relaxed),
    .executions         = transparent_execution_count_.load(std::memory_order_relaxed),
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

RebindQueryInfo SiriusContext::OnFinalizePrepare(ClientContext& context,
                                                 PreparedStatementData& prepared,
                                                 PreparedStatementMode mode)
{
  if (is_internal_query_active()) { return RebindQueryInfo::DO_NOT_REBIND; }
  // Mirror the optimizer hook's gpu_execution gate: when transparent execution
  // is disabled (e.g. compare_gpu_vs_cpu's CPU run after SET gpu_execution=false),
  // never rewrite the physical plan even if we could.
  {
    duckdb::Value setting;
    auto have_setting = context.TryGetCurrentSetting("gpu_execution", setting);
    if (!have_setting || setting.IsNull() || !setting.GetValue<bool>()) {
      captured_logical_plan_.reset();
      return RebindQueryInfo::DO_NOT_REBIND;
    }
  }
  if (!is_initialized_) {
    captured_logical_plan_.reset();
    return RebindQueryInfo::DO_NOT_REBIND;
  }

  // Only intercept SELECT statements.
  if (prepared.statement_type != StatementType::SELECT_STATEMENT) {
    captured_logical_plan_.reset();
    return RebindQueryInfo::DO_NOT_REBIND;
  }

  // If the optimizer hook captured a plan, use it. Otherwise (e.g. iceberg_scan
  // whose bind_data isn't serializable so plan->Copy() failed), re-plan from the
  // unbound SQL statement — this is what gpu_execution(...) does internally and
  // it works even when LogicalGet::Copy can't.
  unique_ptr<LogicalOperator> logical_plan = take_captured_logical_plan();
  // Try to capture the SQL string while the active query context is alive —
  // PreparedStatementData::unbound_statement isn't populated until *after*
  // OnFinalizePrepare returns (see ClientContext::PrepareInternal in DuckDB).
  // ClientContext::GetCurrentQuery() unconditionally derefs active_query;
  // outside a query lifecycle (e.g. plain Prepare()) it would throw, so guard
  // it. When the SQL is unavailable we still proceed with the captured plan
  // (which covers all non-iceberg cases including prepared statements).
  std::string current_query_sql;
  try {
    current_query_sql = context.GetCurrentQuery();
  } catch (std::exception&) {
    current_query_sql.clear();
  }
  if (!logical_plan) {
    if (current_query_sql.empty()) { return RebindQueryInfo::DO_NOT_REBIND; }
    try {
      InternalQueryGuard guard(*this);  // suppress recursive optimizer hooks
      Parser parser(context.GetParserOptions());
      parser.ParseQuery(current_query_sql);
      if (parser.statements.size() != 1) { return RebindQueryInfo::DO_NOT_REBIND; }
      Planner planner(context);
      planner.CreatePlan(std::move(parser.statements[0]));
      Optimizer optimizer(*planner.binder, context);
      logical_plan = optimizer.Optimize(std::move(planner.plan));
    } catch (NotImplementedException& e) {
      record_transparent_fallback();
      spdlog::info("Transparent execution fallback (replan unsupported): {}", e.what());
      return RebindQueryInfo::DO_NOT_REBIND;
    } catch (std::exception& e) {
      record_transparent_fallback();
      spdlog::info("Transparent execution fallback (replan failed): {}", e.what());
      return RebindQueryInfo::DO_NOT_REBIND;
    }
    if (!logical_plan) { return RebindQueryInfo::DO_NOT_REBIND; }
  }

  try {
    // Validate that the captured logical plan is GPU-translatable before we
    // install a reusable transparent execution operator for prepared statements.
    //
    // For plans whose LogicalGet does not implement Copy (e.g. iceberg_scan
    // bind_data has no serializer), validation runs against `logical_plan`
    // directly and consumes it; we then re-plan from `unbound_statement` for
    // the actual execution path. PhysicalSiriusExecution falls back to
    // re-planning per execute when its `logical_plan_` is null.
    sirius::planner::sirius_physical_plan_generator planner(context);
    duckdb::unique_ptr<duckdb::LogicalOperator> validation_plan;
    bool plan_is_copyable = true;
    try {
      validation_plan = logical_plan->Copy(context);
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

    spdlog::info("Transparent execution: Sirius physical plan generated successfully");

    // Create a new DuckDB PhysicalPlan containing our custom operator.
    auto new_physical_plan = make_uniq<PhysicalPlan>(Allocator::Get(context));
    auto& sirius_op        = new_physical_plan->Make<sirius::transparent::PhysicalSiriusExecution>(
      std::move(logical_plan), current_query_sql, prepared.types, prepared.names, 0);
    new_physical_plan->SetRoot(sirius_op);

    // Replace the DuckDB CPU physical plan.
    prepared.physical_plan = std::move(new_physical_plan);
    record_transparent_rebind_success();

    spdlog::info("Transparent execution: physical plan replaced with GPU operator");
  } catch (NotImplementedException& e) {
    record_transparent_fallback();
    spdlog::info("Transparent execution fallback (unsupported): {}", e.what());
  } catch (std::exception& e) {
    record_transparent_fallback();
    spdlog::info("Transparent execution fallback: {}", e.what());
  }

  return RebindQueryInfo::DO_NOT_REBIND;
}

void SiriusContext::throw_if_not_initialized() const
{
  if (!is_initialized_) { throw std::runtime_error("Sirius context is not initialized."); }
}

void SiriusContext::acquire_query_lifecycle_slot()
{
  std::unique_lock lock(query_lifecycle_mutex_);
  auto current_thread = std::this_thread::get_id();
  query_lifecycle_cv_.wait(
    lock, [&] { return active_query_depth_ == 0 || active_query_owner_ == current_thread; });
  active_query_owner_ = current_thread;
  active_query_depth_++;
}

void SiriusContext::release_query_lifecycle_slot()
{
  std::unique_lock lock(query_lifecycle_mutex_);
  D_ASSERT(active_query_depth_ > 0);
  D_ASSERT(active_query_owner_ == std::this_thread::get_id());
  active_query_depth_--;
  if (active_query_depth_ == 0) {
    active_query_owner_ = {};
    lock.unlock();
    query_lifecycle_cv_.notify_one();
  }
}

// ================= Free Functions ================= //

SiriusContextExtensionCallback::SiriusContextExtensionCallback()
{
  if (auto* env = std::getenv("SIRIUS_LOG_DIR")) { Config::LOG_DIR = env; }
  if (auto* env = std::getenv("SIRIUS_LOG_LEVEL")) { Config::LOG_LEVEL = env; }
  InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_SECONDS);
  read_config_file_if_exists();
}

void SiriusContextExtensionCallback::OnConnectionOpened(ClientContext& context)
{
  spdlog::info("Connection opened.");
  if (context_) { context.registered_state->Insert("sirius_state", context_); }
}

void SiriusContextExtensionCallback::OnConnectionClosed(ClientContext& context)
{
  spdlog::info("Connection closed.");
  // remove the context from the registered state
  context.registered_state->Remove("sirius_state");
}

void SiriusContextExtensionCallback::OnExtensionLoaded(DatabaseInstance& db, const string& name)
{
  spdlog::info("Extension loaded: {}", name);
}

void SiriusContextExtensionCallback::OnBeginExtensionLoad(DatabaseInstance& db, const string& name)
{
  spdlog::info("Beginning to load extension: {}", name);
}

void SiriusContextExtensionCallback::OnExtensionLoadFail(DatabaseInstance& db,
                                                         const string& name,
                                                         const ErrorData& error)
{
  spdlog::error("Failed to load extension: {}. Error: {}", name, error.RawMessage());
}

void SiriusContextExtensionCallback::read_config_file_if_exists()
{
  // Check for explicit disable (used by benchmarks/tests that need pure CPU execution)
  if (auto* val = std::getenv("SIRIUS_DISABLE"); val != nullptr && std::string(val) != "0") {
    spdlog::info("Sirius disabled via SIRIUS_DISABLE environment variable.");
    return;
  }

  auto config_path = get_config_file_path();
  if (config_path && std::filesystem::exists(*config_path)) {
    config_.load_from_file(*config_path);
    spdlog::info("Loaded Sirius configuration from file: {}", *config_path);
  } else if (config_path) {
    // SIRIUS_CONFIG_FILE was explicitly set but points to a non-existent file — error
    auto msg = "SIRIUS_CONFIG_FILE points to non-existent file: " + *config_path;
    spdlog::error("{}", msg);
    throw std::runtime_error(msg);
  } else {
    // Check if the user has a legacy .cfg file they may need to migrate
    if (auto legacy_path = find_legacy_config_file()) {
      spdlog::warn(
        "Found legacy config file '{}'. Sirius now uses YAML configuration "
        "(sirius.yaml). Please migrate your settings to the new format. "
        "See docs/super-sirius/configuration.md for details.",
        *legacy_path);
    }
    spdlog::info(
      "No sirius.yaml found (checked $SIRIUS_CONFIG_FILE, ./sirius.yaml, "
      "~/.sirius/sirius.yaml). Using defaults.");
    spdlog::warn(
      "Super Sirius will allocate most GPU and pinned host memory on startup. "
      "If you are using the legacy code path (gpu_buffer_init / gpu_processing), "
      "set SIRIUS_DISABLE=1 to prevent this.");
    config_.apply_defaults();
  }

  context_ = duckdb::make_shared_ptr<SiriusContext>();
  context_->initialize(config_);
}

}  // namespace duckdb
