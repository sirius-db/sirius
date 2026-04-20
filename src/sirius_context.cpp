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
#include "log/logging.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/duckdb_scan_executor.hpp"

#include <cudf/utilities/pinned_memory.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>
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
  if (_internal_query_depth.load(std::memory_order_relaxed) > 0) { return; }

  // Reset operator ID counter so each query starts from 0
  sirius::op::sirius_physical_operator::next_operator_id.store(0);

  auto query = context.GetCurrentQuery();
  spdlog::info("QueryBegin: {}", query.substr(0, std::min(query.size(), size_t(120))));
  bool query_cache_hit = false;
  if (config_.is_scan_caching_enabled()) {
    query_cache_hit = pipeline_executor_->get_scan_executor().cache_scan_results_for_query(query);
  }
  pipeline_executor_->set_scan_caching_config(config_.get_cache_level());

  task_creator_->reset(query_cache_hit);
  task_creator_->set_client_context(context);
}

void SiriusContext::QueryEnd()
{
  // Suppress state mutations triggered by internal connections (e.g. iceberg metadata lookups).
  if (_internal_query_depth.load(std::memory_order_relaxed) > 0) { return; }

  spdlog::info("QueryEnd");
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
}

void SiriusContext::QueryEnd(ClientContext& context) { QueryEnd(); }

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
        prev_pinned_threshold_ = cudf::get_allocate_host_as_pinned_threshold();
        prev_pinned_mr_        = cudf::set_pinned_memory_resource(*small_pinned_allocator_);
        cudf::set_allocate_host_as_pinned_threshold(
          cucascade::memory::small_pinned_host_memory_resource::MAX_SLAB_SIZE);
        spdlog::info("SiriusContext: cuDF pinned memory resource configured (max slab {} B)",
                     cucascade::memory::small_pinned_host_memory_resource::MAX_SLAB_SIZE);
      }
    }
  }

  data_repository_manager_ = std::make_unique<cucascade::shared_data_repository_manager>();

  // Create one downgrade executor per GPU memory space BEFORE pipeline_executor,
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
    auto const& topo     = config_.get_hw_topology();
    for (auto* space : spaces) {
      // Copy the base downgrade_executor_config so we can attach a per-GPU NUMA preference
      // without mutating the shared config owned by sirius_config.
      sirius::exec::downgrade_executor_config dg_cfg = base_cfg;
      if (tier == cucascade::memory::Tier::GPU) {
        auto dev_id = space->get_device_id();
        if (dev_id >= 0 && static_cast<unsigned>(dev_id) < topo.gpus.size()) {
          dg_cfg.preferred_numa_node = topo.gpus[dev_id].numa_node;
        }
      }
      auto executor = std::make_unique<sirius::parallel::downgrade_executor>(
        dg_cfg,
        *data_repository_manager_,
        space->get_id(),
        const_cast<cucascade::memory::memory_space*>(space),
        *memory_manager_);
      // NOTE: do not call executor->start() here -- deferred until after
      // pipeline_executor_ and task_creator_ are constructed.
      downgrade_executors_.push_back(std::move(executor));
    }
  };
  create_executors_for_tier(cucascade::memory::Tier::GPU);

  pipeline_executor_ = std::make_unique<sirius::pipeline::pipeline_executor>(
    config_.get_gpu_pipeline_executor_config(),
    config_.get_duckdb_scan_executor_config(),
    *memory_manager_,
    &config_.get_hw_topology(),
    &downgrade_executors_);

  task_creator_ = std::make_unique<sirius::creator::task_creator>(
    config_.get_task_creator_config(), *memory_manager_, &config_.get_hw_topology());
  task_creator_->set_pipeline_executor(*pipeline_executor_);
  pipeline_executor_->set_task_creator(*task_creator_);

  // Start everything -- downgrade executors deferred until now
  for (auto& executor : downgrade_executors_) {
    executor->start();
  }
  task_creator_->start_thread_pool();
  pipeline_executor_->start();

  // Configure scan caching based on config
  pipeline_executor_->set_scan_caching_config(config_.get_cache_level());

  is_initialized_ = true;
}

void SiriusContext::terminate()
{
  throw_if_not_initialized();

  pipeline_executor_->stop();
  pipeline_executor_.reset();
  task_creator_->stop_thread_pool();
  task_creator_.reset();
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

sirius::pipeline::pipeline_executor& SiriusContext::get_pipeline_executor()
{
  throw_if_not_initialized();
  return *pipeline_executor_;
}

const sirius::pipeline::pipeline_executor& SiriusContext::get_pipeline_executor() const
{
  throw_if_not_initialized();
  return *pipeline_executor_;
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

void SiriusContext::create_query(
  duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>> pipelines)
{
  throw_if_not_initialized();
  query_ = duckdb::make_shared_ptr<sirius::planner::query>(std::move(pipelines));
  pipeline_executor_->prepare_for_query(query_);
  task_creator_->prepare_for_query(*query_);
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

void SiriusContext::throw_if_not_initialized() const
{
  if (!is_initialized_) { throw std::runtime_error("Sirius context is not initialized."); }
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
