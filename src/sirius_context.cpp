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
#include "log/logging.hpp"
#include "memory/numa_small_pinned_mr.hpp"
#include "memory/resource_ref_utils.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "memory/topology_index.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_sql_rewrite.hpp"
#include "transparent/physical_sirius_execution.hpp"
#include "transparent/sirius_optimizer_extension.hpp"

#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>
#include <duckdb/common/allocator.hpp>
#include <duckdb/execution/physical_plan_generator.hpp>
#include <io/types.hpp>
#include <io/uring/uring_ioctx.hpp>
#include <sys/resource.h>
#include <unistd.h>  // for isatty/fileno

#include <cctype>
#include <cstddef>
#include <cstdio>   // for fprintf/fileno (fallback banner)
#include <cstdlib>  // for std::getenv
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

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
  // Suppress all state mutations for internal connections (e.g. internal metadata lookups).
  if (is_internal_query_active()) { return; }

  acquire_query_lifecycle_slot();

  try {
    log_pool_stats("QueryBegin");

    // Clear any stale captured plan from a previous query.
    captured_logical_plan_.reset();

    // Reset operator ID counter so each query starts from 0
    sirius::op::sirius_physical_operator::next_operator_id.store(0);

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
    SIRIUS_LOG_INFO("QueryBegin: SQL: {}", normalized_query);

    task_creator_->reset();
    task_creator_->set_client_context(context);
  } catch (...) {
    release_query_lifecycle_slot();
    throw;
  }
}

void SiriusContext::QueryEnd()
{
  // Suppress state mutations triggered by internal connections (e.g. internal metadata lookups).
  if (is_internal_query_active()) { return; }

  try {
    SIRIUS_LOG_INFO("QueryEnd");
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
        SIRIUS_LOG_WARN(
          "SiriusContext::QueryEnd: operator {} port '{}' still had {} un-consumed "
          "data batch(es) (memory leak).",
          info.operator_id,
          info.port_id,
          info.count);
      }
    }

    // Drop scan-manager providers for this query. Each cached_split_provider
    // holds shared_ptr copies of the pinned entry's host_chunks; if kept past
    // the query, those refs prevent fixed_size_host_memory_resource blocks from
    // returning to the pool even after unpin_table runs. Repositories are
    // already cleared above, so downstream data_batches that referenced
    // sliced host_data_representation are gone before we drop the providers.
    if (scan_manager_) { scan_manager_->reset(); }

    log_pool_stats("QueryEnd");
  } catch (...) {
    release_query_lifecycle_slot();
    throw;
  }

  // Drop per-query global states held by task_creator. These include
  // duckdb_scan_task_global_state, which transitively owns a
  // duckdb::DuckTableScanState referencing BufferManager-owned BlockHandles.
  // If we leave this state alive past QueryEnd, ~task_creator at SiriusContext
  // teardown ends up releasing those BlockHandles after parts of DuckDB's
  // DatabaseInstance have already been torn down (~DBConfig fires ~SiriusContext
  // mid-DB destruction), which SIGSEGVs in ~BlockMemory.
  if (task_creator_) { task_creator_->reset(/*keep_parquet_metadata=*/true); }
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
  // Validate the cached topology before any downstream construction so a stub
  // topology fails loudly rather than producing zero-GPU executors silently.
  // get_hw_topology() is the only authorised source of GPU/NUMA counts —
  // never call raw CUDA/NUMA device-enumeration APIs directly elsewhere.
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
  // memory space configs already reflect the configured topology.num_gpus /
  // topology.gpu_ids selection, unlike the raw hardware topology.
  std::vector<int> telemetry_gpu_ids;
  for (auto const& space_config : config_.get_memory_space_configs()) {
    if (auto const* gpu_config =
          std::get_if<cucascade::memory::gpu_memory_space_config>(&space_config)) {
      telemetry_gpu_ids.push_back(gpu_config->device_id);
    }
  }
  telemetry_context_ = sirius::telemetry::telemetry_context::create(
    config_.get_telemetry_config(), memory_manager_.get(), telemetry_gpu_ids);

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
        "sirius_config apply_defaults (.use_host_per_numa()) or YAML host "
        "configuration.",
        mgpu05_host_spaces.size(),
        topo.num_numa_nodes);
    }
  }

  // Enable P2P peer access for every available GPU pair.
  // cucascade::convert_gpu_to_gpu calls cudaMemcpyPeerAsync on every GPU->GPU
  // conversion. For that call to bypass host staging, peer access must be
  // enabled ONCE at init for every (src, dst) pair the host supports.
  // Non-fatal failure mode: SIRIUS_LOG_ERROR and continue — host-staged fallback
  // in cucascade's converter is a correct alternate path.
  {
    auto const& mgpu06_topo = config_.get_hw_topology();
    if (mgpu06_topo.num_gpus >= 2) {
      peer_access_enabled_pairs_.reserve(static_cast<size_t>(mgpu06_topo.num_gpus) *
                                         (mgpu06_topo.num_gpus - 1));
      for (unsigned i = 0; i < mgpu06_topo.num_gpus; ++i) {
        rmm::cuda_set_device_raii guard_i{rmm::cuda_device_id{static_cast<int>(i)}};
        for (unsigned j = 0; j < mgpu06_topo.num_gpus; ++j) {
          if (i == j) continue;
          int can_access = 0;
          cudaError_t probe_err =
            cudaDeviceCanAccessPeer(&can_access, static_cast<int>(i), static_cast<int>(j));
          if (probe_err != cudaSuccess) {
            SIRIUS_LOG_ERROR("SiriusContext: cudaDeviceCanAccessPeer({},{}) failed: {}",
                             i,
                             j,
                             cudaGetErrorString(probe_err));
            continue;
          }
          if (can_access == 0) {
            SIRIUS_LOG_INFO(
              "SiriusContext: no P2P access {} -> {} -- falling back to host staging", i, j);
            continue;
          }
          cudaError_t enable_err = cudaDeviceEnablePeerAccess(static_cast<int>(j), 0);
          // Always consume sticky error state — cudaErrorPeerAccessAlreadyEnabled
          // (and any other non-fatal condition) persists in the runtime until
          // cudaGetLastError() is called, which would make the NEXT CUDA API
          // call fail spuriously in unrelated code (e.g., thrust::exclusive_scan).
          (void)cudaGetLastError();
          if (enable_err == cudaSuccess || enable_err == cudaErrorPeerAccessAlreadyEnabled) {
            peer_access_enabled_pairs_.emplace(static_cast<int>(i), static_cast<int>(j));
            SIRIUS_LOG_INFO("SiriusContext: P2P enabled {} -> {}", i, j);
          } else {
            SIRIUS_LOG_ERROR("SiriusContext: cudaDeviceEnablePeerAccess({}) from ctx {} failed: {}",
                             j,
                             i,
                             cudaGetErrorString(enable_err));
          }
        }
      }
    } else {
      SIRIUS_LOG_INFO(
        "SiriusContext: skipping peer-access enable loop (num_gpus={}); "
        "single-GPU host has no pairs to enable",
        mgpu06_topo.num_gpus);
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
        int raw_numa = host_space->get_device_id();
        int node     = (raw_numa < 0) ? 0 : raw_numa;
        if (fallback_node < 0) { fallback_node = node; }
        per_node_pools.emplace(
          node, std::make_unique<cucascade::memory::small_pinned_host_memory_resource>(*fsmr));
      }
      if (!per_node_pools.empty()) {
        if (fallback_node < 0) { fallback_node = per_node_pools.begin()->first; }
        std::unordered_map<int, int> device_to_numa_copy;
        for (auto const* gpu_space :
             memory_manager_->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)) {
          int const dev = gpu_space->get_device_id();
          int raw_numa  = -1;
          if (dev >= 0 && static_cast<unsigned>(dev) < topo.gpus.size()) {
            raw_numa = topo.gpus[dev].numa_node;
          }
          device_to_numa_copy[dev] = (raw_numa < 0) ? 0 : raw_numa;
        }
        small_pinned_allocator_ = std::make_unique<sirius::memory::numa_small_pinned_mr>(
          std::move(per_node_pools), std::move(device_to_numa_copy), fallback_node);
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

  data_repository_manager_ = std::make_unique<cucascade::shared_data_repository_manager>();

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
                                                       config_.get_task_queue_ordering(),
                                                       &config_.get_hw_topology(),
                                                       &downgrade_executors_);

  task_creator_ = std::make_unique<sirius::creator::task_creator>(
    config_.get_task_creator_config(), *memory_manager_, &config_.get_hw_topology());
  task_creator_->set_task_scheduler(*task_scheduler_);
  task_scheduler_->set_task_creator(*task_creator_);

  auto hw_topology_index = std::make_shared<const sirius::memory::topology_index>(
    config_.get_hw_topology(), *memory_manager_);
  scan_manager_ = std::make_unique<sirius::scan_manager::sirius_scan_manager>(
    config_.get_scan_manager_config(), *memory_manager_, std::move(hw_topology_index));

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

  // Drop any remaining repositories while the memory manager is still alive.
  data_repository_manager_.reset();

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

std::shared_ptr<const sirius::telemetry::telemetry_context> SiriusContext::get_telemetry_context()
  const
{
  throw_if_not_initialized();
  return telemetry_context_;
}

void SiriusContext::create_query(
  duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>> pipelines,
  sirius::telemetry::query_telemetry_info telemetry_info)
{
  throw_if_not_initialized();
  query_ = duckdb::make_shared_ptr<sirius::planner::query>(
    std::move(pipelines), telemetry_context_->context(), telemetry_info);
  task_scheduler_->prepare_for_query(query_);
  task_creator_->prepare_for_query(*query_);
  scan_manager_->prepare_for_query(*query_,
                                   config_.get_operator_params().enable_pinned_zone_map_pruning);
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

void SiriusContext::set_captured_logical_plan(unique_ptr<LogicalOperator> plan)
{
  captured_logical_plan_ = std::move(plan);
}

unique_ptr<LogicalOperator> SiriusContext::take_captured_logical_plan()
{
  return std::move(captured_logical_plan_);
}

void SiriusContext::set_pending_query_label(std::string label)
{
  pending_query_label_ = std::move(label);
}

std::optional<std::string> SiriusContext::take_pending_query_label()
{
  std::optional<std::string> out;
  out.swap(pending_query_label_);
  return out;
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

  // If the optimizer hook captured a plan, use it. Otherwise (a LogicalGet whose
  // bind_data isn't serializable, so plan->Copy() failed), re-plan from the
  // unbound SQL statement — this is what gpu_execution(...) does internally and
  // it works even when LogicalGet::Copy can't.
  unique_ptr<LogicalOperator> logical_plan = take_captured_logical_plan();
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
      InternalQueryGuard guard(*this);  // suppress recursive optimizer hooks
      Parser parser(context.GetParserOptions());
      parser.ParseQuery(current_query_sql);
      if (parser.statements.size() != 1) { return RebindQueryInfo::DO_NOT_REBIND; }
      Planner planner(context);
      planner.CreatePlan(std::move(parser.statements[0]));
      Optimizer optimizer(*planner.binder, context);
      logical_plan = optimizer.Optimize(std::move(planner.plan));
    } catch (NotImplementedException& e) {
      // No captured plan to inspect on the replan path; guard on the SQL text so
      // a direct read_parquet('s3://') that fails to re-plan does not CPU-fall-back.
      throw_if_s3_no_cpu_fallback(false, current_query_sql, e.what());
      if (!duckdb_fallback_enabled(context)) {
        rethrow_gpu_error_no_fallback(e, "GPU plan generation failed: ");
      }
      record_transparent_fallback();
      SIRIUS_LOG_INFO("Transparent execution fallback (replan unsupported): {}", e.what());
      return RebindQueryInfo::DO_NOT_REBIND;
    } catch (std::exception& e) {
      throw_if_s3_no_cpu_fallback(false, current_query_sql, e.what());
      if (!duckdb_fallback_enabled(context)) {
        rethrow_gpu_error_no_fallback(e, "GPU plan generation failed: ");
      }
      record_transparent_fallback();
      SIRIUS_LOG_INFO("Transparent execution fallback (replan failed): {}", e.what());
      return RebindQueryInfo::DO_NOT_REBIND;
    }
    if (!logical_plan) { return RebindQueryInfo::DO_NOT_REBIND; }
  }

  // Detect an s3:// read from the plan now, while logical_plan is still intact
  // (create_plan below consumes it). S3 is GPU-only: if GPU translation fails we
  // must NOT fall back to CPU for s3:// (see throw_if_s3_no_cpu_fallback).
  bool const plan_reads_s3 = logical_plan_reads_s3(*logical_plan);

  try {
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
      // Preserve LogicalComparisonJoin::filter_pushdown / LogicalGet::dynamic_filters across
      // the Copy round-trip — these fields are not in DuckDB's serialization schema, so a plain
      // Copy strips them. Without this, downstream Sirius wiring would not see runtime-computed
      // dynamic filters even when DuckDB's optimizer produced them.
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
  } catch (NotImplementedException& e) {
    throw_if_s3_no_cpu_fallback(plan_reads_s3, current_query_sql, e.what());
    if (!duckdb_fallback_enabled(context)) {
      rethrow_gpu_error_no_fallback(e, "GPU plan generation failed: ");
    }
    record_transparent_fallback();
    SIRIUS_LOG_INFO("Transparent execution fallback (unsupported): {}", e.what());
  } catch (std::exception& e) {
    throw_if_s3_no_cpu_fallback(plan_reads_s3, current_query_sql, e.what());
    if (!duckdb_fallback_enabled(context)) {
      rethrow_gpu_error_no_fallback(e, "GPU plan generation failed: ");
    }
    record_transparent_fallback();
    SIRIUS_LOG_INFO("Transparent execution fallback: {}", e.what());
  }

  return RebindQueryInfo::DO_NOT_REBIND;
}

RebindQueryInfo SiriusContext::OnExecutePrepared(ClientContext& context,
                                                 PreparedStatementCallbackInfo& info,
                                                 RebindQueryInfo current_rebind)
{
  // GPU eligibility can drift with data alone (e.g. an insert pushes a varchar past
  // the overflow-string limit) and data changes never trigger DuckDB's own rebind.
  // By execute time the CPU plan has been discarded, so a stale
  // PhysicalSiriusExecution would error with no fallback. Rebind instead:
  // OnFinalizePrepare re-decides against current stats and keeps the fresh CPU plan
  // when create_plan now refuses.
  auto& prepared = info.prepared_statement;
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

void SiriusContext::acquire_query_lifecycle_slot()
{
  query_lifecycle_mutex_.lock();
  query_lifecycle_held_.store(true, std::memory_order_release);
}

void SiriusContext::release_query_lifecycle_slot()
{
  query_lifecycle_held_.store(false, std::memory_order_release);
  query_lifecycle_mutex_.unlock();
}

// ================= Free Functions ================= //

SiriusContextExtensionCallback::SiriusContextExtensionCallback()
{
  if (auto* env = std::getenv("SIRIUS_LOG_DIR")) { Config::LOG_DIR = env; }
  if (auto* env = std::getenv("SIRIUS_LOG_LEVEL")) { Config::LOG_LEVEL = env; }
  sirius::InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_SECONDS);
  read_config_file_if_exists();
}

void SiriusContextExtensionCallback::OnConnectionOpened(ClientContext& context)
{
  SIRIUS_LOG_INFO("Connection opened.");
  if (context_) { context.registered_state->Insert("sirius_state", context_); }
}

void SiriusContextExtensionCallback::OnConnectionClosed(ClientContext& context)
{
  SIRIUS_LOG_INFO("Connection closed.");
  // remove the context from the registered state
  context.registered_state->Remove("sirius_state");
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
