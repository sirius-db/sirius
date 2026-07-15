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

#pragma once

#include "creator/task_creator.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "memory/resource_ref_utils.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/task_scheduler.hpp"
#include "planner/query.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_config.hpp"
#include "telemetry/telemetry_context.hpp"

#include <rmm/resource_ref.hpp>

#include <duckdb/common/enums/optimizer_type.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/client_context_state.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/planner/extension_callback.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cucascade::memory {
class small_pinned_host_memory_resource;
}  // namespace cucascade::memory

namespace sirius::memory {
class numa_small_pinned_mr;
}  // namespace sirius::memory

namespace sirius {
class sirius_engine;
}  // namespace sirius

namespace duckdb {

/// \brief Manages the lifetime of the sirius_context within a DuckDB ClientContext.
class SiriusContext : public ClientContextState {
 public:
  struct transparent_execution_stats {
    uint64_t successful_rebinds = 0;
    uint64_t fallbacks          = 0;
    uint64_t executions         = 0;
    // GPU execution was attempted and failed at runtime, and the query completed
    // via DuckDB CPU fallback (same transaction). Distinct from `fallbacks`, which
    // counts plan-time (create_plan) fallbacks that never reached the GPU.
    uint64_t runtime_fallbacks = 0;
  };

  SiriusContext();
  ~SiriusContext() noexcept override;

  // Non-copyable and non-movable
  SiriusContext(const SiriusContext&)            = delete;
  SiriusContext& operator=(const SiriusContext&) = delete;
  SiriusContext(SiriusContext&&)                 = delete;
  SiriusContext& operator=(SiriusContext&&)      = delete;

  /// \brief Called at the beginning of a query execution.
  /// \param context The client context.
  void QueryBegin(ClientContext& context) final;

  /// \brief Called at the end of a query execution.
  void QueryEnd() final;

  /// \brief Called at the end of a query execution with context.
  /// \param context The client context.
  void QueryEnd(ClientContext& context) final;

  /// \brief Called at the end of a query execution with context and error data.
  /// \param context The client context.
  /// \param error Optional error data.
  void QueryEnd(ClientContext& context, optional_ptr<ErrorData> error) final;

  /// \brief Must return true for OnFinalizePrepare to be called by DuckDB.
  bool CanRequestRebind() final { return true; }

  /// \brief Called after physical plan generation, before execution.
  /// Replaces the DuckDB physical plan with a Sirius GPU plan when possible.
  RebindQueryInfo OnFinalizePrepare(ClientContext& context,
                                    PreparedStatementData& prepared_statement,
                                    PreparedStatementMode mode) final;

  /// \brief Called on each execute of a reusable prepared statement. Requests a
  /// rebind for Sirius-backed plans so GPU eligibility is re-decided with current
  /// stats.
  RebindQueryInfo OnExecutePrepared(ClientContext& context,
                                    PreparedStatementCallbackInfo& info,
                                    RebindQueryInfo current_rebind) final;

  /// \brief Initialize the Sirius context with the given configuration.
  void initialize(const sirius::sirius_config& config);

  /**
   * @brief Suppress QueryBegin/QueryEnd side-effects for internal DuckDB connections.
   *
   * Some code paths (e.g. internal metadata lookups) must open a second DuckDB
   * Connection to the same database.  Because OnConnectionOpened registers the
   * SAME SiriusContext on every connection, the new connection's query lifecycle
   * callbacks would fire QueryBegin (resetting next_operator_id and resetting
   * task_creator state) and QueryEnd (clearing all data repositories), corrupting
   * the outer query's state.
   *
   * Use the RAII InternalQueryGuard to bracket any code that opens an internal
   * connection.  The depth counter allows nesting.
   */
  struct InternalQueryGuard {
    explicit InternalQueryGuard(SiriusContext& ctx) noexcept : ctx_(ctx)
    {
      ctx_.enter_internal_query();
    }
    ~InternalQueryGuard() noexcept { ctx_.exit_internal_query(); }
    InternalQueryGuard(const InternalQueryGuard&)            = delete;
    InternalQueryGuard& operator=(const InternalQueryGuard&) = delete;

   private:
    SiriusContext& ctx_;
  };

  void enter_internal_query() noexcept
  {
    _internal_query_depth.fetch_add(1, std::memory_order_relaxed);
  }
  void exit_internal_query() noexcept
  {
    _internal_query_depth.fetch_sub(1, std::memory_order_relaxed);
  }
  [[nodiscard]] bool is_internal_query_active() const noexcept
  {
    return _internal_query_depth.load(std::memory_order_relaxed) > 0;
  }

  /**
   * @brief RAII guard marking a CPU-fallback replay of a failed GPU query.
   *
   * Narrower than InternalQueryGuard: it fires ONLY around
   * run_internal_cpu_fallback_query, and is read ONLY by the sirius_httpfs
   * s3:// open guard, which must refuse serving s3:// data to a CPU plan. A
   * legitimate internal s3:// read (e.g. a future table-format metadata read) runs under
   * InternalQueryGuard but NOT this one, so it is not blocked.
   */
  struct CpuFallbackGuard {
    explicit CpuFallbackGuard(SiriusContext& ctx) noexcept : ctx_(ctx)
    {
      ctx_.enter_cpu_fallback();
    }
    ~CpuFallbackGuard() noexcept { ctx_.exit_cpu_fallback(); }
    CpuFallbackGuard(const CpuFallbackGuard&)            = delete;
    CpuFallbackGuard& operator=(const CpuFallbackGuard&) = delete;

   private:
    SiriusContext& ctx_;
  };

  void enter_cpu_fallback() noexcept
  {
    _cpu_fallback_depth.fetch_add(1, std::memory_order_relaxed);
  }
  void exit_cpu_fallback() noexcept { _cpu_fallback_depth.fetch_sub(1, std::memory_order_relaxed); }
  [[nodiscard]] bool is_cpu_fallback_active() const noexcept
  {
    return _cpu_fallback_depth.load(std::memory_order_relaxed) > 0;
  }

  /// \brief Terminate the Sirius context, releasing all resources.
  void terminate();

  /// \brief Log host and GPU memory pool stats (allocated, peak, and
  ///        tier-specific capacity fields) at a labeled tag — used for
  ///        verifying that allocations return to baseline after each query.
  void log_pool_stats(std::string_view tag) const;

  [[nodiscard]] const cucascade::memory::system_topology_info& get_hw_topology() const noexcept
  {
    return config_.get_hw_topology();
  }

  /// \brief Get the memory reservation manager.
  [[nodiscard]] sirius::memory::sirius_memory_reservation_manager& get_memory_manager();
  [[nodiscard]] const sirius::memory::sirius_memory_reservation_manager& get_memory_manager() const;

  [[nodiscard]] cucascade::shared_data_repository_manager& get_data_repository_manager();
  [[nodiscard]] const cucascade::shared_data_repository_manager& get_data_repository_manager()
    const;

  [[nodiscard]] sirius::pipeline::task_scheduler& get_task_scheduler();
  [[nodiscard]] const sirius::pipeline::task_scheduler& get_task_scheduler() const;

  /// \brief Get the downgrade executor for a specific memory space.
  [[nodiscard]] sirius::parallel::downgrade_executor& get_downgrade_executor(
    cucascade::memory::memory_space_id space_id);
  [[nodiscard]] const sirius::parallel::downgrade_executor& get_downgrade_executor(
    cucascade::memory::memory_space_id space_id) const;

  /// \brief Get all downgrade executors.
  [[nodiscard]] const std::vector<std::unique_ptr<sirius::parallel::downgrade_executor>>&
  get_downgrade_executors() const;

  /// @brief Check whether cudaDeviceEnablePeerAccess succeeded for the given
  ///        (src, dst) GPU pair at SiriusContext::initialize() time.
  ///
  /// Used by Sirius-side P2P-aware converter override (if registered) and by
  /// integration tests verifying the adaptive-scan + P2P path. Returns false
  /// if either device index is out of range, if cudaDeviceCanAccessPeer
  /// reported no access, or if cudaDeviceEnablePeerAccess returned an error
  /// other than cudaErrorPeerAccessAlreadyEnabled.
  ///
  /// @param src Source GPU device id
  /// @param dst Destination GPU device id
  /// @return true iff peer access was successfully enabled at init time.
  [[nodiscard]] bool is_peer_access_enabled(int src, int dst) const noexcept
  {
    return peer_access_enabled_pairs_.count({src, dst}) > 0;
  }

  [[nodiscard]] sirius::creator::task_creator& get_task_creator();
  [[nodiscard]] const sirius::creator::task_creator& get_task_creator() const;

  [[nodiscard]] sirius::scan_manager::sirius_scan_manager& get_scan_manager();
  [[nodiscard]] const sirius::scan_manager::sirius_scan_manager& get_scan_manager() const;

  [[nodiscard]] std::shared_ptr<const sirius::telemetry::telemetry_context> get_telemetry_context()
    const;

  /// \brief Start a query with its pipelines.
  /// \param pipelines The ordered pipelines for the query.
  /// \param telemetry_info Info useful for emitting identifiable telemetry.
  void create_query(duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>> pipelines,
                    sirius::telemetry::query_telemetry_info telemetry_info);

  /// \brief Get the current query.
  [[nodiscard]] duckdb::shared_ptr<sirius::planner::query> get_query();
  [[nodiscard]] duckdb::shared_ptr<const sirius::planner::query> get_query() const;

  /// \brief Get the current Sirius configuration (const).
  [[nodiscard]] const sirius::sirius_config& get_config() const noexcept { return config_; }

  /// \brief Get the current Sirius configuration (mutable, e.g. for SET command callbacks).
  [[nodiscard]] sirius::sirius_config& get_config() noexcept { return config_; }

  /// \brief Whether the Sirius context has been initialized (config loaded, GPU ready).
  [[nodiscard]] bool is_initialized() const noexcept { return is_initialized_; }

  /// \brief Whether the shared query lifecycle slot is currently held by any connection.
  [[nodiscard]] bool is_query_lifecycle_active() const noexcept;

  /// \brief Store a captured logical plan for transparent GPU execution.
  /// Called by the optimizer extension hook after copying the optimized logical plan.
  void set_captured_logical_plan(duckdb::unique_ptr<duckdb::LogicalOperator> plan);

  /// \brief Take ownership of the captured logical plan (moves it out).
  /// Called by OnFinalizePrepare to generate the Sirius physical plan.
  duckdb::unique_ptr<duckdb::LogicalOperator> take_captured_logical_plan();

  /// \brief Stash a label for the next query to pick up and use as a telemetry
  /// label for easy identification. Set by the `sirius_set_query_label` SQL
  /// function; consumed once by the next sirius_interface construction
  /// (transparent path or gpu_execution).
  void set_pending_query_label(std::string label);

  /// \brief Take and clear the stashed pending query label.
  [[nodiscard]] std::optional<std::string> take_pending_query_label();

  /// \brief Save the connection's disabled optimizer set before transparent execution mutates it.
  void set_transparent_original_disabled_optimizers(std::set<duckdb::OptimizerType> disabled);

  /// \brief Restore the connection's disabled optimizer set after transparent optimization.
  void restore_transparent_disabled_optimizers(ClientContext& context);

  /// \brief Snapshot counters for transparent execution observability.
  [[nodiscard]] transparent_execution_stats get_transparent_execution_stats() const noexcept;

  /// \brief Record a successful transparent rebind to Sirius.
  void record_transparent_rebind_success() noexcept;

  /// \brief Record a transparent fallback back to DuckDB.
  void record_transparent_fallback() noexcept;

  /// \brief Record that a transparently rebound query actually executed through Sirius.
  void record_transparent_execution() noexcept;

  /// \brief Record that a GPU execution failed at runtime and the query completed
  /// via DuckDB CPU fallback (same transaction).
  void record_transparent_runtime_fallback() noexcept;

 private:
  void throw_if_not_initialized() const;
  void acquire_query_lifecycle_slot();
  void release_query_lifecycle_slot();

  mutable std::mutex mutex_;
  std::atomic<int> _internal_query_depth{0};
  std::atomic<int> _cpu_fallback_depth{0};
  // The current Super Sirius runtime is shared across connections, so query
  // lifecycle callbacks and engine execution must be serialized to avoid
  // cross-connection state corruption. Held for the duration of
  // QueryBegin→QueryEnd; InternalQueryGuard paths skip it.
  std::mutex query_lifecycle_mutex_;
  std::atomic<bool> query_lifecycle_held_{false};
  bool is_initialized_ = false;
  sirius::sirius_config config_;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory_manager_;
  // P2P: set of (src, dst) GPU pairs where cudaDeviceEnablePeerAccess
  // succeeded in initialize(). Populated under rmm::cuda_set_device_raii, one
  // call per pair. Consumed by is_peer_access_enabled() and any Sirius-side
  // converter override. Holds no CUDA resources — just a set of int pairs —
  // so destruction order relative to memory_manager_ is unconstrained;
  // placed adjacent to gpu_ioctxs_ for multi-GPU state locality.
  struct peer_pair_hash {
    size_t operator()(std::pair<int, int> const& p) const noexcept
    {
      return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(p.second);
    }
  };
  std::unordered_set<std::pair<int, int>, peer_pair_hash> peer_access_enabled_pairs_;
  // NUMA-aware cuDF small-pinned MR. Owns one
  // small_pinned_host_memory_resource per host space (one per NUMA node)
  // and dispatches each cuDF allocate/deallocate to the slab pool whose
  // NUMA node matches the current CUDA device. Replaces the previous
  // single-pool path that hardcoded host_spaces[0] and funneled every
  // GPU's cuDF metadata allocation through one NUMA domain.
  // Destroyed before memory_manager_ (declared after it — reverse
  // destruction order). prev_pinned_mr_ is restored in terminate() before
  // these are torn down to prevent cuDF from holding a dangling ref.
  std::unique_ptr<sirius::memory::numa_small_pinned_mr> small_pinned_allocator_;
  std::optional<sirius::memory::host_device_resource_view<sirius::memory::numa_small_pinned_mr>>
    small_pinned_allocator_view_{};
  // Previous cuDF pinned resource and threshold — restored in terminate() before the view and
  // allocator are destroyed to prevent dangling references.
  std::optional<rmm::host_device_async_resource_ref> prev_pinned_mr_{};
  std::size_t prev_pinned_threshold_{0};
  std::shared_ptr<const sirius::telemetry::telemetry_context> telemetry_context_;
  std::unique_ptr<cucascade::shared_data_repository_manager> data_repository_manager_;
  std::unique_ptr<sirius::pipeline::task_scheduler> task_scheduler_;
  std::vector<std::unique_ptr<sirius::parallel::downgrade_executor>> downgrade_executors_;
  std::unique_ptr<sirius::creator::task_creator> task_creator_;
  std::unique_ptr<sirius::scan_manager::sirius_scan_manager> scan_manager_;
  duckdb::shared_ptr<sirius::planner::query> query_;

  /// Captured optimized logical plan for transparent GPU execution.
  /// Set by the optimizer extension hook, consumed by OnFinalizePrepare.
  duckdb::unique_ptr<duckdb::LogicalOperator> captured_logical_plan_;

  /// Label set by the `sirius_set_query_label` SQL function, consumed at the
  /// next sirius_interface construction site. Cleared on take.
  std::optional<std::string> pending_query_label_{std::nullopt};

  /// Snapshot of the connection's disabled optimizer set before the transparent
  /// optimizer hook mutates it.
  std::optional<std::set<duckdb::OptimizerType>> transparent_original_disabled_optimizers_;

  std::atomic<uint64_t> transparent_rebind_success_count_{0};
  std::atomic<uint64_t> transparent_fallback_count_{0};
  std::atomic<uint64_t> transparent_execution_count_{0};
  std::atomic<uint64_t> transparent_runtime_fallback_count_{0};
};

/// todo(amin): when duckdb is updated, we need to enable OnExtensionLoaded to support sirius
/// extensions
class SiriusContextExtensionCallback : public ExtensionCallback {
 public:
  SiriusContextExtensionCallback();

  /// \brief Called when a new connection is opened.
  /// \param context The client context.
  void OnConnectionOpened(ClientContext& context) final;

  /// \brief Called when a connection is closed.
  /// \param context The client context.
  void OnConnectionClosed(ClientContext& context) final;

  /// \brief Called when an extension is loaded.
  /// \param db The database instance.
  /// \param name The name of the loaded extension.
  void OnExtensionLoaded(DatabaseInstance& db, const string& name) final;

  void OnBeginExtensionLoad(DatabaseInstance& db, const string& name) final;

  //! Called after an extension fails to load loading
  void OnExtensionLoadFail(DatabaseInstance& db, const string& name, const ErrorData& error) final;

 private:
  void read_config_file_if_exists();

  sirius::sirius_config config_;
  duckdb::shared_ptr<SiriusContext> context_;
};

/// \brief Read the per-session `enable_duckdb_fallback` setting (default true).
///
/// Gates both plan-time and runtime fallback from GPU to DuckDB CPU. Set per
/// connection via `SET enable_duckdb_fallback = ...`.
bool duckdb_fallback_enabled(ClientContext& context);

/// \brief Print the "GPU execution failed, falling back to DuckDB" banner.
///
/// Written to stdout in red (ANSI) when stdout is a TTY, plain text otherwise so
/// piped/redirected output is not corrupted. Shared by the transparent runtime
/// fallback and the legacy gpu_execution() CALL path so the message stays in sync.
void print_cpu_fallback_banner();

}  // namespace duckdb
