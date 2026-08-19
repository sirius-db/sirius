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
#include "data/data_repository_manager_registry.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "memory/resource_ref_utils.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
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
#include <shared_mutex>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cucascade::memory {
class small_pinned_host_memory_resource;
}  // namespace cucascade::memory

namespace sirius::memory {
class numa_small_pinned_mr;
class topology_index;
}  // namespace sirius::memory

namespace sirius {
class sirius_engine;
}  // namespace sirius

namespace duckdb {

/// \brief Per-connection Sirius state, registered on every ClientContext under
/// its own key ("sirius_connection_state").
///
/// Unlike the shared SiriusContext (one per DatabaseInstance, the same instance
/// on every connection), each connection owns exactly one of these, so
/// planning-attempt bookkeeping and connection-scoped flags live here without
/// cross-connection synchronization. A ClientContext serializes its own
/// operations, so the non-atomic members need no lock; the depth counters stay
/// atomic because sirius_httpfs may read them from IO threads.
class SiriusConnectionState : public ClientContextState {
 public:
  SiriusConnectionState();

  /// DuckDB calls CanRequestRebind on every registered state before each
  /// CreatePreparedStatement bind/optimize pass. That makes it the natural
  /// start-of-planning-attempt marker: bump the generation and drop any capture
  /// left by a previous attempt (e.g. Connection::ExtractPlan runs the optimizer
  /// hooks but never reaches OnFinalizePrepare, so its capture would otherwise
  /// linger). Returning false leaves the rebind decision to the shared
  /// SiriusContext, which returns true.
  bool CanRequestRebind() final
  {
    begin_planning_attempt();
    return false;
  }

  /// A new query on this connection invalidates any leftover capture.
  void QueryBegin(ClientContext& context) final { captured_plan_.reset(); }

  void QueryEnd() final { pinned_update_guard_.reset(); }

  [[nodiscard]] bool has_pinned_update_guard() const noexcept
  {
    return pinned_update_guard_.has_value();
  }
  void set_pinned_update_guard(std::shared_lock<std::shared_mutex> guard)
  {
    pinned_update_guard_.emplace(std::move(guard));
  }

  /// \brief Per-connection monotonic query ordinal, advanced by the shared
  /// SiriusContext's QueryBegin for SQL↔(instance, connection, query) log
  /// correlation.
  uint64_t next_query_ordinal() noexcept { return ++query_ordinal_; }
  [[nodiscard]] uint64_t current_query_ordinal() const noexcept { return query_ordinal_; }

  /// \brief Start a new planning attempt: advance the generation and clear any
  /// stale capture from a previous attempt.
  void begin_planning_attempt() noexcept
  {
    ++planning_generation_;
    captured_plan_.reset();
  }

  /// \brief Store the optimizer-hook capture, stamped with the current
  /// planning generation.
  void set_captured_plan(unique_ptr<LogicalOperator> plan)
  {
    captured_plan_       = std::move(plan);
    captured_generation_ = planning_generation_;
  }

  /// \brief Consume the capture iff it belongs to the CURRENT planning attempt;
  /// a stale capture (generation mismatch) is dropped and nullptr is returned,
  /// which sends OnFinalizePrepare down its existing replan-from-SQL path.
  unique_ptr<LogicalOperator> take_captured_plan_if_current()
  {
    if (!captured_plan_ || captured_generation_ != planning_generation_) {
      captured_plan_.reset();
      return nullptr;
    }
    return std::move(captured_plan_);
  }

  /// \brief Drop the capture without touching the generation (used by
  /// OnFinalizePrepare's not-taking-over early-outs).
  void clear_captured_plan() noexcept { captured_plan_.reset(); }

  void set_pending_query_label(std::string label) { pending_query_label_ = std::move(label); }
  [[nodiscard]] std::optional<std::string> take_pending_query_label()
  {
    auto label = std::move(pending_query_label_);
    pending_query_label_.reset();
    return label;
  }

  void enter_internal_query() noexcept
  {
    internal_query_depth_.fetch_add(1, std::memory_order_relaxed);
  }
  void exit_internal_query() noexcept
  {
    internal_query_depth_.fetch_sub(1, std::memory_order_relaxed);
  }
  [[nodiscard]] bool is_internal_query_active() const noexcept
  {
    return internal_query_depth_.load(std::memory_order_relaxed) > 0;
  }

  void enter_cpu_fallback() noexcept
  {
    cpu_fallback_depth_.fetch_add(1, std::memory_order_relaxed);
  }
  void exit_cpu_fallback() noexcept { cpu_fallback_depth_.fetch_sub(1, std::memory_order_relaxed); }
  [[nodiscard]] bool is_cpu_fallback_active() const noexcept
  {
    return cpu_fallback_depth_.load(std::memory_order_relaxed) > 0;
  }

  /// \brief Monotonic id for window-keyed logging (colliding DuckDB connection
  /// objects across DatabaseInstances still get distinct ids).
  [[nodiscard]] uint64_t connection_id() const noexcept { return connection_id_; }

 private:
  uint64_t planning_generation_ = 0;
  uint64_t captured_generation_ = 0;
  /// Optimizer-hook capture for the current planning attempt of THIS connection.
  unique_ptr<LogicalOperator> captured_plan_;
  /// Label set by `sirius_set_query_label`, consumed by the next
  /// sirius_interface construction on this connection.
  std::optional<std::string> pending_query_label_;
  std::atomic<int> internal_query_depth_{0};
  std::atomic<int> cpu_fallback_depth_{0};
  std::optional<std::shared_lock<std::shared_mutex>> pinned_update_guard_;
  uint64_t connection_id_;
  uint64_t query_ordinal_ = 0;
};

/// \brief Resolve the per-connection Sirius state, or nullptr when Sirius has
/// not registered on this connection.
shared_ptr<SiriusConnectionState> get_sirius_connection_state(ClientContext& context);

/// \brief Thrown by the health check when the Sirius runtime was latched
/// unavailable before this query touched it. An ExecutorException
/// (non-invalidating); entry points may fall a local query back to CPU on it,
/// but must never let the S3 branch rewrite it.
class SiriusRuntimeUnavailableException : public ExecutorException {
 public:
  explicit SiriusRuntimeUnavailableException(const string& msg) : ExecutorException(msg) {}
};

/// \brief Thrown when the execution-window BEGIN mutations failed: the shared
/// runtime was possibly part-mutated by THIS query and has been latched
/// unavailable. Entry points must rethrow it as-is — never CPU-fall-back
/// (unlike SiriusRuntimeUnavailableException above). Typed, so detection never
/// depends on message text.
class SiriusBeginWindowFailureException : public ExecutorException {
 public:
  explicit SiriusBeginWindowFailureException(const string& msg) : ExecutorException(msg) {}
};

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

  /// Monotonic counters describing compressed-materialization activity.
  ///
  /// These counters intentionally describe columns rather than queries: a
  /// single scan or pinned chunk can narrow or restore several columns.
  struct compressed_materialization_stats {
    uint64_t scan_columns_narrowed = 0;
    uint64_t scan_columns_restored = 0;
    uint64_t pin_columns_narrowed  = 0;
    /// Plan-time count of TABLE_SCAN nodes that received a narrow physical
    /// sidecar (post-residency-gate, pre-propagation/pruning — a later pass may
    /// still clear or prune it).
    uint64_t scan_sidecars_installed = 0;
    /// Runtime count of input-batch columns that crossed an engaged hash
    /// PARTITION with a carrier narrower than their native mapping. Derived
    /// from actual batch types, so a regression anywhere in the narrow-carrier
    /// chain drops it to zero.
    uint64_t partition_narrow_columns = 0;
    /// Plan-time count of narrow scan sidecar targets flipped back to native; the keep/retract rule
    /// is `apply_tier_narrowing_policy`'s.
    uint64_t scan_narrow_targets_retracted = 0;
  };

  SiriusContext();
  ~SiriusContext() noexcept override;

  // Non-copyable and non-movable
  SiriusContext(const SiriusContext&)            = delete;
  SiriusContext& operator=(const SiriusContext&) = delete;
  SiriusContext(SiriusContext&&)                 = delete;
  SiriusContext& operator=(SiriusContext&&)      = delete;

  /// \brief Called at the beginning of a query execution. Holds no lock and
  /// performs no shared mutations (those run inside the execution windows);
  /// it only logs the SQL for log-analysis correlation.
  /// \param context The client context.
  void QueryBegin(ClientContext& context) final;

  /// \brief Called at the end of a query execution. Releases nothing; slot
  /// ownership is scope-bound (StandaloneQueryScope/SlotGuard).
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
   * Some code paths (e.g. internal metadata lookups, the transparent replan)
   * must run nested planning or open a second Connection while a query is in
   * flight. The guard marks the TARGET connection's per-connection state so
   * that connection's lifecycle callbacks and optimizer hooks become no-ops;
   * unrelated connections are not affected (the old SiriusContext-wide depth
   * let one connection's guard silently disable every other connection's
   * lifecycle). The depth counter allows nesting. Resolving no per-connection
   * state (Sirius not registered) makes the guard a no-op.
   */
  struct InternalQueryGuard {
    explicit InternalQueryGuard(ClientContext& context) noexcept
      : state_(get_sirius_connection_state(context))
    {
      if (state_) { state_->enter_internal_query(); }
    }
    ~InternalQueryGuard() noexcept
    {
      if (state_) { state_->exit_internal_query(); }
    }
    InternalQueryGuard(const InternalQueryGuard&)            = delete;
    InternalQueryGuard& operator=(const InternalQueryGuard&) = delete;

   private:
    shared_ptr<SiriusConnectionState> state_;
  };

  /// \brief Whether the given connection is inside an internal-query bracket.
  [[nodiscard]] static bool is_internal_query_active(ClientContext& context) noexcept;

  /**
   * @brief RAII guard marking a CPU-fallback replay of a failed GPU query.
   *
   * Narrower than InternalQueryGuard: it fires ONLY around the CPU-fallback
   * replay, and is read ONLY by the sirius_httpfs s3:// open guard, which must
   * refuse serving s3:// data to a CPU plan. Binds to the TARGET executing
   * connection's state (the explicit fallback path replays on a different
   * Connection than the one that issued the query).
   */
  struct CpuFallbackGuard {
    explicit CpuFallbackGuard(ClientContext& context) noexcept
      : state_(get_sirius_connection_state(context))
    {
      if (state_) { state_->enter_cpu_fallback(); }
    }
    ~CpuFallbackGuard() noexcept
    {
      if (state_) { state_->exit_cpu_fallback(); }
    }
    CpuFallbackGuard(const CpuFallbackGuard&)            = delete;
    CpuFallbackGuard& operator=(const CpuFallbackGuard&) = delete;

   private:
    shared_ptr<SiriusConnectionState> state_;
  };

  /// \brief Health of the shared Sirius runtime. Set to UNAVAILABLE when a
  /// mandatory per-query cleanup step fails: the shared scan/task/repository
  /// state can no longer be trusted, so every later attempt to enter a Sirius
  /// execution or plan-generation window gets a stable, session-preserving
  /// error (never INTERNAL/FATAL — those would invalidate the whole
  /// DatabaseInstance and defeat "CPU queries continue"). CPU / non-Sirius
  /// paths never consult this.
  enum class runtime_health : uint8_t { OK, UNAVAILABLE };
  [[nodiscard]] runtime_health get_runtime_health() const noexcept
  {
    return runtime_unavailable_.load(std::memory_order_acquire) ? runtime_health::UNAVAILABLE
                                                                : runtime_health::OK;
  }
  void mark_runtime_unavailable() noexcept
  {
    runtime_unavailable_.store(true, std::memory_order_release);
  }
  /// \brief Throw the stable runtime-unavailable error (non-invalidating).
  [[noreturn]] void throw_runtime_unavailable() const;

  /**
   * @brief Lock-only RAII over the query-lifecycle slot, for plan-generation
   * windows (OnFinalizePrepare validation, explicit-path plan building,
   * runtime-sensitive SET callbacks). Same-scope/same-thread by construction:
   * the destructor releases what the constructor acquired, so no release path
   * exists outside the acquiring scope.
   */
  class SlotGuard {
   public:
    /// @p context is the acquiring connection: a cancellation that arrived
    /// while waiting is honored AFTER the lock is obtained and BEFORE any
    /// shared mutation (the cancelled waiter never late-enters the window).
    SlotGuard(SiriusContext& ctx, ClientContext& context);
    ~SlotGuard() noexcept;
    SlotGuard(const SlotGuard&)            = delete;
    SlotGuard& operator=(const SlotGuard&) = delete;

   private:
    SiriusContext& ctx_;
  };

  /**
   * @brief Full execution-window RAII: begin mutations + slot acquire in the
   * constructor; an explicit finish() runs the mandatory per-query cleanup
   * (and may throw); the destructor is a noexcept backstop that runs only when
   * finish() did not complete — it attempts the cleanup once, marks the
   * runtime UNAVAILABLE if that fails, and always releases the slot.
   *
   * Every acquire/release pair lives in one C++ scope on one thread, so
   * release is exactly-once by construction and DuckDB's QueryEnd delivery
   * (unreliable for abandoned results) plays no part in slot ownership.
   */
  class StandaloneQueryScope {
   public:
    StandaloneQueryScope(SiriusContext& ctx, ClientContext& context, std::string_view window_label);
    ~StandaloneQueryScope() noexcept;
    StandaloneQueryScope(const StandaloneQueryScope&)            = delete;
    StandaloneQueryScope& operator=(const StandaloneQueryScope&) = delete;

    /// \brief Run the mandatory per-query cleanup and release the slot. May
    /// throw (the query then errors); the destructor will not run the cleanup
    /// again after a finish() attempt — a mandatory-cleanup failure marks the
    /// runtime UNAVAILABLE instead of risking a second pass over half-cleaned
    /// state. Slot release is guaranteed on every path by a non-throwing
    /// releaser; all window logging is best-effort and can neither retain the
    /// slot nor poison the runtime.
    void finish();

    /// \brief This window's query id — the key its data repositories are registered under.
    /// Pass it to the execution path (sirius_execute_query) so operators wire into this
    /// query's manager rather than a shared one.
    [[nodiscard]] sirius::query_id_t query_id() const noexcept { return window_id_; }

   private:
    enum class scope_state : uint8_t { ACTIVE, FINISHED, FAILED };
    /// Best-effort window begin/end log line — never throws.
    void log_window_event(char const* event, char const* outcome) const noexcept;
    SiriusContext& ctx_;
    /// This window's query id; see sirius::query_id_t.
    sirius::query_id_t window_id_;
    uint64_t connection_id_;
    uint64_t query_ordinal_;
    /// Pool-stat tags carrying the full window key, rendered into fixed
    /// buffers before the slot is acquired so the logging paths between
    /// acquire and release do not depend on allocation. Release itself is
    /// guaranteed by the try/catch structure and the noexcept backstops, not
    /// by an absence of allocation in the window body.
    char begin_tag_[192] = {};
    char end_tag_[192]   = {};
    scope_state state_   = scope_state::ACTIVE;
  };

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

  /// \brief The data repository manager owned by @p query_id's query
  /// \return The manager, or nullptr
  [[nodiscard]] sirius::data::data_repository_manager_registry::manager_ptr
  get_data_repository_manager(sirius::query_id_t query_id) const;

  /// \brief Snapshot of every in-flight query's manager, ascending by query id.
  /// Memory pressure is a global condition, so the downgrade executors sweep across all of them.
  [[nodiscard]] std::vector<sirius::data::data_repository_manager_registry::manager_ptr>
  get_data_repository_managers() const;

  /// \brief The registry itself, for subsystems that hold a long-lived binding to it.
  [[nodiscard]] sirius::data::data_repository_manager_registry& get_data_repository_registry();

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

  /// Coordinate update execution with pin-registry mutations.
  std::shared_lock<std::shared_mutex> lock_pinned_table_updates();
  std::unique_lock<std::shared_mutex> lock_pinned_table_registry();

  [[nodiscard]] std::shared_ptr<const sirius::telemetry::telemetry_context> get_telemetry_context()
    const;

  /// \brief Start a query with its pipelines.
  /// \param pipelines The ordered pipelines for the query.
  /// \param telemetry_info Info useful for emitting identifiable telemetry.
  void create_query(duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>> pipelines,
                    sirius::query_id_t query_id,
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

  /// \brief Snapshot counters for transparent execution observability.
  [[nodiscard]] transparent_execution_stats get_transparent_execution_stats() const noexcept;

  [[nodiscard]] sirius::op::dynamic_filter_stats& get_dynamic_filter_stats() noexcept
  {
    return dynamic_filter_stats_;
  }
  [[nodiscard]] sirius::op::dynamic_filter_stats_snapshot get_dynamic_filter_stats_snapshot()
    const noexcept
  {
    return dynamic_filter_stats_.snapshot();
  }

  /// \brief Record a successful transparent rebind to Sirius.
  void record_transparent_rebind_success() noexcept;

  /// \brief Record a transparent fallback back to DuckDB.
  void record_transparent_fallback() noexcept;

  /// \brief Record that a transparently rebound query actually executed through Sirius.
  void record_transparent_execution() noexcept;

  /// \brief Record that a GPU execution failed at runtime and the query completed
  /// via DuckDB CPU fallback (same transaction).
  void record_transparent_runtime_fallback() noexcept;

  /// \brief Snapshot counters for compressed-materialization observability.
  [[nodiscard]] compressed_materialization_stats get_compressed_materialization_stats()
    const noexcept;

  /// \brief Record columns narrowed while materializing a scan batch.
  void record_compressed_materialization_scan_columns_narrowed(uint64_t count = 1) noexcept;

  /// \brief Record columns restored to their native type at a scan boundary.
  void record_compressed_materialization_scan_columns_restored(uint64_t count = 1) noexcept;

  /// \brief Record columns narrowed while materializing a pinned chunk.
  void record_compressed_materialization_pin_columns_narrowed(uint64_t count = 1) noexcept;

  /// \brief Record a TABLE_SCAN node that received a narrow physical sidecar at plan time.
  void record_compressed_materialization_scan_sidecar_installed() noexcept;

  /// \brief Record narrow-carrier columns crossing an engaged hash PARTITION.
  void record_compressed_materialization_partition_narrow_columns(uint64_t count = 1) noexcept;

  /// \brief Record narrow scan targets flipped back to native by the tier narrowing policy.
  void record_compressed_materialization_scan_narrow_targets_retracted(uint64_t count = 1) noexcept;

 private:
  void throw_if_not_initialized() const;
  /// Acquire the slot. Errors on same-thread reacquire — a nested acquire on
  /// one thread would otherwise be a silent permanent wait. After acquiring
  /// (and before returning) re-checks BOTH runtime health and the acquiring
  /// connection's cancellation: a cancelled waiter never late-enters the
  /// window (it releases and throws instead of running any shared mutation).
  void acquire_query_lifecycle_slot(ClientContext* context);
  void release_query_lifecycle_slot() noexcept;
  /// The begin-of-window shared mutations (repository-manager registration,
  /// task_creator reset) — runs INSIDE the held slot, per the frozen
  /// "after acquire + health check, before final create_plan" placement.
  /// GPU admission happens later, in sirius_engine::initialize_internal().
  void begin_execution_window(ClientContext& context,
                              sirius::query_id_t query_id,
                              std::string_view window_label,
                              std::string_view pool_tag);
  /// The mandatory per-query cleanup (the former QueryEnd body, order
  /// preserved). Runs INSIDE the held slot; may throw. Only the mandatory
  /// steps (query/drain/repositories/scan/task resets) can throw out of it —
  /// telemetry and logging inside are best-effort and never abort the
  /// remaining steps. @p query_id selects which query's repositories to drop;
  /// @p end_tag keys the pool-stats log line to the window.
  void run_mandatory_cleanup(sirius::query_id_t query_id, std::string_view end_tag);
  /// noexcept variant for the StandaloneQueryScope destructor backstop: one
  /// attempt; on failure marks the runtime UNAVAILABLE.
  void run_mandatory_cleanup_backstop(sirius::query_id_t query_id,
                                      std::string_view end_tag) noexcept;

  /// \brief Best-effort task_creator reset for latched-unavailable paths,
  /// where no later window will ever run the in-cleanup reset.
  void drop_task_creator_state_best_effort() noexcept;

  mutable std::mutex mutex_;
  // The Super Sirius runtime is shared across connections, so plan generation
  // and engine execution must be serialized (single-flight). The slot is
  // scope-bound: held only inside StandaloneQueryScope / SlotGuard windows
  // (acquire and release in the same scope on the same thread), never across
  // DuckDB's user-visible result lifetime, so an abandoned stream or pending
  // result holds nothing.
  std::mutex query_lifecycle_mutex_;
  // Pin and unpin take this exclusively; updates hold it from validation
  // through QueryEnd so neither operation can pass the other between checks.
  std::shared_mutex pinned_table_update_mutex_;
  std::atomic<bool> query_lifecycle_held_{false};
  // Hash of the holder's thread id, written under the gate while held, 0 when
  // free. Read (relaxed) before acquiring ONLY to detect a same-thread
  // reacquire, which is turned into a diagnosable fatal error instead of a
  // silent permanent wait.
  std::atomic<size_t> holder_thread_hash_{0};
  // See runtime_health: latched when a mandatory cleanup step fails.
  std::atomic<bool> runtime_unavailable_{false};
  // Monotonic execution-window id for window-keyed logging.
  /// 32-bit to match sirius::query_id_t, which task_creator packs into the scheduling
  /// priority. The first window gets id 1, so 0 is never a live query id.
  std::atomic<std::uint32_t> next_window_id_{0};
  bool is_initialized_ = false;
  sirius::sirius_config config_;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory_manager_;
  // Single source of truth for the GPU<->NUMA hardware topology, scoped to the
  // memory manager's reserved GPU/HOST spaces. Shared by shared_ptr copy with
  // the small-pinned allocator, downgrade executors, task_creator, and
  // scan_manager so every NUMA-aware routing decision reads one consistent
  // index instead of rebuilding ad-hoc device<->NUMA maps. Owns a copy of the
  // topology and holds no device resources, so teardown order is unconstrained.
  std::shared_ptr<const sirius::memory::topology_index> topology_index_;
  // P2P: set of (src, dst) GPU pairs where cudaDeviceEnablePeerAccess
  // succeeded in initialize(). Populated under rmm::cuda_set_device_raii, one
  // call per pair. Consumed by is_peer_access_enabled() and any Sirius-side
  // converter override. Holds no CUDA resources — just a set of int pairs —
  // so destruction order relative to memory_manager_ is unconstrained.
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
  /// One data repository manager per in-flight query, keyed by query_id.
  sirius::data::data_repository_manager_registry data_repository_registry_;
  std::unique_ptr<sirius::pipeline::task_scheduler> task_scheduler_;
  std::vector<std::unique_ptr<sirius::parallel::downgrade_executor>> downgrade_executors_;
  std::unique_ptr<sirius::creator::task_creator> task_creator_;
  std::unique_ptr<sirius::scan_manager::sirius_scan_manager> scan_manager_;
  duckdb::shared_ptr<sirius::planner::query> query_;

  sirius::op::dynamic_filter_stats dynamic_filter_stats_;
  std::atomic<uint64_t> transparent_rebind_success_count_{0};
  std::atomic<uint64_t> transparent_fallback_count_{0};
  std::atomic<uint64_t> transparent_execution_count_{0};
  std::atomic<uint64_t> transparent_runtime_fallback_count_{0};
  std::atomic<uint64_t> compressed_materialization_scan_columns_narrowed_count_{0};
  std::atomic<uint64_t> compressed_materialization_scan_columns_restored_count_{0};
  std::atomic<uint64_t> compressed_materialization_pin_columns_narrowed_count_{0};
  std::atomic<uint64_t> compressed_materialization_scan_sidecars_installed_count_{0};
  std::atomic<uint64_t> compressed_materialization_partition_narrow_columns_count_{0};
  std::atomic<uint64_t> compressed_materialization_scan_narrow_targets_retracted_count_{0};
};

/// Installs the sink selected by `Config::LOG_BACKEND` (with `Config::LOG_*`).
///
/// `spdlog` and `noop` install unconditionally; `duckdb` needs `db` and, given a
/// null `db`, defers (leaves the current sink) so a caller without one yet can
/// still select it. An unknown backend throws only when `db` is non-null, so the
/// null (best-effort) path never throws.
void install_configured_log_sink(DatabaseInstance* db);

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

  /// \brief The configuration this callback read from sirius.yaml, or compiled defaults when no
  ///        file was found.
  ///
  /// The constructor reads the file, so this is populated before InitialGPUConfigs registers the
  /// extension options. Options whose value DuckDB stores per connection take their registered
  /// default from here, which is what makes a YAML value the default every connection inherits
  /// and reports through `current_setting`.
  [[nodiscard]] const sirius::sirius_config& get_loaded_config() const noexcept { return config_; }

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

/// \brief Read the per-session `enable_compressed_materialization` setting.
///
/// DuckDB stores this value per connection, so it is the only authority on whether narrowing
/// runs: planning and `CALL pin_table` both resolve it through here, against the context whose
/// work they are doing. Reading it anywhere else would let one connection's `SET` decide another
/// connection's behavior while `current_setting` still reported the old value.
///
/// The registered default carries the YAML value (see InitialGPUConfigs), so a
/// `sirius.operator_params` entry is what a connection inherits until it sets its own.
bool compressed_materialization_enabled(ClientContext& context);

/// \brief Print the "GPU execution failed, falling back to DuckDB" banner.
///
/// Written to stdout in red (ANSI) when stdout is a TTY, plain text otherwise so
/// piped/redirected output is not corrupted. Shared by the transparent runtime
/// fallback and the legacy gpu_execution() CALL path so the message stays in sync.
void print_cpu_fallback_banner();

}  // namespace duckdb
