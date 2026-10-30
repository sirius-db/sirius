/*
 * Copyright 2026, Sirius Contributors.
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

/**
 * @file preload_executor.hpp
 * @brief The preload executor: moves batches the downgrade executor spilled back up (HOST->GPU,
 *        DISK->GPU, else DISK->HOST) before their consumers ask for them.
 */

#include "event/query_event_subscriber.hpp"
#include "memory/device_preload_budget.hpp"
#include "memory/memory_wait_board.hpp"
#include "memory/topology_index.hpp"
#include "preload/candidate_store.hpp"
#include "preload/config.hpp"
#include "preload/consumption_tracker.hpp"
#include "preload/hw_caps.hpp"
#include "preload/preload_policy.hpp"
#include "preload/preload_stats.hpp"
#include "preload/promotion.hpp"
#include "query_id.hpp"

#include <cucascade/memory/common.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

namespace sirius::exec {
class query_lifecycle_registry;
template <typename Task>
class multi_index_priority_queue;
}  // namespace sirius::exec

namespace sirius::memory {
class sirius_memory_reservation_manager;
}  // namespace sirius::memory

namespace sirius::op {
class sirius_physical_operator;
}  // namespace sirius::op

namespace sirius::parallel {
class itask;
}  // namespace sirius::parallel

namespace sirius::pipeline {
class sirius_pipeline;
}  // namespace sirius::pipeline

namespace sirius::planner {
class query;
}  // namespace sirius::planner

namespace sirius::preload {

class preload_streams;

/**
 * @brief What the preload executor borrows. Everything must outlive the executor, except where
 *        a member says otherwise.
 */
/**
 * @brief The capacity and downgrade stop fraction of a memory space, as Sirius configures them.
 *
 * The pinned cuCascade exposes neither get_capacity() nor get_downgrade_stop_threshold(), so
 * Sirius derives both from its own memory configuration and supplies them to preload here.
 */
struct space_capacity_info {
  std::size_t capacity{0};              ///< configured capacity in bytes (base of every fraction)
  double downgrade_stop_fraction{0.0};  ///< downgrade stop threshold, as a fraction of capacity
};

/// Look up the capacity info of a memory space by its id. Required.
using space_capacity_fn = std::function<space_capacity_info(cucascade::memory::memory_space_id)>;

struct preload_dependencies {
  exec::query_lifecycle_registry& lifecycle;
  memory::sirius_memory_reservation_manager& memory;
  /// NUMA placement and C2C attributes of the GPUs. Null: every NUMA node is unknown, no GPU has
  /// C2C, and a single HOST space is local to every GPU.
  std::shared_ptr<memory::topology_index const> topology;
  /// The scheduler's task queue, visited (never modified) for class-A candidates. Null in tests
  /// that need no queued tasks.
  exec::multi_index_priority_queue<parallel::itask>* task_queue{nullptr};
  /// Hold-off state the GPU executors maintain. Required.
  std::shared_ptr<memory::memory_wait_board const> wait_board;
  /// Preload's stream pools, one per GPU. Non-null only when device preloading is enabled; must
  /// then outlive every repository that may hold a batch preload moved to a GPU.
  std::shared_ptr<preload_streams> streams;
  /// The device preloading budget board. Non-null exactly when device preloading is enabled
  /// (budget > 0 and the GPU spaces do not track reservations per stream). Shared with the task
  /// scheduler, which releases a claimed task's charges.
  std::shared_ptr<memory::device_preload_budget> device_budget;
  /// The pipeline executor's threads per GPU: the head margin in auto mode.
  int pipeline_threads_per_gpu{1};
  /// Capacity and downgrade stop fraction of each memory space, derived from Sirius's memory
  /// configuration (the pinned cuCascade has no getter for them). Required.
  space_capacity_fn space_capacity;
  hw_caps_override caps{};
};

/// One finished promotion attempt, as recorded for tests.
struct promotion_record {
  std::uint64_t batch_id{0};
  query_id_t query{};
  klass k{klass::A};
  target t{};
  cucascade::memory::memory_space_id from{cucascade::memory::Tier::SIZE, -1};
  promotion_outcome outcome{promotion_outcome::failed};
  promotion_kind kind{promotion_kind::none};
  std::size_t bytes{0};
};

/**
 * @brief The repositories that feed @p op, as preload tracks them.
 *
 * One entry per port with a repository. The destination priority is the consuming pipeline's;
 * FULL-barrier producers are every FULL port's source pipeline; a partition-draining consumer is
 * recorded when @p op is one.
 */
[[nodiscard]] std::vector<repo_info> repositories_of(op::sirius_physical_operator const& op);

/// @ref repositories_of over every operator of @p pipelines.
[[nodiscard]] std::vector<repo_info> collect_repositories(
  std::span<std::shared_ptr<pipeline::sirius_pipeline> const> pipelines);

/**
 * @brief Undoes downgrades ahead of the consumer, driven only by events.
 *
 * Candidates come from @c batch_downgraded events. A planner thread, woken only by events and by
 * its own lanes finishing jobs, orders them (inputs of queued tasks past the head margin first,
 * then repository batches whose consumer is waiting on a FULL barrier or drains partitions in
 * order) and hands them to pull-based lanes: one HOST->GPU lane per GPU and one lane per DISK
 * space. Each lane worker runs @ref promote_one.
 *
 * Concurrency:
 * - One leaf mutex guards the candidate store, the epochs, the lanes' ready lists and their
 *   counters. While holding it, the executor never calls into the task queue, a repository, a
 *   batch, the publisher or an allocator, nor takes a memory space's lock; it only reads the wait
 *   board's and the memory spaces' atomic counters and checks @c weak_ptr::expired.
 * - The planner copies what it needs under the mutex, scans the queue and the repositories
 *   without it, then reads the memory spaces' counters and installs the plan under it again.
 * - A worker takes a job under the mutex, promotes without it, and reports under it.
 * - Every wait is a condition-variable wait without a timeout. A memory or lock refusal is
 *   stamped with the epoch seen when its job was taken, so an event during the attempt always
 *   allows one more.
 *
 * Lifetime: construct, @ref start, then @ref stop (also run by the destructor). The executor must
 * be destroyed before anything it borrows.
 *
 * Limits: no-preference inputs go to a GPU only on a single-GPU system (else HOST staging), and
 * a misplaced preload is counted only for tasks inside a bucket's margin and window.
 */
class preload_executor final : public event::query_event_subscriber {
 public:
  /**
   * @param publisher Publisher of the events preload listens to; must be owned by a shared_ptr.
   * @param config Preload configuration.
   * @param deps Borrowed collaborators (see @ref preload_dependencies).
   * @throws std::invalid_argument if the wait board or the stream pools are missing.
   */
  preload_executor(event::query_event_publisher& publisher,
                   preload_executor_config config,
                   preload_dependencies deps);
  ~preload_executor() override;

  preload_executor(preload_executor const&)            = delete;
  preload_executor& operator=(preload_executor const&) = delete;
  preload_executor(preload_executor&&)                 = delete;
  preload_executor& operator=(preload_executor&&)      = delete;

  /// Start the lanes and the planner, then deliver events. Idempotent; a no-op after @ref stop.
  /// A GPUDirect Storage probe that throws is logged and counts as no GDS.
  void start();

  /// Stop delivering events, join the planner and the lanes (a running promotion finishes its
  /// batch first), and drop all candidates. Idempotent and terminal. Must not be called from an
  /// event hook.
  void stop() noexcept;

  /// Track query @p q: its repositories, from its pipelines, and the GPUs it runs on. Without
  /// this, preload ignores every event of @p q.
  void prepare_for_query(query_id_t q, planner::query const& query, std::vector<int> active_gpus);

  /// @ref prepare_for_query with the repositories already collected.
  void prepare_for_query(query_id_t q, std::vector<repo_info> repos, std::vector<int> active_gpus);

  /// Purge everything kept for @p q. Never waits; outcomes that land later only update
  /// statistics. Idempotent.
  void retire_query(query_id_t q) noexcept;

  /// After @p q's repositories are erased: purge again, log its summary and re-plan, since its
  /// memory was freed. Idempotent.
  void forget_query(query_id_t q) noexcept;

  /// Counters read one by one: not consistent across counters while the executor runs.
  [[nodiscard]] preload_stats stats() const;

  [[nodiscard]] std::string_view name() const noexcept override { return "preload"; }

  // ---- tests only ------------------------------------------------------------------------------

  /**
   * @brief Wait until at least @p min_events events were handled, no pass is pending or running,
   *        and no lane is busy or has work it could take.
   * @return false on timeout.
   */
  bool wait_until_quiescent_for_testing(std::chrono::milliseconds limit,
                                        std::uint64_t min_events = 0);
  [[nodiscard]] std::uint64_t events_handled_for_testing() const;
  /// Start recording finished promotion attempts for @ref promotion_log_for_testing.
  void enable_promotion_log_for_testing();
  [[nodiscard]] std::vector<promotion_record> promotion_log_for_testing() const;
  /// Run @p hook inside each promotion, once its target reservation is made and admitted.
  void set_promotion_hook_for_testing(std::function<void(promotion_job const&)> hook);
  /// Run @p hook inside each GPU promotion after the conversion completes and before the outcome
  /// is installed.
  void set_post_convert_hook_for_testing(std::function<void(promotion_job const&)> hook);
  /// Preloaded, unconsumed bytes on GPU index @p gpu.
  [[nodiscard]] std::size_t outstanding_bytes_for_testing(std::size_t gpu) const;
  /// The class-B part of @ref outstanding_bytes_for_testing.
  [[nodiscard]] std::size_t outstanding_speculative_bytes_for_testing(std::size_t gpu) const;
  /// Worker threads of each lane: GPUs' HOST->GPU lanes by index, then DISK lanes by index.
  [[nodiscard]] std::vector<int> lane_threads_for_testing() const;

 private:
  void on_batch_downgraded(event::event_id_t,
                           event::timestamp_t,
                           query_id_t query_id,
                           event::downgraded_batch const& batch) noexcept override;
  void on_task_completed(event::event_id_t,
                         event::timestamp_t,
                         query_id_t query_id,
                         std::size_t operator_id,
                         int gpu_id,
                         event::task_outcome outcome) noexcept override;
  void on_memory_wait_ended_for_task(event::event_id_t,
                                     event::timestamp_t,
                                     query_id_t query_id,
                                     std::size_t operator_id,
                                     int gpu_id,
                                     std::size_t granted_bytes) noexcept override;
  void on_task_created(event::event_id_t,
                       event::timestamp_t,
                       query_id_t query_id,
                       std::size_t operator_id,
                       op::SiriusPhysicalOperatorType operator_type,
                       exec::queue_priority priority) noexcept override;
  void on_task_deployed(event::event_id_t,
                        event::timestamp_t,
                        query_id_t query_id,
                        std::size_t operator_id,
                        op::SiriusPhysicalOperatorType operator_type,
                        int gpu_id) noexcept override;
  void on_pipeline_closed(event::event_id_t,
                          event::timestamp_t,
                          query_id_t query_id,
                          std::size_t pipeline_id,
                          std::size_t source_operator_id) noexcept override;
  void on_task_queue_empty(event::event_id_t, event::timestamp_t) noexcept override;
  void on_executor_awaiting_task(event::event_id_t,
                                 event::timestamp_t,
                                 int gpu_id) noexcept override;

  /// A GPU target, fixed at construction.
  struct gpu_state {
    cucascade::memory::memory_space* space{nullptr};
    int device{-1};
    int numa{-1};
    bool c2c{false};
    int h2g_cap{1};
    std::size_t capacity{0};
    gpu_ceiling_bytes ceilings{};
    std::size_t lookahead_bytes{0};
    std::size_t speculative_bytes{0};
    std::size_t max_batch_bytes{0};
    std::size_t max_inflight_bytes{0};
    /// Index of the HOST space on this GPU's NUMA node, if any.
    std::optional<std::size_t> local_host;
  };

  /// A HOST staging target, fixed at construction.
  struct host_state {
    cucascade::memory::memory_space* space{nullptr};
    int device{-1};
    std::size_t capacity{0};
    std::size_t ceiling{0};
    /// Indices of the GPUs on this HOST's NUMA node.
    std::vector<std::size_t> local_gpus;
  };

  enum class lane_kind : std::uint8_t { h2g, disk };

  /// A ready item in a lane, with what the lane predicate needs to judge it.
  struct lane_item {
    ready_item r;
    query_id_t query{};
    cucascade::memory::Tier source{cucascade::memory::Tier::SIZE};
    /// Bytes the job will reserve: what it counts against the in-flight cap.
    std::size_t est{0};
    /// Already counted as refused by the in-flight cap (carried over when the plan is rebuilt).
    bool counted_inflight{false};
  };

  /// Why a lane cannot take an item now.
  enum class take_check : std::uint8_t {
    ok,       ///< takeable
    blocked,  ///< gone, in flight, gated, held off or out of tokens
    inflight  ///< the target GPU's in-flight bytes would exceed their cap
  };

  /// A pull-based lane. Counters and the ready list are guarded by the executor's mutex.
  struct lane {
    std::size_t idx{0};
    lane_kind kind{lane_kind::h2g};
    /// GPU index (h2g) or DISK index (disk).
    std::size_t space_index{0};
    cucascade::memory::memory_space* space{nullptr};
    int threads{1};
    int effective_cap{1};
    /// DISK lanes: transfers use GPUDirect Storage, so DISK->GPU takes no HOST->GPU token.
    bool gds{false};
    /// DISK lanes: the cap no longer depends on which io context the space adopts.
    bool refined{false};
    int active{0};
    std::vector<lane_item> ready;
    std::condition_variable_any cv;
    std::vector<std::jthread> workers;
  };

  /// A job taken from a lane.
  struct taken_job {
    promotion_job job;
    lane_item item;
    bool token{false};
  };

  /// Inputs of one queued task, as the planner saw them.
  struct visited_task {
    exec::queue_priority priority{0};
    std::uint32_t position{0};
    /// GPU index of the exact bucket; empty for the no-preference bucket.
    std::optional<std::size_t> gpu;
    /// Inside the lookahead window (not in the head margin).
    bool in_window{false};
    std::vector<std::uint64_t> inputs;
  };

  /// A candidate found at the front of a repository partition.
  struct repo_item {
    query_id_t query{};
    std::uint64_t batch_id{0};
    std::uintptr_t repo{0};
    exec::queue_priority priority{0};
    std::uint32_t partition{0};
    std::uint32_t position{0};
    bool ready{false};
    bool partition_consumer{false};
    std::size_t rank{0};
    std::optional<int> placement_device;
  };

  /// Live state of one GPU or HOST space, read under the executor's mutex.
  struct space_reading {
    std::size_t consumed{0};
    /// Bytes running jobs into the space will reserve but have not yet (see @ref _unreserved).
    std::size_t unreserved{0};
    bool above_trigger{false};
  };

  /// What a planner pass did, for its log line (written after the mutex is released).
  struct pass_summary {
    std::size_t class_a_tasks{0};
    std::size_t class_b_items{0};
    std::size_t planned{0};
    std::size_t assigned{0};
    /// Each GPU's reading and its budget at the start of the assignment (kept only for the
    /// DEBUG log).
    std::vector<std::pair<space_reading, gpu_budget_state>> gpus;
    /// Each HOST's reading and its budget at the start of the assignment (likewise).
    std::vector<std::pair<space_reading, host_budget_state>> hosts;
  };

  /// What a planner pass gathered: the queue and repository scans and the DISK io contexts
  /// outside the executor's mutex, the space readings under it.
  struct plan_input {
    bool scanned{false};
    std::vector<visited_task> class_a;
    std::vector<repo_item> class_b;
    std::vector<std::pair<std::uint64_t, ledger_verdict>> verdicts;
    std::vector<space_reading> gpus;
    std::vector<space_reading> hosts;
    /// The per-pass DEBUG line is enabled: the pass keeps each space's budget for it.
    bool log_budgets{false};
  };

  /// One planned item before assignment.
  struct plan_entry {
    plan_key key{};
    std::uint64_t batch_id{0};
    query_id_t query{};
    klass k{klass::A};
    cucascade::memory::memory_space_id source{cucascade::memory::Tier::SIZE, -1};
    assign_item a{};
  };

  class lane_slot;

  void planner_loop(std::stop_token const& st) noexcept;
  void lane_loop(lane& l, std::stop_token const& st) noexcept;

  // Planner phases 1 and 2. Called without the executor mutex, which is a leaf: they take the
  // queue's lock, and hold each query's work lease while they touch its batches or repositories.
  [[nodiscard]] std::vector<visited_task> visit_queues() const;
  /// At most @c max_repository_scans repositories per pass, resuming after the last one scanned.
  /// Planner thread only: it moves @c _scan_cursor.
  void scan_repositories(pass_snapshot const& snap, plan_input& in);

  // Planner phase 3 and lane bookkeeping. Every member below requires the executor mutex.
  /// Under the mutex, so that no job is taken between this and the assignment. Reads only the
  /// spaces' atomic counters.
  void read_spaces(plan_input& in) const;
  pass_summary install_plan(plan_input const& in);
  /// @return Whether an entry was appended (false when @p c has no usable target).
  bool add_entry(std::vector<plan_entry>& entries,
                 candidate const& c,
                 target_request const& request,
                 plan_key key);
  void rebuild_ready_lists(std::vector<plan_entry> const& entries, assignment_result const& result);
  /// Pure, so safe in a wait predicate. Board loads are an early filter; the worker checks again.
  [[nodiscard]] take_check check_take(lane const& l, lane_item const& item) const noexcept;
  [[nodiscard]] bool has_takeable(lane const& l) const noexcept;
  /// Also drops items whose candidate is gone, and counts each item it finds over the in-flight
  /// cap once.
  [[nodiscard]] std::optional<taken_job> take_first_takeable(lane& l);
  [[nodiscard]] bool needs_token(lane const& l, lane_item const& item) const noexcept;
  /// @pre The job's lane slot is released.
  /// @return Whether this is the first failed promotion of its query (worth one warning).
  bool apply_result(taken_job const& taken, promotion_result const& result) noexcept;
  /// Whether a finished job can change what a new pass assigns: its outcome changed a candidate
  /// or a gate, an event came while it ran, or the last pass refused an item for room that the
  /// job's bytes were holding.
  [[nodiscard]] bool replan_after(promotion_job const& job,
                                  promotion_outcome outcome) const noexcept;
  void purge_ready_items(target t, purge_scope scope) noexcept;
  void purge_query_items(query_id_t q) noexcept;
  void clear_ready_lists() noexcept;
  void notify_lanes_with_work() noexcept;
  void mark_dirty(std::uint32_t reasons) noexcept;
  /// An idle event: asks for a pass unless one was already asked for since the last other event.
  void mark_idle() noexcept;
  [[nodiscard]] bool quiescent() const noexcept;
  [[nodiscard]] std::optional<std::size_t> lane_of(cucascade::memory::memory_space_id source,
                                                   target t) const noexcept;
  /// With no NUMA information at all, HOST 0; with NUMA known but no HOST space on that node,
  /// none, since cross-NUMA staging is a non-goal.
  [[nodiscard]] std::optional<std::size_t> staging_host(int gpu_device,
                                                        candidate const& c) const noexcept;
  [[nodiscard]] int numa_of(int gpu_device) const noexcept;
  /// The @ref _unreserved count of @p t, or null for a target that is not a GPU or HOST space.
  [[nodiscard]] std::atomic<std::size_t>* unreserved_of(target t) const noexcept;
  void log_unexpected(std::string_view where) noexcept;

  // Construction helpers.
  /// Capacity, reservation limit and downgrade stop threshold of @p space, in bytes. Capacity and
  /// the stop fraction come from @ref _space_capacity; the reservation limit from get_max_memory().
  [[nodiscard]] space_limits limits_of(cucascade::memory::memory_space const& space) const noexcept;
  void build_spaces();
  void build_lanes();
  void log_start() const noexcept;
  /// Add @p inputs, a query's input readiness, to its @p stats and to the executor's counters.
  void add_input_readiness(per_query_stats& stats, input_readiness_counts const& inputs) noexcept;
  void join_threads() noexcept;

  preload_executor_config _cfg;
  exec::query_lifecycle_registry& _lifecycle;
  memory::sirius_memory_reservation_manager& _memory;
  std::shared_ptr<memory::topology_index const> _topology;
  exec::multi_index_priority_queue<parallel::itask>* _queue;
  std::shared_ptr<memory::memory_wait_board const> _board;
  std::shared_ptr<preload_streams> _streams;
  std::shared_ptr<memory::device_preload_budget> _budget;
  /// Device preloading is enabled: GPU promotions, preload streams and the budget board exist.
  /// When false, only DISK->HOST staging runs.
  bool _device_preloading{false};
  space_capacity_fn _space_capacity;
  hw_caps_override _caps;
  std::size_t _head_margin{0};

  std::vector<gpu_state> _gpus;
  std::vector<host_state> _hosts;
  std::vector<cucascade::memory::memory_space*> _disks;
  std::vector<promotion_gpu> _promotion_gpus;
  std::vector<promotion_host> _promotion_hosts;
  /// Explicit space-to-lane maps, built in @ref build_lanes. A GPU lane exists only with device
  /// preloading, so GPU lanes do not always precede DISK lanes: the index must be looked up, not
  /// computed as `gpu_count + disk_index`.
  std::vector<std::optional<std::size_t>> _gpu_lane;  ///< GPU index -> HOST->GPU lane index
  std::vector<std::size_t> _disk_lane;                ///< DISK index -> DISK lane index

  /// Per GPU, then per HOST: bytes running jobs will reserve in the space but have not yet. A
  /// lane slot adds the job's estimate under `_mu`; @ref promote_one takes it back, without
  /// `_mu`, once the reservation exists. A planner pass loads it before the space's consumed
  /// bytes, so each running job is in exactly one of the two, or briefly in both.
  std::unique_ptr<std::atomic<std::size_t>[]> _unreserved;

  mutable preload_counters _counters;

  /// Test hook, read by each promotion.
  mutable std::mutex _hook_mu;
  std::shared_ptr<std::function<void(promotion_job const&)>> _hook;
  std::shared_ptr<std::function<void(promotion_job const&)>> _post_convert_hook;
  promotion_env _env;

  /// Guards start() and stop(), including starting and stopping event delivery.
  std::mutex _lifecycle_mu;
  enum class run_state : std::uint8_t { idle, running, stopped };
  run_state _state{run_state::idle};
  std::atomic<bool> _unexpected_logged{false};

  /// Planner thread only: the last repository scanned, where the next pass resumes.
  std::optional<std::pair<query_id_t, std::uintptr_t>> _scan_cursor;

  /// The executor's leaf lock: guards everything below.
  mutable std::mutex _mu;
  candidate_store _store;
  std::vector<std::unique_ptr<lane>> _lanes;
  std::uint32_t _dirty{0};
  bool _planner_busy{false};
  /// The last pass refused an item for room (ceiling or window), which running jobs count in.
  bool _room_refused{true};
  /// An idle event already asked for a pass, and nothing else has since. Idle events change no
  /// state, so repeating one (the scheduler re-publishes them while a device waits) adds nothing.
  bool _idle_wake_pending{false};
  std::uint64_t _events_handled{0};
  std::chrono::steady_clock::time_point _dirty_since{};
  bool _log_promotions{false};
  std::vector<promotion_record> _promotion_log;
  std::condition_variable_any _planner_cv;
  std::condition_variable_any _quiescent_cv;

  std::jthread _planner;
};

}  // namespace sirius::preload
