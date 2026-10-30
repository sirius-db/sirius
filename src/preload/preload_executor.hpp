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
 * @brief The preload executor: moves spilled inputs of queued tasks back up (HOST->GPU,
 *        DISK->GPU, else DISK->HOST) before the tasks run.
 */

#include "event/downgraded_task.hpp"
#include "event/query_event_subscriber.hpp"
#include "exec/queue_priority.hpp"
#include "preload/config.hpp"
#include "preload/promotion.hpp"
#include "query_id.hpp"

#include <cucascade/memory/common.hpp>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cucascade {
class data_batch;
namespace memory {
class memory_space;
}  // namespace memory
}  // namespace cucascade

namespace sirius::exec {
class query_lifecycle_registry;
}  // namespace sirius::exec

namespace sirius::memory {
class sirius_memory_reservation_manager;
class topology_index;
}  // namespace sirius::memory

namespace sirius::preload {

class preload_streams;

/**
 * @brief The capacity and downgrade stop fraction of a memory space, as Sirius configures them.
 *
 * The pinned cuCascade exposes neither, so Sirius derives both from its memory configuration.
 */
struct space_capacity_info {
  std::size_t capacity{0};  ///< configured capacity in bytes: the base of every fraction
  double downgrade_stop_fraction{0.0};  ///< downgrade stop threshold, as a fraction of capacity
};

/// Look up the capacity info of a memory space by its id.
using space_capacity_fn = std::function<space_capacity_info(cucascade::memory::memory_space_id)>;

/// What the preload executor borrows. Everything must outlive the executor.
struct preload_dependencies {
  exec::query_lifecycle_registry& lifecycle;
  memory::sirius_memory_reservation_manager& memory;
  /// NUMA placement of the GPUs. Null: the first HOST space is local to every GPU.
  std::shared_ptr<memory::topology_index const> topology;
  /// Preload's stream pools, one per GPU. Required when @c device_budget > 0; must then outlive
  /// every repository that may hold a batch preload moved to a GPU.
  std::shared_ptr<preload_streams> streams;
  /// The pipeline executor's threads per GPU: the queue-depth gate and the pop-credit cap.
  int pipeline_threads_per_gpu{1};
  /// Capacity and downgrade stop fraction of each memory space. Required.
  space_capacity_fn space_capacity;
};

/// What the executor did since it was constructed.
struct preload_counters {
  std::uint64_t pops{0};              ///< task_popped events
  std::uint64_t promoted_to_gpu{0};   ///< HOST->GPU and DISK->GPU promotions
  std::uint64_t promoted_to_host{0};  ///< DISK->HOST stagings
  std::uint64_t bytes_to_gpu{0};
  std::uint64_t bytes_to_host{0};
  std::uint64_t gated{0};         ///< pops with candidates for the GPU but none past the gate
  std::uint64_t no_candidate{0};  ///< pops with no candidate for the GPU at all
  std::uint64_t no_room{0};       ///< nothing fit a budget or a ceiling (checked first), or a
                                  ///< reservation failed
  std::uint64_t not_needed{0};    ///< the batch was gone, busy or already promoted
  std::uint64_t failed{0};        ///< unsupported representation or an error during the copy
  std::uint64_t stale_event{0};   ///< task_downgraded delivered after its task was popped
};

/**
 * @brief Promotes spilled inputs of queued tasks, driven by two events.
 *
 * - @c task_downgraded: inputs of a queued task are off the GPU, moved there by the downgrade
 *   executor or already there when the scheduler queued the task; the task becomes a candidate.
 * - @c task_popped for GPU g: the task leaves the candidates and releases its budget, and g
 *   gets one credit. g's worker spends each credit on at most one promotion: the first input,
 *   in pop order, of a candidate that may run on g and passes the gate (g's queue is short, or
 *   the candidate belongs to the popped task's query and pipeline), moved to g if it fits g's
 *   budget and ceiling, else, from DISK, to g's local HOST space if it fits that space's budget
 *   and ceiling.
 *
 * Concurrency: one mutex guards all state. The event hooks run on the subscriber thread; each GPU
 * has one worker thread. A worker selects a job under the mutex and runs @ref promote_one
 * without it, since that copies data and takes batch locks.
 *
 * Lifetime: construct, @ref start, then @ref stop (also run by the destructor). The executor must
 * be destroyed before anything it borrows.
 */
class preload_executor final : public event::query_event_subscriber {
 public:
  /**
   * @param publisher Publisher of the two events; must be owned by a shared_ptr.
   * @param config Preload configuration.
   * @param deps Borrowed collaborators (see @ref preload_dependencies).
   * @throws std::invalid_argument if @c space_capacity is missing, or device promotion is
   *         configured without stream pools.
   */
  preload_executor(event::query_event_publisher& publisher,
                   preload_executor_config config,
                   preload_dependencies deps);
  ~preload_executor() override;

  preload_executor(preload_executor const&)            = delete;
  preload_executor& operator=(preload_executor const&) = delete;
  preload_executor(preload_executor&&)                 = delete;
  preload_executor& operator=(preload_executor&&)      = delete;

  /// Start one worker per GPU, then event delivery. Idempotent; a no-op after @ref stop.
  void start();

  /// Stop event delivery and join the workers; a running promotion finishes first. Idempotent and
  /// terminal.
  void stop() noexcept;

  /// Drop every candidate of @p q and release its budget. Never waits: a promotion of @p q that
  /// is still running releases its own budget when it ends.
  void retire_query(query_id_t q) noexcept;

  /// A snapshot of the counters.
  [[nodiscard]] preload_counters counters() const;

  [[nodiscard]] std::string_view name() const noexcept override { return "preload"; }

  // ---- tests only ------------------------------------------------------------------------------

  /// Event hooks completed so far.
  [[nodiscard]] std::uint64_t events_handled_for_testing() const;
  /// Wait until no GPU has a credit left or a promotion running. @return false on timeout.
  [[nodiscard]] bool wait_idle_for_testing(std::chrono::milliseconds timeout) const;
  /// Budget charged on space @p id (GPU or HOST): running promotions plus promoted inputs whose
  /// task has not been popped.
  [[nodiscard]] std::size_t outstanding_for_testing(cucascade::memory::memory_space_id id) const;
  [[nodiscard]] std::size_t candidate_count_for_testing() const;
  /// Run @p hook inside each promotion, on the worker, once the target reservation is made and
  /// before the batch is locked. Call before @ref start: the workers read it without a lock.
  void set_promotion_hook_for_testing(std::function<void(promotion_job const&)> hook);

 private:
  void on_task_downgraded(event::event_id_t event_id,
                          event::timestamp_t,
                          query_id_t query_id,
                          event::downgraded_task const& task) noexcept override;
  void on_task_popped(event::event_id_t event_id,
                      event::timestamp_t,
                      query_id_t query_id,
                      std::uint64_t task_id,
                      std::size_t pipeline_id,
                      int gpu,
                      std::size_t tasks_left) noexcept override;

  /// Budget held on one space; @c bytes 0 = none.
  struct charge {
    cucascade::memory::Tier tier{cucascade::memory::Tier::SIZE};
    std::size_t space{0};
    std::size_t bytes{0};
  };

  /// One spilled input of a candidate task.
  struct input_state {
    std::weak_ptr<cucascade::data_batch> batch;
    std::uint64_t batch_id{0};
    /// Last known tier: HOST or DISK from the event, GPU or HOST after a promotion.
    cucascade::memory::Tier location{cucascade::memory::Tier::SIZE};
    /// The budget and reservation estimate, slack included: the largest device size reported
    /// (the GPU size before a spill, or the peak to materialize it for an input queued off the
    /// GPU) with slack. Raised to the exact need of a promotion that found no room, so it is not
    /// retried until that fits.
    std::size_t est_bytes{0};
    /// A worker is promoting it; the job, not this input, holds that charge.
    bool in_flight{false};
    /// Its last promotion failed or found the batch busy; not retried until downgraded again.
    bool skip{false};
    /// Budget this input holds after a successful promotion.
    charge held{};
  };

  /// A queued task with spilled inputs.
  struct candidate_task {
    query_id_t query{};
    std::size_t pipeline_id{0};
    std::optional<int> preferred_device;
    std::vector<input_state> inputs;
  };

  /// Pop order: queue priority, then task id.
  using candidate_key = std::pair<exec::queue_priority, std::uint64_t>;
  using candidate_map = std::map<candidate_key, candidate_task>;

  /// One pop on a GPU, waiting for that GPU's worker.
  struct pop_credit {
    std::size_t tasks_left{0};
    /// The popped task's query and pipeline: pipeline ids are per query.
    query_id_t query{};
    std::size_t pipeline_id{0};
  };

  /// A promotion target and the budget charged on it.
  struct space_state {
    cucascade::memory::memory_space* space{nullptr};
    std::size_t budget_bytes{0};
    std::size_t ceiling_bytes{0};
    std::size_t outstanding{0};
  };

  struct gpu_state : space_state {
    int device{-1};
    /// Index of the HOST space DISK inputs are staged into when the GPU has no room.
    std::optional<std::size_t> local_host;
    std::deque<pop_credit> credits;
    /// The worker is running a promotion.
    bool busy{false};
  };

  void build_spaces(preload_dependencies const& deps);
  void worker_loop(std::size_t g, std::stop_token const& st) noexcept;
  /// Pick and charge one promotion for a pop on GPU @p g; prunes expired inputs on the way.
  [[nodiscard]] std::optional<promotion_job> select(std::size_t g, pop_credit const& credit);
  /// Settle the charge of a finished @p job against its input, if that is still a candidate.
  void apply(promotion_job const& job, promotion_result const& result) noexcept;
  /// The candidate of task @p task_id, added if new, at its place for @p priority.
  [[nodiscard]] candidate_task& upsert(std::uint64_t task_id, exec::queue_priority priority);
  /// The candidate of task @p task_id, or the end of @c _candidates.
  [[nodiscard]] candidate_map::iterator find_candidate(std::uint64_t task_id);
  /// Release the budget @p it holds and erase it. @return the next candidate.
  candidate_map::iterator drop(candidate_map::iterator it) noexcept;
  [[nodiscard]] bool fits(space_state const& s, std::size_t bytes) const;
  [[nodiscard]] space_state& state_of(charge const& c) noexcept;
  void add(charge const& c) noexcept;
  void release(charge& c) noexcept;
  void remember_pop(std::uint64_t task_id, event::event_id_t event_id);
  [[nodiscard]] bool idle() const noexcept;
  void log_unexpected(std::string_view where) noexcept;
  void join_workers() noexcept;

  preload_executor_config _cfg;
  exec::query_lifecycle_registry& _lifecycle;
  std::shared_ptr<preload_streams> _streams;
  /// The queue-depth gate, and the most credits a GPU keeps.
  std::size_t _pipeline_threads{1};
  /// Read by every promotion, without @c _mu.
  promotion_env _env;

  /// Serializes start() and stop(); never taken by hooks or workers.
  std::mutex _lifecycle_mu;
  enum class run_state : std::uint8_t { idle, running, stopped };
  run_state _state{run_state::idle};
  std::vector<std::jthread> _workers;

  /// Guards everything below.
  mutable std::mutex _mu;
  std::condition_variable_any _work_cv;
  mutable std::condition_variable_any _idle_cv;
  std::vector<gpu_state> _gpus;
  std::vector<space_state> _hosts;
  /**
   * Candidates in pop order; @c _candidate_keys holds the key of each by task id. Budget
   * invariant: a space's @c outstanding is the sum of the charges of running jobs targeting it
   * plus the @c held charges of inputs here. @ref select adds a job's charge; @ref apply releases
   * it or moves it, trued up, into @c held, releasing what was held before; a popped, retired,
   * re-downgraded or expired input releases its @c held.
   *
   * A task_downgraded delivered after its task's task_popped is dropped by @c _recent_pops; for
   * what that window misses, and for a task never popped because its query is tearing down,
   * @ref select drops the inputs whose batch expired, and candidates left empty, that it walks
   * past, and @ref retire_query drops the rest.
   */
  candidate_map _candidates;
  std::unordered_map<std::uint64_t, candidate_key> _candidate_keys;
  /// Recently popped tasks and the id of their task_popped event, oldest first in
  /// @c _recent_pop_order; bounded.
  std::unordered_map<std::uint64_t, event::event_id_t> _recent_pops;
  std::deque<std::uint64_t> _recent_pop_order;
  preload_counters _counters;
  std::uint64_t _events_handled{0};
  bool _unexpected_logged{false};
};

}  // namespace sirius::preload
