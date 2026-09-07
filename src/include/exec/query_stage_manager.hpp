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

#include "blockingconcurrentqueue.h"
#include "exec/queue_priority.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "query_id.hpp"

#include <pthread.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::exec {

/// Discriminator for @ref query_stage_event.  One entry per @c notify_* entry
/// point on @ref query_stage_manager; the enumerator is what the dispatch in
/// @ref query_stage_listener switches on, so the two must stay in step.
enum class query_stage_event_type : std::uint8_t {
  task_created,
  task_deployed,
  failed_to_create_task,
  task_queue_empty,
  pipeline_closed,
  executor_awaiting_task,
  memory_downgrade_for_task,
  wait_for_memory_for_task
};

/// One published event: the tag that says which it is, and the arguments the
/// reporter passed, kept together so the event can be queued and replayed on
/// another thread exactly as it was raised.
///
/// The tag is a template parameter rather than a member so every event is a
/// distinct type, which is what lets @ref system_events be a @c std::variant
/// and the dispatch be a compile-time @c if @c constexpr chain rather than a
/// switch that could silently fall through.
template <query_stage_event_type EventType, typename... Types>
struct query_stage_event {
  using param_type = std::tuple<Types...>;

  static constexpr query_stage_event_type event_type = EventType;

  param_type data;
};

using task_created_event = query_stage_event<query_stage_event_type::task_created,
                                             query_id_t,
                                             std::size_t,
                                             op::SiriusPhysicalOperatorType,
                                             queue_priority>;

using task_deployed_event = query_stage_event<query_stage_event_type::task_deployed,
                                              query_id_t,
                                              std::size_t,
                                              op::SiriusPhysicalOperatorType,
                                              int>;

using failed_to_create_task_event = query_stage_event<query_stage_event_type::failed_to_create_task,
                                                      query_id_t,
                                                      std::size_t,
                                                      std::size_t>;

using task_queue_empty_event = query_stage_event<query_stage_event_type::task_queue_empty>;

using pipeline_closed_event =
  query_stage_event<query_stage_event_type::pipeline_closed, query_id_t, std::size_t, std::size_t>;

using executor_awaiting_task_event =
  query_stage_event<query_stage_event_type::executor_awaiting_task, int>;

using memory_downgrade_for_task_event =
  query_stage_event<query_stage_event_type::memory_downgrade_for_task,
                    query_id_t,
                    std::size_t,
                    int,
                    std::size_t>;

using wait_for_memory_for_task_event =
  query_stage_event<query_stage_event_type::wait_for_memory_for_task,
                    query_id_t,
                    std::size_t,
                    int,
                    std::size_t>;

/// Every event a @ref query_stage_manager can publish.  Listeners receive this
/// and nothing else, so adding an event is: an enumerator, an alias, an arm
/// here, and an arm in the dispatch.
using system_events = std::variant<task_created_event,
                                   task_deployed_event,
                                   failed_to_create_task_event,
                                   task_queue_empty_event,
                                   pipeline_closed_event,
                                   executor_awaiting_task_event,
                                   memory_downgrade_for_task_event,
                                   wait_for_memory_for_task_event>;

/// Number of distinct events, and so the width of the manager's routing table.
inline constexpr std::size_t n_system_events = std::variant_size_v<system_events>;

/// An event's slot in that table. The variant arms are declared in enumerator
/// order, which is what lets the tag double as the index -- reorder one list
/// without the other and every event routes to the wrong subscribers.
template <typename Event>
inline constexpr std::size_t event_index_v = static_cast<std::size_t>(Event::event_type);

static_assert(
  std::is_same_v<std::variant_alternative_t<event_index_v<task_created_event>, system_events>,
                 task_created_event>);
static_assert(
  std::is_same_v<
    std::variant_alternative_t<event_index_v<wait_for_memory_for_task_event>, system_events>,
    wait_for_memory_for_task_event>);

/// Every event, for the @ref query_stage_manager::register_listener overload
/// that takes no subscription list.
inline constexpr std::array<query_stage_event_type, n_system_events> all_system_events{
  query_stage_event_type::task_created,
  query_stage_event_type::task_deployed,
  query_stage_event_type::failed_to_create_task,
  query_stage_event_type::task_queue_empty,
  query_stage_event_type::pipeline_closed,
  query_stage_event_type::executor_awaiting_task,
  query_stage_event_type::memory_downgrade_for_task,
  query_stage_event_type::wait_for_memory_for_task};

/// The mailbox a single listener is fed through.  Events travel as a
/// @c shared_ptr because one raised event goes to every subscriber: the payload
/// is built once and the queues carry references to it.  A null entry is the
/// close sentinel -- see @ref query_stage_manager::stop.
using system_event_queue =
  duckdb_moodycamel::BlockingConcurrentQueue<std::shared_ptr<system_events>>;

/// What @ref query_stage_manager::register_listener hands back: the queue the
/// listener drains, and the token that tells it when to stop draining.
struct listener_registration {
  std::shared_ptr<system_event_queue> queue;
  std::stop_token stop_token;
};

/**
 * @brief Observer of where a query is in its execution, assembled from the
 *        points at which work is created, dispatched, and runs out.
 *
 * The task creator and the task scheduler each see one half of the picture: the
 * creator knows what work exists and why it could not make more, the scheduler
 * knows what got dispatched and when a GPU went hungry.  Neither can say on its
 * own whether a query is scan-bound, waiting on a barrier, or simply done.
 * Reporting both halves here is what lets that be answered in one place.
 *
 * Reporters call the @c notify_* entry points, which build the event once and
 * hand a reference to it to every registered listener's queue.  Publishing is
 * therefore a push and never a callback: a reporter is never made to wait on
 * what a listener does with the event, which matters because these are raised
 * from the creator, scheduler and executor hot paths.  Nothing here is virtual:
 * the manager is the fixed relay and the listener is the extension point.
 *
 * Lifetime: constructed by SiriusContext into a @c shared_ptr and handed to its
 * reporters by reference, each of which extends it via @c shared_from_this ---
 * so a reporter's handle is never null and never dangles.  Queues are owned by
 * their listeners and only referenced here, so the manager never keeps a
 * listener alive.
 *
 * Thread safety: the queue set is guarded by a shared mutex.  Publishing takes
 * it shared, so the reporting threads do not serialise against each other;
 * @ref register_listener and @ref unregister_listener take it exclusively.
 */
class query_stage_manager : public std::enable_shared_from_this<query_stage_manager> {
 public:
  query_stage_manager()  = default;
  ~query_stage_manager() = default;

  query_stage_manager(query_stage_manager const&)            = delete;
  query_stage_manager& operator=(query_stage_manager const&) = delete;

  // -- registration ----------------------------------------------------------
  //
  // Components that want these events register rather than being reached into,
  // so the creator and scheduler keep one collaborator and know nothing about
  // who is listening.

  /// Mint a mailbox for one listener.  The returned queue is owned by the
  /// caller -- the manager holds only a reference to it -- and the returned
  /// token is the manager's, so every listener stops when the manager does.
  ///
  /// Registering after events have started flowing is safe; the listener simply
  /// misses what was published before it arrived.  Registering after @ref stop
  /// yields an already-stopped token and a queue nothing will ever be pushed
  /// to, so the listener finishes immediately rather than hanging.
  /// Mint a mailbox subscribed to every event.
  [[nodiscard]] listener_registration register_listener()
  {
    return register_listener(all_system_events);
  }

  /// Mint a mailbox subscribed to @p events and nothing else.
  ///
  /// Publishing walks one subscriber list per event, so a listener that names
  /// two events is skipped outright by the other six rather than being handed a
  /// payload it will drop. With a handful of listeners that is a rounding error;
  /// it is worth having because the cost grows with listeners x events while the
  /// useful work does not.
  ///
  /// The list is the subscription -- a callback the listener overrides but does
  /// not name here is never called. Keeping the two in step is the caller's job,
  /// which is the trade for @c override still catching a misspelt or drifted
  /// callback signature.
  [[nodiscard]] listener_registration register_listener(
    std::span<query_stage_event_type const> events)
  {
    auto queue = std::make_shared<system_event_queue>();
    {
      std::unique_lock g{_queues_mtx};
      if (!_stopped) {
        _queues.push_back(queue);
        for (auto e : events) {
          auto& bucket = _by_event[static_cast<std::size_t>(e)];
          // A repeated event would double-deliver, and a caller assembling the
          // list from overlapping sets should not have to care.
          if (std::ranges::find(bucket, queue.get()) == bucket.end()) {
            bucket.push_back(queue.get());
          }
        }
      }
    }
    return listener_registration{std::move(queue), _stop_source.get_token()};
  }

  [[nodiscard]] listener_registration register_listener(
    std::initializer_list<query_stage_event_type> events)
  {
    return register_listener(
      std::span<query_stage_event_type const>{events.begin(), events.size()});
  }

  /// Drop @p queue's registration.  Blocks until any in-flight publish has
  /// finished, so no further event reaches @p queue once this returns.  Safe to
  /// call with a queue that was never registered, or twice.
  void unregister_listener(std::shared_ptr<system_event_queue> const& queue) noexcept
  {
    std::unique_lock g{_queues_mtx};
    std::erase_if(_queues, [&queue](auto const& q) { return q == queue; });
    // The buckets hold raw pointers into what _queues owns, so they have to be
    // cleared in the same critical section: a pointer left behind would outlive
    // the queue it names.
    auto* raw = queue.get();
    for (auto& bucket : _by_event) {
      std::erase(bucket, raw);
    }
  }

  /// Stop every listener and close the manager to further publishing.  Requests
  /// stop on the shared token and pushes the null sentinel to each queue, which
  /// is what wakes a listener parked on an empty mailbox; subsequent @c notify_*
  /// calls are no-ops.  Does not join the listeners: a listener owns its own
  /// thread and joins it in its own destructor.
  void stop() noexcept
  {
    _stop_source.request_stop();
    std::vector<std::shared_ptr<system_event_queue>> queues;
    {
      std::unique_lock g{_queues_mtx};
      _stopped = true;
      queues.swap(_queues);
      for (auto& bucket : _by_event) {
        bucket.clear();
      }
    }
    for (auto const& q : queues) {
      // try_ so this stays genuinely noexcept: the allocating enqueue could
      // throw, and a listener that misses the sentinel still comes out on the
      // stop token within a poll interval.  The sentinel only buys promptness.
      std::ignore = q->try_enqueue(nullptr);
    }
  }

  // -- reporting -------------------------------------------------------------

  void notify_task_created(query_id_t query_id,
                           std::size_t operator_id,
                           op::SiriusPhysicalOperatorType operator_type,
                           queue_priority priority) noexcept
  {
    publish(task_created_event{{query_id, operator_id, operator_type, priority}});
  }

  void notify_task_deployed(query_id_t query_id,
                            std::size_t operator_id,
                            op::SiriusPhysicalOperatorType operator_type,
                            int gpu_id) noexcept
  {
    publish(task_deployed_event{{query_id, operator_id, operator_type, gpu_id}});
  }

  void notify_failed_to_create_task(query_id_t query_id,
                                    std::size_t source_operator_id,
                                    std::size_t failed_operator_id) noexcept
  {
    publish(failed_to_create_task_event{{query_id, source_operator_id, failed_operator_id}});
  }

  void notify_task_queue_empty() noexcept { publish(task_queue_empty_event{}); }

  void notify_pipeline_closed(query_id_t query_id,
                              std::size_t pipeline_id,
                              std::size_t source_operator_id) noexcept
  {
    publish(pipeline_closed_event{{query_id, pipeline_id, source_operator_id}});
  }

  void notify_executor_awaiting_task(int gpu_id) noexcept
  {
    publish(executor_awaiting_task_event{{gpu_id}});
  }

  void notify_memory_downgrade_for_task(query_id_t query_id,
                                        std::size_t operator_id,
                                        int gpu_id,
                                        std::size_t shortfall_bytes) noexcept
  {
    publish(memory_downgrade_for_task_event{{query_id, operator_id, gpu_id, shortfall_bytes}});
  }

  void notify_wait_for_memory_for_task(query_id_t query_id,
                                       std::size_t operator_id,
                                       int gpu_id,
                                       std::size_t bytes_needed) noexcept
  {
    publish(wait_for_memory_for_task_event{{query_id, operator_id, gpu_id, bytes_needed}});
  }

 private:
  /// Build the event once and fan a reference to it out to every mailbox.
  ///
  /// The empty check ahead of the allocation is not just an optimisation: with
  /// nobody listening these entry points sit on hot paths and should cost a
  /// lock and a branch, not a heap allocation.  Publishing is @c noexcept, so a
  /// failed allocation drops the event rather than propagating out into a
  /// reporter that has no way to handle it.
  template <typename Event>
  void publish(Event&& event) noexcept
  {
    try {
      std::shared_lock g{_queues_mtx};
      auto const& subscribers = _by_event[event_index_v<std::decay_t<Event>>];
      if (subscribers.empty()) { return; }
      auto payload = std::make_shared<system_events>(std::forward<Event>(event));
      for (auto* q : subscribers) {
        q->enqueue(payload);
      }
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Telemetry is not worth failing execution over.
    }
  }

  /// Shared by every listener: one request_stop takes them all down together.
  std::stop_source _stop_source;

  mutable std::shared_mutex _queues_mtx;
  /// Owns the registered queues; the buckets below only point into it.
  std::vector<std::shared_ptr<system_event_queue>> _queues;
  /// One subscriber list per event, indexed by @ref event_index_v. This is what
  /// makes a notify cost only the listeners that asked for that event.
  std::array<std::vector<system_event_queue*>, n_system_events> _by_event;
  bool _stopped{false};
};

/// Always-false, but dependent on its argument, so the exhaustiveness
/// @c static_assert in the dispatch only fires for a tag with no arm.
template <query_stage_event_type>
inline constexpr bool unhandled_event_type = false;

/**
 * @brief Receives the execution-stage events a @ref query_stage_manager
 *        publishes, on a thread of its own.
 *
 * The listener registers a mailbox with the manager at construction and drains
 * it from a single worker started by @ref start.  That indirection is the
 * point: the hooks below run on the listener's thread, not on the creator,
 * scheduler or executor thread that raised the event, so an implementation is
 * free to do real work in them and cannot stall execution by doing so.
 *
 * The flip side is that a hook is a report of something that already happened
 * and may no longer be true by the time it is read.  Implementations should
 * treat the arguments as a snapshot rather than as live state.
 *
 * Thread safety: hooks are called from the listener's worker and nowhere else,
 * so they are serialised against each other and see events in publication
 * order.  They still race against the implementation's own public API, which is
 * called from elsewhere.
 *
 * Subclassing: the worker dispatches into virtual hooks, so it must be stopped
 * before the derived part of the object is destroyed.  A derived class whose
 * destructor can run while events are in flight must call @ref stop itself ---
 * the base destructor is too late, the derived members are already gone by
 * then.
 */
class query_stage_listener {
 public:
  /// Registers a mailbox for every event; no thread runs until @ref start.
  explicit query_stage_listener(query_stage_manager& manager)
    : query_stage_listener(manager, all_system_events)
  {
  }

  /// Registers a mailbox for @p events only.
  ///
  /// Name the events whose hooks you override and no others: the manager skips
  /// this listener entirely when publishing anything else, so an overridden hook
  /// left out of the list is simply never called. That is the cost of keeping
  /// the hooks virtual -- @c override still catches a misspelt or drifted
  /// signature, which a callback-detecting scheme cannot.
  query_stage_listener(query_stage_manager& manager, std::span<query_stage_event_type const> events)
    : _manager(manager.weak_from_this()), _events(events.begin(), events.end())
  {
    // Straight off the reference rather than through the weak handle: a manager
    // not owned by a shared_ptr has no weak_from_this, and a listener that
    // silently registered with nothing would be a very quiet bug.
    adopt(manager.register_listener(events));
  }

  query_stage_listener(query_stage_manager& manager,
                       std::initializer_list<query_stage_event_type> events)
    : query_stage_listener(manager,
                           std::span<query_stage_event_type const>{events.begin(), events.size()})
  {
  }

  virtual ~query_stage_listener() { stop(); }

  query_stage_listener(query_stage_listener const&)            = delete;
  query_stage_listener& operator=(query_stage_listener const&) = delete;

  /// Start draining the mailbox.  Idempotent: a second call while the worker
  /// runs does nothing.  A no-op once the manager has stopped.
  ///
  /// Restarting after @ref stop works and takes a fresh mailbox: the old one
  /// was handed back so that a stopped listener costs the publishers nothing,
  /// which also means events raised while it was stopped are gone rather than
  /// backed up waiting.
  void start()
  {
    if (_worker.joinable()) { return; }
    if (!_registered) { register_mailbox(); }
    if (_queue == nullptr || _stop_token.stop_requested()) { return; }
    _draining.store(true, std::memory_order_relaxed);
    // Armed here rather than in the constructor so a listener that is never
    // started never gets the callback -- and so @ref on_stop_requested cannot
    // fire before the derived object is fully built.
    _stop_cb.emplace(_stop_token, [this] { on_stop_requested(); });
    _worker = std::jthread([this] { run(); });
    set_thread_name();
  }

  /// Stop this listener's worker and join it.  Safe to call when not started,
  /// and safe to call twice.  Does not stop the manager: other listeners keep
  /// running.
  ///
  /// The mailbox is handed back first, so nothing further is published to a
  /// queue that has no worker left to drain it.  That matters for the per-query
  /// listeners: one that is stopped but still held elsewhere would otherwise go
  /// on accumulating the whole context's events, unread.
  void stop() noexcept
  {
    unregister_mailbox();
    if (!_worker.joinable()) {
      _stop_cb.reset();
      return;
    }
    // The sentinel is what wakes a worker parked on an empty mailbox; the flag
    // is what stops it from working through whatever is queued behind it.
    _draining.store(false, std::memory_order_relaxed);
    std::ignore = _queue->try_enqueue(nullptr);  // see query_stage_manager::stop
    _worker.join();
    _stop_cb.reset();
  }

  /// Whether the worker is up.  Named apart from any @c is_running a subclass
  /// has for its own work, which is a different question.
  [[nodiscard]] bool is_listening() const noexcept { return _worker.joinable(); }

  /// Name for logs and for the worker's thread name, which is
  /// @c "listener-<name>" truncated to what pthread accepts.
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;

  /// A task has been created for @p operator_id and handed to the scheduler.
  virtual void on_task_created(query_id_t query_id,
                               std::size_t operator_id,
                               op::SiriusPhysicalOperatorType operator_type,
                               queue_priority priority) noexcept
  {
  }

  /// A task has been popped from the queue and pushed to the executor for
  /// @p gpu_id -- the point at which queued work becomes running work.
  virtual void on_task_deployed(query_id_t query_id,
                                std::size_t operator_id,
                                op::SiriusPhysicalOperatorType operator_type,
                                int gpu_id) noexcept
  {
  }

  /// No task was created for @p source_operator_id: the walk from it found
  /// nobody able to produce.  Distinguishes "nothing to do yet" from "nothing
  /// left to do".
  ///
  /// @p failed_operator_id is where the walk actually stopped.  It differs from
  /// @p source_operator_id whenever the walk descended through operators that
  /// were themselves waiting on input, and it is the one that says where the
  /// pipeline is stuck -- the source may simply be waiting on it.  The two are
  /// equal when the source itself could not produce.
  virtual void on_failed_to_create_task(query_id_t query_id,
                                        std::size_t source_operator_id,
                                        std::size_t failed_operator_id) noexcept
  {
  }

  /// The scheduler found its queue empty.  Says nothing about whether more work
  /// is coming -- pair with @c on_failed_to_create_task to tell those apart.
  virtual void on_task_queue_empty() noexcept {}

  /// A pipeline reached its closed state, so its source operator will produce
  /// no further tasks.
  virtual void on_pipeline_closed(query_id_t query_id,
                                  std::size_t pipeline_id,
                                  std::size_t source_operator_id) noexcept
  {
  }

  /// An executor asked for work for @p gpu_id and the scheduler had tasks but
  /// none it could send there.  Not the same as an empty queue: this is work
  /// existing but being unplaceable, i.e. a GPU idling against a non-empty
  /// queue.
  virtual void on_executor_awaiting_task(int gpu_id) noexcept {}

  /// An executor could not reserve the memory the task from @p operator_id
  /// needs on @p gpu_id, and is spilling to free @p shortfall_bytes before it
  /// can run.  That task is parked for as long as that takes, so the GPU is
  /// about to do no work at all -- which makes it the one moment the device's
  /// IO path is unambiguously free.
  virtual void on_memory_downgrade_for_task(query_id_t query_id,
                                            std::size_t operator_id,
                                            int gpu_id,
                                            std::size_t shortfall_bytes) noexcept
  {
  }

  /// An executor could not reserve @p bytes_needed for the task from
  /// @p operator_id on @p gpu_id and is about to block until the memory frees
  /// up.  Distinct from @ref on_memory_downgrade_for_task, which is the executor
  /// actively spilling to make room: here it is simply waiting on someone else
  /// to release.  Either way the task is parked and the GPU is about to do no
  /// work, which is what makes it worth reporting.
  ///
  /// Raised BEFORE the blocking call, because a listener told only once the wait
  /// is over learns nothing it can act on.
  virtual void on_wait_for_memory_for_task(query_id_t query_id,
                                           std::size_t operator_id,
                                           int gpu_id,
                                           std::size_t bytes_needed) noexcept
  {
  }

 protected:
  /// Invoked once when the manager requests stop, on the thread that called
  /// @ref query_stage_manager::stop -- not on the worker, which may still be
  /// finishing an event.  The hook for tearing down whatever the listener was
  /// driving; the worker's own exit needs no help.
  virtual void on_stop_requested() noexcept {}

 private:
  /// Take a mailbox and the manager's stop token.  A manager that has already
  /// gone away leaves an unregistered listener holding a queue nobody writes
  /// to, which is exactly what a listener with nothing to hear should be.
  void register_mailbox() noexcept
  {
    // Same subscription as the first time: a restarted listener must not quietly
    // widen to every event.
    if (auto manager = _manager.lock()) { adopt(manager->register_listener(_events)); }
  }

  void adopt(listener_registration reg) noexcept
  {
    _queue      = std::move(reg.queue);
    _stop_token = std::move(reg.stop_token);
    _registered = true;
  }

  void unregister_mailbox() noexcept
  {
    if (!_registered) { return; }
    if (auto manager = _manager.lock()) { manager->unregister_listener(_queue); }
    _registered = false;
  }

  /// Drain until stopped, replaying each event into its hook.
  ///
  /// The timed wait rather than a plain blocking one is what makes the token a
  /// real stop signal: a manager that goes away without pushing the sentinel
  /// still gets the worker out within the poll interval, instead of leaving a
  /// thread parked forever on a queue nobody will write to again.
  void run() noexcept
  {
    using namespace std::chrono_literals;
    constexpr auto poll_interval = 100ms;

    while (_draining.load(std::memory_order_relaxed) && !_stop_token.stop_requested()) {
      std::shared_ptr<system_events> event;
      if (!_queue->wait_dequeue_timed(event, poll_interval)) { continue; }
      if (event == nullptr) { break; }  // close sentinel
      dispatch(*event);
    }
  }

  /// Replay one event into the hook that matches its tag.  The @c if
  /// @c constexpr chain is exhaustive over @ref query_stage_event_type, so a new
  /// event that forgets an arm here fails to compile rather than being dropped.
  void dispatch(system_events const& event) noexcept
  {
    std::visit(
      [this](auto const& e) {
        constexpr auto type = std::decay_t<decltype(e)>::event_type;
        std::apply(
          [this](auto const&... args) {
            if constexpr (type == query_stage_event_type::task_created) {
              on_task_created(args...);
            } else if constexpr (type == query_stage_event_type::task_deployed) {
              on_task_deployed(args...);
            } else if constexpr (type == query_stage_event_type::failed_to_create_task) {
              on_failed_to_create_task(args...);
            } else if constexpr (type == query_stage_event_type::task_queue_empty) {
              on_task_queue_empty(args...);
            } else if constexpr (type == query_stage_event_type::pipeline_closed) {
              on_pipeline_closed(args...);
            } else if constexpr (type == query_stage_event_type::executor_awaiting_task) {
              on_executor_awaiting_task(args...);
            } else if constexpr (type == query_stage_event_type::memory_downgrade_for_task) {
              on_memory_downgrade_for_task(args...);
            } else if constexpr (type == query_stage_event_type::wait_for_memory_for_task) {
              on_wait_for_memory_for_task(args...);
            } else {
              static_assert(unhandled_event_type<type>, "unhandled query_stage_event_type");
            }
          },
          e.data);
      },
      event);
  }

  void set_thread_name() noexcept
  {
    // pthread caps thread names at 16 bytes including the terminator, and
    // silently keeps the old name when given more -- so truncate rather than
    // hand it something it will reject.
    constexpr std::size_t max_len = 15;
    std::string thread_name       = "listener-";
    thread_name.append(name());
    if (thread_name.size() > max_len) { thread_name.resize(max_len); }
    pthread_setname_np(_worker.native_handle(), thread_name.c_str());
  }

  /// Weak so a listener never keeps the manager alive; used only to deregister.
  std::weak_ptr<query_stage_manager> _manager;
  /// The events this listener subscribed to, kept so a restart re-subscribes to
  /// the same set rather than to all of them.
  std::vector<query_stage_event_type> _events;
  std::shared_ptr<system_event_queue> _queue;
  std::stop_token _stop_token;
  /// Whether @c _queue is still in the manager's fan-out set.
  bool _registered{false};
  std::optional<std::stop_callback<std::function<void()>>> _stop_cb;
  /// Cleared by @ref stop so the worker leaves without draining the backlog.
  std::atomic<bool> _draining{true};
  std::jthread _worker;
};

}  // namespace sirius::exec
