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

#include "log/logging.hpp"
#include "query_id.hpp"

#include <absl/functional/any_invocable.h>

#include <condition_variable>
#include <latch>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::exec {

/**
 * @brief A thread pool with bounded concurrency.
 *
 * Merges a fixed-size thread pool with a counting semaphore, so admission and execution
 * live in one object. Usage:
 *
 *  1. reserve() — blocks when at capacity, returns a slot (invalid if interrupted).
 *  2. dispatch(slot&&, fn) — consumes the slot and enqueues fn on a worker thread.
 *     Dropping the slot without calling dispatch() releases it back immediately.
 *
 * Lifecycle:
 *  - interrupt(): wake all blocked reserve() calls without disturbing in-flight work.
 *                 Paired with resume() for drain-and-restart patterns.
 *  - resume():    re-enable after interrupt().
 *  - wait_all():  block until all in-flight tasks complete.
 *  - stop():      interrupt + join all worker threads (called automatically by destructor).
 */
class bounded_thread_pool {
 public:
  /**
   * @brief RAII handle representing a reserved execution slot.
   *
   * Obtained via bounded_thread_pool::reserve(). Either bounded_thread_pool::dispatch()
   * is called with this slot to submit work, or the slot goes out of scope without a
   * dispatch (which releases the slot back to the pool immediately without running work).
   */
  class slot {
   public:
    explicit slot(bounded_thread_pool* pool = nullptr) noexcept : pool_(pool) {}

    slot(slot&& other) noexcept : pool_(other.pool_), query_(other.query_)
    {
      other.pool_ = nullptr;
      other.query_.reset();
    }

    slot& operator=(slot&& other) noexcept
    {
      if (this != &other) {
        release();
        pool_       = other.pool_;
        query_      = other.query_;
        other.pool_ = nullptr;
        other.query_.reset();
      }
      return *this;
    }

    ~slot() { release(); }

    slot(const slot&)            = delete;
    slot& operator=(const slot&) = delete;

    [[nodiscard]] bool is_valid() const noexcept { return pool_ != nullptr; }
    explicit operator bool() const noexcept { return is_valid(); }

    /**
     * @brief Attribute this slot to @p query_id, so drain_and_wait(query_id) waits for it.
     *
     * Deliberately separate from reserve(). Every manager loop in this codebase reserves a slot
     * and THEN blocks waiting for a task, so the query is not known at reserve() time — and an
     * idle manager must not be counted against any query, or drain_and_wait() would block until
     * that manager happened to receive work. Attribution therefore happens once the task has been
     * popped and its query is known.
     *
     * Idempotent; a slot can only be attached once.
     */
    void attach(sirius::query_id_t query_id)
    {
      if (pool_ == nullptr || query_.has_value()) { return; }
      query_ = query_id;
      pool_->attach_slot(query_id);
    }

    /// \brief The query this slot is attributed to, or nullopt while untagged.
    [[nodiscard]] std::optional<sirius::query_id_t> query() const noexcept { return query_; }

   private:
    void release()
    {
      if (auto* p = std::exchange(pool_, nullptr); p != nullptr) {
        const auto q = query_;
        query_.reset();
        p->release_slot(q);
      }
    }

    bounded_thread_pool* pool_{nullptr};
    std::optional<sirius::query_id_t> query_{};
  };

  explicit bounded_thread_pool(int capacity,
                               const std::string& name                             = "btp",
                               std::vector<int> cpu_ids                            = {},
                               absl::AnyInvocable<void() noexcept> per_thread_init = nullptr)
    : capacity_(capacity)
  {
    threads_.reserve(capacity);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int id : cpu_ids) {
      CPU_SET(id, &cpuset);
    }

    std::unique_ptr<std::latch> init_latch;
    if (per_thread_init) { init_latch = std::make_unique<std::latch>(capacity); }

    auto* init_fn_ptr = per_thread_init ? &per_thread_init : nullptr;
    auto* latch_ptr   = init_latch.get();

    for (int i = 0; i < capacity; ++i) {
      auto& t = threads_.emplace_back([this, init_fn_ptr, latch_ptr]() {
        if (init_fn_ptr) {
          (*init_fn_ptr)();
          latch_ptr->count_down();
        }
        work_loop();
      });
      if (!name.empty()) {
        pthread_setname_np(t.native_handle(), (name + "_" + std::to_string(i)).c_str());
      }
      if (!cpu_ids.empty()) {
        pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
      }
    }

    if (init_latch) { init_latch->wait(); }
  }

  ~bounded_thread_pool() { stop(); }

  bounded_thread_pool(const bounded_thread_pool&)            = delete;
  bounded_thread_pool& operator=(const bounded_thread_pool&) = delete;

  /**
   * @brief Reserve a slot — blocks when at capacity.
   *
   * Returns an invalid slot if interrupted or stopped.
   */
  [[nodiscard]] slot reserve()
  {
    std::unique_lock lock(mu_);
    cv_capacity_.wait(lock, [&] { return active_ < capacity_ || interrupted_ || stop_requested_; });
    if (interrupted_ || stop_requested_) { return slot{}; }
    ++active_;
    return slot{this};
  }

  /**
   * @brief Interrupt all blocked reserve() calls.
   *
   * In-flight tasks continue to completion. Call resume() to re-enable.
   */
  void interrupt()
  {
    std::lock_guard lock(mu_);
    interrupted_ = true;
    cv_capacity_.notify_all();
  }

  /**
   * @brief Re-enable scheduling after interrupt().
   */
  void resume()
  {
    std::lock_guard lock(mu_);
    interrupted_ = false;
  }

  /**
   * @brief Block until all active slots have been released (all in-flight tasks done).
   */
  void wait_all()
  {
    std::unique_lock lock(mu_);
    cv_idle_.wait(lock, [&] { return active_ == 0; });
  }

  /**
   * @brief Stop worker threads. Interrupts pending calls and joins all threads.
   */
  void stop() noexcept
  {
    {
      std::lock_guard lock(mu_);
      if (stop_requested_) { return; }
      stop_requested_ = true;
      interrupted_    = true;
      cv_capacity_.notify_all();
      cv_work_.notify_all();
    }
    for (auto& t : threads_) {
      if (t.joinable()) { t.join(); }
    }
  }

  /**
   * @brief Dispatch a function using the given slot.
   *
   * Consumes the slot (it becomes invalid). The function runs on a worker thread;
   * the slot is released automatically when the task completes.
   */
  void dispatch(slot&& s, absl::AnyInvocable<void()> fn)
  {
    if (not s) { return; }
    const auto query = s.query();
    {
      std::lock_guard lock(mu_);
      work_queue_.push_back(
        work_item{query, [s = std::move(s), fn = std::move(fn)]() mutable noexcept {
                    try {
                      fn();
                    } catch (const std::exception& e) {
                      SIRIUS_LOG_ERROR("Exception in bounded_thread_pool task: {}", e.what());
                    } catch (...) {
                      SIRIUS_LOG_ERROR("Unknown exception in bounded_thread_pool task");
                    }
                    // Destroy before slot is released upon lambda exit.
                    fn = nullptr;
                    // When slot, s, goes out of scope, release_slot is automatically
                    // invoked, clearing the path for another task to pick up that slot.
                  }});
    }
    cv_work_.notify_one();
  }

  /**
   * @brief Wait until @p query_id has no work left — queued or running — without dropping any.
   *
   * A slot is attached before it is dispatched, so the per-query count covers queued-but-unstarted
   * work as well as work in progress; waiting for it to reach zero therefore means every task of
   * this query has actually RUN. That is what the success path needs: dropping there would
   * silently discard work the query legitimately scheduled.
   *
   * Untagged slots (an idle manager holding a reservation) are ignored, so this returns while
   * other queries — and the manager itself — carry on. wait_all() cannot: it waits for
   * active_ == 0, which a parked manager makes unreachable.
   */
  void wait_for_query(sirius::query_id_t query_id)
  {
    std::unique_lock lock(mu_);
    cv_query_idle_.wait(lock, [&] {
      auto it = active_by_query_.find(query_id);
      return it == active_by_query_.end() || it->second == 0;
    });
  }

  /**
   * @brief Wait until @p query_id has no active slots AND no untagged slot remains active.
   *
   * The error-path bracket's wait. Attribution happens at attach() time — after a task is
   * popped and its query is known — so a task dispatched WITHOUT an attach (one with no
   * pipeline, whose query cannot be determined) runs under an untagged slot. Such a task could
   * in principle belong to the failing query, so the bracket waits for it conservatively; a
   * co-tenant's task, by contrast, runs under a slot attached to the CO-TENANT's query and is
   * ignored — its (possibly long) memory wait no longer extends this query's cleanup the way
   * wait_all() did.
   *
   * Callers must first ensure no new untagged slots can appear (the bracket joins the manager
   * thread — the only reserve() caller — before waiting); otherwise a manager parked in pop()
   * holds an untagged slot indefinitely and this never returns.
   */
  void wait_for_query_and_untagged(sirius::query_id_t query_id)
  {
    std::unique_lock lock(mu_);
    cv_query_idle_.wait(lock, [&] {
      if (active_ - attached_active_ > 0) { return false; }
      auto it = active_by_query_.find(query_id);
      return it == active_by_query_.end() || it->second == 0;
    });
  }

  /// \brief Number of active slots not attributed to any query. Test/diagnostic aid.
  [[nodiscard]] int active_untagged()
  {
    std::lock_guard lock(mu_);
    return active_ - attached_active_;
  }

  /**
   * @brief Drop @p query_id's queued work and wait for its running work to finish.
   *
   * The per-query counterpart of wait_all(). Untagged slots — notably a manager thread parked in
   * its own queue's pop() while holding a reservation — are ignored, which is precisely why this
   * can return while other queries keep running. wait_all() cannot: it waits for active_ == 0,
   * which an idle manager makes unreachable.
   *
   * Queued items are moved out under the lock and destroyed after releasing it: destroying a work
   * item runs ~slot, which re-enters release_slot() and would deadlock on a held mutex.
   */
  void drain_and_wait(sirius::query_id_t query_id)
  {
    std::list<work_item> dropped;
    {
      std::lock_guard lock(mu_);
      for (auto it = work_queue_.begin(); it != work_queue_.end();) {
        if (it->query == query_id) {
          auto next = std::next(it);
          dropped.splice(dropped.end(), work_queue_, it);
          it = next;
        } else {
          ++it;
        }
      }
    }
    dropped.clear();  // releases the dropped items' slots, outside the lock

    std::unique_lock lock(mu_);
    cv_query_idle_.wait(lock, [&] {
      auto it = active_by_query_.find(query_id);
      return it == active_by_query_.end() || it->second == 0;
    });
  }

  /// \brief Number of running slots attributed to @p query_id. Test/diagnostic aid.
  [[nodiscard]] int active_for_query(sirius::query_id_t query_id)
  {
    std::lock_guard lock(mu_);
    auto it = active_by_query_.find(query_id);
    return it == active_by_query_.end() ? 0 : it->second;
  }

 private:
  /// One unit of dispatched work plus the query it is attributed to (nullopt when untagged).
  struct work_item {
    std::optional<sirius::query_id_t> query;
    absl::AnyInvocable<void() noexcept> fn;
  };

  // Called by slot::attach() once the query behind a reservation is known. Converting an
  // untagged slot to a tagged one can unblock wait_for_query_and_untagged (its untagged count
  // just dropped), hence the notify.
  void attach_slot(sirius::query_id_t query_id)
  {
    {
      std::lock_guard lock(mu_);
      ++active_by_query_[query_id];
      ++attached_active_;
    }
    cv_query_idle_.notify_all();
  }

  // Called exclusively by the slot destructor — covers both the drop-without-dispatch
  // and post-task-completion cases. @p query is the slot's attribution, if it had one.
  void release_slot(std::optional<sirius::query_id_t> query)
  {
    bool query_idle        = false;
    bool untagged_released = false;
    {
      std::lock_guard lock(mu_);
      --active_;
      if (query.has_value()) {
        --attached_active_;
        auto it = active_by_query_.find(*query);
        if (it != active_by_query_.end() && --it->second <= 0) {
          active_by_query_.erase(it);
          query_idle = true;
        }
      } else {
        untagged_released = true;
      }
    }
    if (query_idle || untagged_released) { cv_query_idle_.notify_all(); }
    cv_capacity_.notify_one();
    cv_idle_.notify_all();
  }

  void work_loop()
  {
    while (true) {
      absl::AnyInvocable<void() noexcept> fn;
      {
        std::unique_lock lock(mu_);
        cv_work_.wait(lock, [&] { return !work_queue_.empty() || stop_requested_; });
        if (stop_requested_ && work_queue_.empty()) { break; }
        fn = std::move(work_queue_.front().fn);
        work_queue_.pop_front();
      }
      if (fn == nullptr) { break; }
      fn();
    }
  }

  std::mutex mu_;
  std::condition_variable cv_capacity_;    // reserve() waits here when at capacity
  std::condition_variable cv_idle_;        // wait_all() waits here
  std::condition_variable cv_work_;        // worker threads wait here for work
  std::condition_variable cv_query_idle_;  // drain_and_wait(query_id) waits here

  int active_{0};
  /// Active slots currently attributed to a query (== the sum of active_by_query_ values).
  /// active_ - attached_active_ is the untagged count wait_for_query_and_untagged waits on.
  int attached_active_{0};
  const int capacity_;
  bool interrupted_{false};
  bool stop_requested_{false};

  /// std::list, not std::queue: drain_and_wait(query_id) must remove one query's items from the
  /// middle without disturbing the ordering of everyone else's.
  std::list<work_item> work_queue_;
  /// Running slots per query. Absent key == zero; untagged slots are not represented at all.
  std::map<sirius::query_id_t, int> active_by_query_;
  std::vector<std::thread> threads_;
};

}  // namespace sirius::exec
