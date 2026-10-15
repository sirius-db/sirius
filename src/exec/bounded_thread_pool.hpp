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

#include "exec/invocable.hpp"
#include "exec/thread_util.hpp"
#include "log/logging.hpp"
#include "query_id.hpp"

#include <condition_variable>
#include <latch>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace sirius::exec {

/**
 * @brief A thread pool with bounded concurrency.
 *
 * Merges a fixed-size thread pool with a counting semaphore, eliminating the need for
 * a separate kiosk object. Usage:
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

    slot(slot&& other) noexcept : pool_(std::exchange(other.pool_, nullptr)) {}

    slot& operator=(slot&& other) noexcept
    {
      if (this != &other) {
        release();
        pool_ = std::exchange(other.pool_, nullptr);
      }
      return *this;
    }

    ~slot() { release(); }

    slot(const slot&)            = delete;
    slot& operator=(const slot&) = delete;

    [[nodiscard]] bool is_valid() const noexcept { return pool_ != nullptr; }
    explicit operator bool() const noexcept { return is_valid(); }

   private:
    void release()
    {
      if (auto* p = std::exchange(pool_, nullptr)) { p->release_slot(); }
    }

    bounded_thread_pool* pool_{nullptr};
  };

  explicit bounded_thread_pool(int capacity,
                               const std::string& name                                  = "btp",
                               std::vector<int> cpu_ids                                 = {},
                               sirius::exec::invocable<void() noexcept> per_thread_init = nullptr)
    : capacity_(capacity)
  {
    threads_.reserve(capacity);

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
        std::ignore = sirius::exec::thread_util::set_thread_name(t, name + "_" + std::to_string(i));
      }
      if (!cpu_ids.empty()) {
        std::ignore = sirius::exec::thread_util::set_thread_affinity(t, cpu_ids);
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
   * Consumes the slot and callable only after queue allocation succeeds. On exception both
   * remain with the caller, which can report failure before destroying captured tasks.
   * The slot is released after callable destruction. The optional query tag is for queue disposal;
   * the caller owns query resource lifetime accounting.
   */
  void dispatch(slot&& s,
                sirius::exec::invocable<void()>&& fn,
                std::optional<sirius::query_id_t> query = std::nullopt)
  {
    if (not s) { return; }
    {
      std::lock_guard lock(mu_);
      auto& item = work_queue_.emplace_back();
      // Commit with nonthrowing transfers only after allocation. Workers cannot see the empty
      // entry while mu_ is held. No user captures are destroyed under the queue lock.
      item.query       = query;
      item.reservation = std::move(s);
      item.fn          = std::move(fn);
    }
    cv_work_.notify_one();
  }

  /// Dispose of queued callables only. Manager-held/running work is covered by its work lease;
  /// callers must close publication before disposal and wait for leases before releasing state.
  /// A manager can still dispatch already-held work after this sweep; disposal is not a barrier.
  /// Detach under the lock and destroy outside it: captures and slots may reenter the pool.
  void discard_queued(sirius::query_id_t query_id)
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
    dropped.clear();  // releases captures and slots outside the lock
  }

 private:
  /// Query identity is queue metadata only. Idle reservations carry no query or work claim.
  struct work_item {
    // Reverse member destruction retires captures before returning execution capacity.
    slot reservation;
    std::optional<sirius::query_id_t> query;
    sirius::exec::invocable<void()> fn;
  };

  // Capacity covers both undispatched reservations and queued/running worker callables.
  void release_slot()
  {
    {
      std::lock_guard lock(mu_);
      --active_;
    }
    cv_capacity_.notify_one();
    cv_idle_.notify_all();
  }

  void work_loop()
  {
    while (true) {
      slot reservation;
      sirius::exec::invocable<void()> fn;
      {
        std::unique_lock lock(mu_);
        cv_work_.wait(lock, [&] { return !work_queue_.empty() || stop_requested_; });
        if (stop_requested_ && work_queue_.empty()) { break; }
        reservation = std::move(work_queue_.front().reservation);
        fn          = std::move(work_queue_.front().fn);
        work_queue_.pop_front();
      }
      if (fn == nullptr) { break; }
      try {
        fn();
      } catch (const std::exception& e) {
        log::log_noexcept(
          [&] { SIRIUS_LOG_ERROR("Exception in bounded_thread_pool task: {}", e.what()); });
      } catch (...) {
        log::log_noexcept(
          [] { SIRIUS_LOG_ERROR("Unknown exception in bounded_thread_pool task"); });
      }
      // fn is destroyed before reservation, outside mu_, including after worker exceptions.
    }
  }

  std::mutex mu_;
  std::condition_variable cv_capacity_;  // reserve() waits here when at capacity
  std::condition_variable cv_idle_;      // wait_all() waits here
  std::condition_variable cv_work_;      // worker threads wait here for work

  int active_{0};
  const int capacity_;
  bool interrupted_{false};
  bool stop_requested_{false};

  /// std::list, not std::queue: discard_queued(query_id) removes one query's items from the
  /// middle without disturbing the ordering of everyone else's.
  std::list<work_item> work_queue_;
  std::vector<std::thread> threads_;
};

}  // namespace sirius::exec
