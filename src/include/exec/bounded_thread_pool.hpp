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

#include <absl/functional/any_invocable.h>

#include <condition_variable>
#include <latch>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::exec {

/**
 * @brief A thread pool with bounded concurrency.
 *
 * Merges a fixed-size thread pool with a counting semaphore, eliminating the need for
 * a separate kiosk object. Provides two scheduling modes:
 *
 *  - One-shot:  schedule(fn) — atomically reserves a slot and dispatches. Blocks when
 *               at capacity. Returns false if interrupted.
 *
 *  - Two-phase: reserve() returns a slot (blocks when at capacity, returns an invalid
 *               slot if interrupted). The caller then calls slot.dispatch(fn) after any
 *               inter-step coordination (e.g. signalling a task_request before waiting
 *               for the task to appear on a queue). Dropping the slot without calling
 *               dispatch() releases it back to the pool immediately.
 *
 * Lifecycle:
 *  - interrupt(): wake all blocked reserve()/schedule() calls without disturbing
 *                 in-flight work. Paired with resume() for drain-and-restart patterns.
 *  - resume():    re-enable after interrupt().
 *  - wait_all():  block until all in-flight tasks complete.
 *  - stop():      interrupt + join all worker threads (called automatically by destructor).
 */
class bounded_thread_pool {
 public:
  /**
   * @brief RAII handle representing a reserved execution slot.
   *
   * Obtained via bounded_thread_pool::reserve(). Either dispatch() is called to submit
   * work using this slot, or the slot goes out of scope without a dispatch (which
   * releases the slot back to the pool immediately without running any work).
   */
  class slot {
   public:
    explicit slot(bounded_thread_pool* pool = nullptr) noexcept : pool_(pool) {}

    slot(slot&& other) noexcept : pool_(other.pool_) { other.pool_ = nullptr; }

    slot& operator=(slot&& other) noexcept
    {
      if (this != &other) {
        release();
        pool_       = other.pool_;
        other.pool_ = nullptr;
      }
      return *this;
    }

    ~slot() { release(); }

    slot(const slot&)            = delete;
    slot& operator=(const slot&) = delete;

    [[nodiscard]] bool is_valid() const noexcept { return pool_ != nullptr; }
    explicit operator bool() const noexcept { return is_valid(); }

    /**
     * @brief Dispatch a function using this slot.
     *
     * The slot is consumed (becomes invalid). The function runs on a worker thread
     * and the slot is released automatically when it completes.
     */
    void dispatch(absl::AnyInvocable<void()> fn)
    {
      auto* p = std::exchange(pool_, nullptr);
      if (p) { p->dispatch_slot(std::move(fn)); }
    }

   private:
    void release()
    {
      if (auto* p = std::exchange(pool_, nullptr); p != nullptr) { p->release_slot(); }
    }

    bounded_thread_pool* pool_{nullptr};
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
   * @brief Interrupt all blocked reserve()/schedule() calls.
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

  // Called by slot::dispatch() — pushes fn to the worker queue with automatic slot release.
  void dispatch_slot(absl::AnyInvocable<void()> fn)
  {
    {
      std::lock_guard lock(mu_);
      work_queue_.push([this, fn = std::move(fn)]() mutable noexcept {
        try {
          fn();
        } catch (const std::exception& e) {
          SIRIUS_LOG_ERROR("Exception in bounded_thread_pool task: {}", e.what());
        } catch (...) {
          SIRIUS_LOG_ERROR("Unknown exception in bounded_thread_pool task");
        }
        release_slot();
      });
    }
    cv_work_.notify_one();
  }

  // Called by slot destructor (dropped without dispatch) or after task completion.
  void release_slot()
  {
    {
      std::lock_guard lock(mu_);
      --active_;
    }
    cv_capacity_.notify_one();
    cv_idle_.notify_all();
  }

 private:
  void work_loop()
  {
    while (true) {
      absl::AnyInvocable<void() noexcept> fn;
      {
        std::unique_lock lock(mu_);
        cv_work_.wait(lock, [&] { return !work_queue_.empty() || stop_requested_; });
        if (stop_requested_ && work_queue_.empty()) { break; }
        fn = std::move(work_queue_.front());
        work_queue_.pop();
      }
      fn();
    }
  }

  std::mutex mu_;
  std::condition_variable cv_capacity_;  // reserve()/schedule() wait here when at capacity
  std::condition_variable cv_idle_;      // wait_all() waits here
  std::condition_variable cv_work_;      // worker threads wait here for work

  int active_{0};
  const int capacity_;
  bool interrupted_{false};
  bool stop_requested_{false};

  std::queue<absl::AnyInvocable<void() noexcept>> work_queue_;
  std::vector<std::thread> threads_;
};

}  // namespace sirius::exec
