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

#include <exec/thread_pool.hpp>

#include <concepts>
#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include <source_location>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sirius::exec {

/// Concept matching the two task signatures the dispatcher supports:
/// either zero-arg @c F() or stop-token-aware @c F(std::stop_token).
/// @c wrap dispatches between the two via constexpr-if internally.
template <typename F>
concept scoped_dispatcher_task = std::invocable<F> || std::invocable<F, std::stop_token>;

class scoped_dispatcher {
 public:
  scoped_dispatcher(static_thread_pool& pool,
                    std::size_t max_concurrent_task = 0,
                    queue_priority priority         = 0)
    : pool_(pool),
      max_inflight_(max_concurrent_task == 0 ? pool.num_threads()
                                             : std::min(max_concurrent_task, pool.num_threads())),
      priority_(priority)
  {
  }

  ~scoped_dispatcher()
  {
    // User contract: drain (via wait_for_all, optionally preceded by
    // request_stop) before destruction. Loud assert beats silent UAF.
    std::lock_guard lk(mu_);
    assert(inflight_ == 0 && pending_empty() &&
           "scoped_dispatcher destroyed with outstanding work; "
           "call request_stop()/wait_for_all() first");
  }

  scoped_dispatcher(const scoped_dispatcher&)            = delete;
  scoped_dispatcher& operator=(const scoped_dispatcher&) = delete;

  // @brief Does not wait for an inflight slot. Overflow goes to the dispatcher-local
  // pending queue (unbounded by contract). Callable wrapping and queue insertion can throw.
  // A pool submission failure cancels pending work, releases the reserved slot, and rethrows.
  // Once stopped, successfully wrapped calls silently drop their task. Wrapping happens
  // before the stop check and can still throw even when the dispatcher is already stopped.
  void enqueue(scoped_dispatcher_task auto&& f, int domain = -1, bool coalescing = false)
  {
    queued_task task{wrap(std::forward<decltype(f)>(f)), domain};

    std::unique_lock lk(mu_);
    if (stop_.stop_requested()) return;

    if (inflight_ < max_inflight_) {
      ++inflight_;
      lk.unlock();
      submit(std::move(task));
    } else {
      pending_[coalescing].push_back(std::move(task));
    }
  }

  size_t num_domains() const noexcept { return pool_.num_domains(); }

  // @brief Blocking. Waits for an inflight slot, then submits directly to the pool.
  // Bypasses the pending queue — this is the producer-side back-pressure path.
  // Returns false iff request_stop() was observed while waiting (or already set).
  bool schedule(scoped_dispatcher_task auto&& f)
  {
    queued_task task{wrap(std::forward<decltype(f)>(f)), -1};

    std::unique_lock lk(mu_);
    bool ok = cv_slot_.wait(lk, stop_.get_token(), [this] { return inflight_ < max_inflight_; });
    if (!ok || stop_.stop_requested()) return false;

    ++inflight_;
    lk.unlock();
    submit(std::move(task));
    return true;
  }

  // Bind once before stopping. Called once, outside mu_, after stopped work and its
  // captures drain. The callback must not throw and must own its captured state;
  // wait_for_all() may return while the notification is still being delivered.
  void set_stop_drain_callback(std::function<void()> notify)
  {
    std::lock_guard lock(mu_);
    if (!notify || stop_.stop_requested() || stop_drained_)
      throw std::logic_error("dispatcher drain notification must bind once before stop");
    stop_drained_ = std::move(notify);
  }

  // Signal stop and dispose at most max_discard queued captures on this caller.
  // False means the caller must continue cleanup; running work drains independently.
  // The default preserves synchronous queue disposal for non-coordinator callers.
  bool request_stop(std::size_t max_discard = std::numeric_limits<std::size_t>::max())
  {
    {
      std::lock_guard lk(mu_);
      // Keep wait_for_all() blocked until callbacks and discarded captures have retired,
      // including when another caller is stopping the dispatcher concurrently.
      ++inflight_;
    }
    // Stop callbacks may reenter the dispatcher, so request cancellation outside mu_.
    stop_.request_stop();
    for (std::size_t count = 0;; ++count) {
      queued_task discarded;
      {
        std::unique_lock lk(mu_);
        if (pending_empty() || count == max_discard) {
          bool const empty = pending_empty();
          release_slot(lk);
          return empty;
        }
        discarded = take_pending();
      }
      // Destroy outside mu_. Moving one callable at a time avoids allocating a temporary
      // container, so this cleanup is also usable after an allocation failure in submit().
    }
  }

  void wait_for_all()
  {
    std::unique_lock lk(mu_);
    cv_done_.wait(lk, [this] { return inflight_ == 0 && pending_empty(); });
  }

 private:
  using task_t = sirius::exec::invocable<void()>;
  struct queued_task {
    task_t work;      // Owns the callback and its captures until execution or cancellation.
    int domain = -1;  // Pool domain index; -1 permits execution by any pool worker.
  };

  static void log_exception_ptr(const std::exception_ptr& eptr,
                                std::source_location loc = std::source_location::current())
  {
    try {
      if (eptr) std::rethrow_exception(eptr);
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR(
        "Exception in scoped_dispatcher task at {}:{}: {}", loc.file_name(), loc.line(), e.what());
    } catch (...) {
      SIRIUS_LOG_ERROR(
        "Unknown exception in scoped_dispatcher task at {}:{}", loc.file_name(), loc.line());
    }
  }

  template <typename F>
  task_t wrap(F&& f)
  {
    return [this, f = std::forward<F>(f)]() mutable {
      if (!stop_.stop_requested()) {
        try {
          invoke_with_optional_stop_token(f, stop_.get_token());
        } catch (...) { /* swallow */
          log_exception_ptr(std::current_exception());
        }
      }
    };
  }

  template <typename F>
  static void invoke_with_optional_stop_token(F& f, std::stop_token st)
  {
    if constexpr (std::is_invocable_v<F&, std::stop_token>) {
      f(std::move(st));
    } else {
      f();
    }
  }

  bool pending_empty() const { return pending_[0].empty() && pending_[1].empty(); }
  queued_task take_pending()
  {
    // Alternate classes when both have work. Metadata cannot indefinitely delay
    // a ready batch, and a sustained coalescing stream cannot starve footers.
    auto selected = pending_[next_class_].empty() ? !next_class_ : next_class_;
    auto result   = std::move(pending_[selected].front());
    pending_[selected].pop_front();
    next_class_ = !selected;
    return result;
  }

  void run(queued_task task) noexcept
  {
    if (!pool_.num_domains()) {
      run_unrouted(std::move(task));
      return;
    }
    try {
      task.work();
    } catch (...) {
      try {
        log_exception_ptr(std::current_exception());
      } catch (...) {
      }
    }
    task.work = nullptr;  // Dispose captures outside mu_.
    {
      std::unique_lock lk(mu_);
      if (stop_.stop_requested() || pending_empty()) {
        release_slot(lk);
        return;
      }
      task = take_pending();
    }
    // Each handoff is routed anew: never execute another domain's work locally.
    // submit failure stops/disposes pending work and releases this reserved slot.
    try {
      submit(std::move(task));
    } catch (...) {
      try {
        log_exception_ptr(std::current_exception());
      } catch (...) {
      }
    }
  }

  // Preserve the allocation-failure fallback for existing pools without placement.
  // Pending captures remain here until a continuation actually starts.
  void run_unrouted(queued_task task) noexcept
  {
    bool can_yield = true;
    for (;;) {
      if (!task.work) {
        std::unique_lock lk(mu_);
        if (stop_.stop_requested() || pending_empty()) {
          release_slot(lk);
          return;
        }
        task = take_pending();
      }
      try {
        task.work();
      } catch (...) {
        try {
          log_exception_ptr(std::current_exception());
        } catch (...) {
        }
      }
      task.work = nullptr;
      {
        std::unique_lock lk(mu_);
        if (stop_.stop_requested() || pending_empty()) {
          release_slot(lk);
          return;
        }
      }
      if (can_yield) {
        try {
          pool_.schedule([this] { run_unrouted({}); }, priority_);
          return;
        } catch (...) {
          can_yield = false;
        }
      }
    }
  }

  // Terminal handoff: unlock before notification and do not touch the dispatcher
  // afterwards. Its waiter may release the last owner as soon as work drains.
  void release_slot(std::unique_lock<std::mutex>& lock) noexcept
  {
    std::function<void()> notify;
    --inflight_;
    cv_slot_.notify_one();
    if (inflight_ == 0 && pending_empty()) {
      cv_done_.notify_all();
      if (stop_.stop_requested()) notify = std::move(stop_drained_);
    }
    lock.unlock();
    if (notify) notify();
  }

  void submit(queued_task task)
  {
    try {
      auto domain = task.domain;
      pool_.schedule(
        [this, task = std::move(task)]() mutable { run(std::move(task)); }, priority_, domain);
    } catch (...) {
      // No worker owns this slot. Cancel pending work without executing user tasks on the
      // producer thread, then release the failed slot and propagate the original exception.
      request_stop();
      std::unique_lock lk(mu_);
      release_slot(lk);
      throw;
    }
  }

  static_thread_pool& pool_;
  std::size_t max_inflight_;
  const queue_priority priority_;

  std::mutex mu_;
  std::condition_variable cv_done_;      // drained signal; not stop-aware
  std::condition_variable_any cv_slot_;  // slot-open signal; stop-aware via wait(token, pred)
  // Overflow beyond max_inflight_: metadata at index 0, coalescing at index 1.
  // Each task retains its domain; these are class queues, not per-domain queues.
  std::deque<queued_task> pending_[2];
  // Preferred class on the next handoff; alternate when both classes have work.
  bool next_class_      = true;
  std::size_t inflight_ = 0;  // Worker slots plus active cancellation cleanup calls.
  std::stop_source stop_;
  std::function<void()> stop_drained_;
};

}  // namespace sirius::exec
