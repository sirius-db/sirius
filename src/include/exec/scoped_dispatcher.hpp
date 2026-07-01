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

#include <absl/functional/any_invocable.h>
#include <exec/thread_pool.hpp>

#include <concepts>
#include <condition_variable>
#include <mutex>
#include <source_location>
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
  scoped_dispatcher(static_thread_pool& pool, std::size_t max_concurrent_task = 0)
    : pool_(pool),
      max_inflight_(max_concurrent_task == 0 ? pool.num_threads()
                                             : std::min(max_concurrent_task, pool.num_threads()))
  {
  }

  ~scoped_dispatcher()
  {
    // User contract: drain (via wait_for_all, optionally preceded by
    // request_stop) before destruction. Loud assert beats silent UAF.
    std::lock_guard lk(mu_);
    assert(inflight_ == 0 && pending_.empty() &&
           "scoped_dispatcher destroyed with outstanding work; "
           "call request_stop()/wait_for_all() first");
  }

  scoped_dispatcher(const scoped_dispatcher&)            = delete;
  scoped_dispatcher& operator=(const scoped_dispatcher&) = delete;

  // @brief Non-blocking. Always returns immediately. Overflow goes to the
  // dispatcher-local pending queue (unbounded by contract).
  // After request_stop(), enqueue is a silent no-op.
  void enqueue(scoped_dispatcher_task auto&& f)
  {
    auto task = wrap(std::forward<decltype(f)>(f));

    std::unique_lock lk(mu_);
    if (stop_.stop_requested()) return;

    if (inflight_ < max_inflight_) {
      ++inflight_;
      lk.unlock();
      pool_.schedule(std::move(task));
    } else {
      pending_.push_back(std::move(task));
    }
  }

  // @brief Blocking. Waits for an inflight slot, then submits directly to the pool.
  // Bypasses the pending queue — this is the producer-side back-pressure path.
  // Returns false iff request_stop() was observed while waiting (or already set).
  bool schedule(scoped_dispatcher_task auto&& f)
  {
    auto task = wrap(std::forward<decltype(f)>(f));

    std::unique_lock lk(mu_);
    bool ok = cv_slot_.wait(lk, stop_.get_token(), [this] { return inflight_ < max_inflight_; });
    if (!ok) return false;  // stop requested

    ++inflight_;
    lk.unlock();
    pool_.schedule(std::move(task));
    return true;
  }

  void request_stop()
  {
    std::lock_guard lk(mu_);
    stop_.request_stop();  // wakes cv_slot_ waiters via stop_callback
    pending_.clear();      // drop queued-but-not-started work
    if (inflight_ == 0) { cv_done_.notify_all(); }
  }

  void wait_for_all()
  {
    std::unique_lock lk(mu_);
    cv_done_.wait(lk, [this] { return inflight_ == 0 && pending_.empty(); });
  }

 private:
  using task_t = absl::AnyInvocable<void()>;

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
      on_task_done();
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

  void on_task_done()
  {
    task_t next;
    bool have_next = false;
    {
      std::lock_guard lk(mu_);
      if (!stop_.stop_requested() && !pending_.empty()) {
        // Hand the slot to a queued task; inflight_ unchanged → no slot opened.
        next = std::move(pending_.front());
        pending_.pop_front();
        have_next = true;
      } else {
        --inflight_;
        cv_slot_.notify_one();  // a real slot opened — wake one schedule()
        if (inflight_ == 0 && pending_.empty()) { cv_done_.notify_all(); }
      }
    }
    if (have_next) pool_.schedule(std::move(next));
  }

  static_thread_pool& pool_;
  std::size_t max_inflight_;

  std::mutex mu_;
  std::condition_variable cv_done_;      // drained signal; not stop-aware
  std::condition_variable_any cv_slot_;  // slot-open signal; stop-aware via wait(token, pred)
  std::deque<task_t> pending_;
  std::size_t inflight_ = 0;
  std::stop_source stop_;
};

}  // namespace sirius::exec
