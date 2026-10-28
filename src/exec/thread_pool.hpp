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
#include "exec/queue_priority.hpp"
#include "exec/thread_util.hpp"
#include "log/logging.hpp"

#include <concepts>
#include <condition_variable>
#include <deque>
#include <exception>
#include <latch>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace sirius::exec {

class static_thread_pool {
 public:
  explicit static_thread_pool(int num_threads,
                              const std::string& name  = "thread_pool",
                              std::vector<int> cpu_ids = {},
                              sirius::exec::invocable<void() noexcept> per_thread_init = nullptr)
  {
    threads_.reserve(num_threads);

    std::unique_ptr<std::latch> init_latch;
    if (per_thread_init) { init_latch = std::make_unique<std::latch>(num_threads); }

    auto* init_fn_ptr = per_thread_init ? &per_thread_init : nullptr;
    auto* latch_ptr   = init_latch.get();

    try {
      for (int i = 0; i < num_threads; ++i) {
        auto& t = threads_.emplace_back([this, init_fn_ptr, latch_ptr]() {
          if (init_fn_ptr) {
            (*init_fn_ptr)();
            latch_ptr->count_down();
          }
          work_loop();
        });
        if (!name.empty()) {
          std::ignore =
            sirius::exec::thread_util::set_thread_name(t, name + "_" + std::to_string(i));
        }
        if (!cpu_ids.empty()) {
          std::ignore = sirius::exec::thread_util::set_thread_affinity(t, cpu_ids);
        }
      }

      if (init_latch) { init_latch->wait(); }
    } catch (...) {
      stop();
      for (auto& thread : threads_) {
        if (thread.joinable()) { thread.join(); }
      }
      throw;
    }
  }

  static_thread_pool(const static_thread_pool&)            = delete;
  static_thread_pool& operator=(const static_thread_pool&) = delete;

  ~static_thread_pool()
  {
    stop();
    for (auto& t : threads_) {
      if (t.joinable()) { t.join(); }
    }
  }

  // Lower priorities run first; equal priorities retain submission order. Running
  // callbacks are not preempted. Callers without a priority retain FIFO ordering.
  void schedule(std::invocable auto&& fn, queue_priority priority = 0)
  {
    sirius::exec::invocable<void() noexcept> callback =
      [callable = std::forward<decltype(fn)>(fn)]() mutable noexcept {
        try {
          callable();
        } catch (const std::exception& e) {
          SIRIUS_LOG_ERROR("Exception thrown from thread_pool on_error handler {}", e.what());
        } catch (...) {
          SIRIUS_LOG_ERROR("Unknown exception thrown from thread_pool");
        }
      };
    // Keep the callable outside the lock's lifetime: failed submissions can destroy
    // captures which reenter the pool or their dispatcher.
    std::lock_guard l(mu_);
    if (stop_requested_) { throw std::runtime_error("thread pool is stopped"); }
    auto [bucket, inserted] = queue_.try_emplace(priority);
    try {
      bucket->second.emplace_back(std::move(callback));
    } catch (...) {
      // A failed allocation must not leave an empty bucket for a worker to pop.
      if (inserted) queue_.erase(bucket);
      throw;
    }
    cv_.notify_one();
  }

  /// \brief Alias for schedule(). Lets static_thread_pool satisfy the same
  ///        scheduler shape as scoped_dispatcher (which exposes enqueue()),
  ///        so generic dispatchers can target either.
  void enqueue(std::invocable auto&& fn) { schedule(std::forward<decltype(fn)>(fn)); }

  [[nodiscard]] std::size_t num_threads() const noexcept { return threads_.size(); }

  void stop() noexcept
  {
    std::unique_lock l(mu_);
    stop_requested_ = true;
    cv_.notify_all();
  }

 private:
  [[nodiscard]] bool has_work_or_stopped() const { return !queue_.empty() || stop_requested_; }

  void work_loop()
  {
    while (!stop_requested_) {
      sirius::exec::invocable<void() noexcept> func;
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this] { return has_work_or_stopped(); });
        if (stop_requested_) { break; }
        auto bucket = queue_.begin();
        func        = std::move(bucket->second.front());
        bucket->second.pop_front();
        if (bucket->second.empty()) queue_.erase(bucket);
      }
      if (func == nullptr) { break; }
      func();
    }
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::map<queue_priority, std::deque<sirius::exec::invocable<void() noexcept>>> queue_;
  std::atomic<bool> stop_requested_{false};
  std::vector<std::thread> threads_;
};

}  // namespace sirius::exec
