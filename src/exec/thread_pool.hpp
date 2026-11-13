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
                              sirius::exec::invocable<void() noexcept> per_thread_init = nullptr,
                              std::vector<std::vector<int>> domains                    = {})
  {
    if (num_threads <= 0 || domains.size() > static_cast<size_t>(num_threads))
      throw std::invalid_argument("thread pool needs at least one worker per domain");
    domains_ = std::move(domains);
    queues_.resize(domains_.size() + 1);
    threads_.reserve(num_threads);

    std::unique_ptr<std::latch> init_latch;
    init_latch = std::make_unique<std::latch>(num_threads);
    std::atomic<bool> affinity_failed{false};

    auto* init_fn_ptr = per_thread_init ? &per_thread_init : nullptr;
    auto* latch_ptr   = init_latch.get();

    try {
      for (int i = 0; i < num_threads; ++i) {
        auto domain = domains_.empty() ? 0 : i % domains_.size() + 1;
        auto& t     = threads_.emplace_back(
          [this, init_fn_ptr, latch_ptr, domain, &cpu_ids, &affinity_failed]() {
            auto const& cpus = domain ? domains_[domain - 1] : cpu_ids;
            if (!cpus.empty()) {
              cpu_set_t expected{}, actual{};
              CPU_ZERO(&expected);
              for (int cpu : cpus) {
                if (cpu < 0 || cpu >= CPU_SETSIZE) {
                  if (domain) affinity_failed = true;
                  continue;
                }
                CPU_SET(cpu, &expected);
              }
              if (!thread_util::set_current_thread_affinity(cpus) ||
                  pthread_getaffinity_np(pthread_self(), sizeof(actual), &actual) != 0 ||
                  !CPU_EQUAL(&expected, &actual)) {
                if (domain) affinity_failed = true;
              }
            }
            if (init_fn_ptr) { (*init_fn_ptr)(); }
            latch_ptr->count_down();
            work_loop(domain);
          });
        if (!name.empty()) {
          std::ignore =
            sirius::exec::thread_util::set_thread_name(t, name + "_" + std::to_string(i));
        }
      }

      init_latch->wait();
      if (affinity_failed)
        throw std::runtime_error("thread pool CPU affinity could not be applied");
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

  // Lower priorities run first within each domain; equal priorities retain FIFO
  // within each queue. Unrouted and domain queues alternate on equal priority. Running
  // callbacks are not preempted. Callers without a priority retain FIFO ordering.
  void schedule(std::invocable auto&& fn, queue_priority priority = 0, int domain = -1)
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
    if (domain < -1 || domain >= static_cast<int>(domains_.size()))
      throw std::invalid_argument("invalid thread pool domain");
    auto& queue             = queues_[domain + 1];
    auto [bucket, inserted] = queue.try_emplace(priority);
    try {
      bucket->second.emplace_back(std::move(callback));
    } catch (...) {
      // A failed allocation must not leave an empty bucket for a worker to pop.
      if (inserted) queue.erase(bucket);
      throw;
    }
    cv_.notify_all();
  }

  /// \brief Alias for schedule(). Lets static_thread_pool satisfy the same
  ///        scheduler shape as scoped_dispatcher (which exposes enqueue()),
  ///        so generic dispatchers can target either.
  void enqueue(std::invocable auto&& fn) { schedule(std::forward<decltype(fn)>(fn)); }

  [[nodiscard]] std::size_t num_threads() const noexcept { return threads_.size(); }

  size_t num_domains() const noexcept { return domains_.size(); }

  void stop() noexcept
  {
    std::unique_lock l(mu_);
    stop_requested_ = true;
    cv_.notify_all();
  }

 private:
  [[nodiscard]] bool has_work_or_stopped(size_t domain) const
  {
    return !queues_[0].empty() || !queues_[domain].empty() || stop_requested_;
  }

  void work_loop(size_t domain)
  {
    // Per-worker tie breaker between global and local queues, not task classes.
    // Higher-priority work still wins; equal priorities alternate when both are ready.
    bool global_turn = true;
    while (!stop_requested_) {
      sirius::exec::invocable<void() noexcept> func;
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this, domain] { return has_work_or_stopped(domain); });
        if (stop_requested_) { break; }
        auto* queue = &queues_[domain];
        if (queue->empty() ||
            (!queues_[0].empty() &&
             (queues_[0].begin()->first < queue->begin()->first ||
              (queues_[0].begin()->first == queue->begin()->first && global_turn))))
          queue = &queues_[0];
        global_turn = queue != &queues_[0];
        auto bucket = queue->begin();
        func        = std::move(bucket->second.front());
        bucket->second.pop_front();
        if (bucket->second.empty()) queue->erase(bucket);
      }
      if (func == nullptr) { break; }
      func();
    }
  }

  std::mutex mu_;               // Protects all queue contents and submission/stop ordering.
  std::condition_variable cv_;  // Wakes workers when work arrives or shutdown is requested.
  using queue_type = std::map<queue_priority, std::deque<sirius::exec::invocable<void() noexcept>>>;
  // Immutable CPU masks indexed by pool domain, not OS NUMA node ID. Workers are
  // assigned round-robin and retain that mask for their lifetime. Empty means unrouted.
  std::vector<std::vector<int>> domains_;
  // One global queue at index 0 plus one queue per domain at index domain + 1.
  // Workers consume global work and their own domain only. Metadata and coalescing
  // share these queues; each queue orders by priority, then FIFO within that priority.
  std::vector<queue_type> queues_;
  // Stops acceptance and worker dequeueing; already-running callbacks finish.
  // Pending callbacks are not drained by stop(). Atomic for the outer worker loop.
  std::atomic<bool> stop_requested_{false};
  // Owned workers, joined at destruction (or if construction fails).
  std::vector<std::thread> threads_;
};

}  // namespace sirius::exec
