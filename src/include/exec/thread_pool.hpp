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
#include <vector>

namespace sirius::exec {

class thread_pool {
 public:
  explicit thread_pool(int num_threads,
                       const std::string& name                             = "thread_pool",
                       std::vector<int> cpu_ids                            = {},
                       absl::AnyInvocable<void() noexcept> per_thread_init = nullptr)
  {
    threads_.reserve(num_threads);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    std::for_each(
      cpu_ids.begin(), cpu_ids.end(), [&cpuset](int cpu_id) { CPU_SET(cpu_id, &cpuset); });

    std::unique_ptr<std::latch> init_latch;
    if (per_thread_init) { init_latch = std::make_unique<std::latch>(num_threads); }

    auto* init_fn_ptr = per_thread_init ? &per_thread_init : nullptr;
    auto* latch_ptr   = init_latch.get();

    for (int i = 0; i < num_threads; ++i) {
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

  thread_pool(const thread_pool&)            = delete;
  thread_pool& operator=(const thread_pool&) = delete;

  ~thread_pool()
  {
    stop();
    for (auto& t : threads_) {
      if (t.joinable()) { t.join(); }
    }
  }

  void schedule(absl::AnyInvocable<void()> fn)
  {
    assert(fn != nullptr);
    std::lock_guard l(mu_);
    queue_.emplace([this, fn = std::move(fn)]() mutable noexcept {
      try {
        fn();
      } catch (const std::exception& e) {
        SIRIUS_LOG_ERROR("Exception in thread pool task: {}, {}", e.what(), "closing thread pool");
        stop();
      } catch (...) {
        SIRIUS_LOG_ERROR("Exception in thread pool task, closing thread pool");
        stop();
      }
    });
    cv_.notify_one();
  }

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
      absl::AnyInvocable<void() noexcept> func;
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this] { return has_work_or_stopped(); });
        if (stop_requested_) { break; }
        func = std::move(queue_.front());
        queue_.pop();
      }
      if (func == nullptr) { break; }
      func();
    }
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<absl::AnyInvocable<void() noexcept>> queue_;
  std::atomic<bool> stop_requested_{false};
  std::vector<std::thread> threads_;
};

}  // namespace sirius::exec
