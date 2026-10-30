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

#include <chrono>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace sirius::scan_manager {
// One serialized step per attempt. Parked attempts consume no worker; workers also
// service deadlines, so no timer or retirement thread is needed.
class coordinator_pool {
  struct state;
  struct entry;

 public:
  using clock = std::chrono::steady_clock;
  enum class progress { runnable, parked, finished };
  struct step_result {
    progress next;
    std::optional<clock::time_point> deadline;
  };
  class registration;
  class wakeup {
   public:
    void operator()() const noexcept;

   private:
    friend class registration;
    std::weak_ptr<state> state_;
    size_t index_ = 0, generation_ = 0;
  };
  class registration {
   public:
    ~registration();
    registration(registration const&)            = delete;
    registration& operator=(registration const&) = delete;
    void wake() noexcept;
    // Does not retain the attempt or its callbacks. Stale wakes cannot target reused slots.
    wakeup notifier() const noexcept;
    // Waits until the final step has returned. The owner must service provider work first.
    void wait();

   private:
    friend class coordinator_pool;
    registration(std::shared_ptr<state>, std::shared_ptr<entry>);
    std::shared_ptr<state> state_;
    std::shared_ptr<entry> entry_;
  };
  explicit coordinator_pool(size_t workers);
  ~coordinator_pool();
  coordinator_pool(coordinator_pool const&)            = delete;
  coordinator_pool& operator=(coordinator_pool const&) = delete;
  // Starts workers lazily. Registration is dormant until wake(); wake never allocates
  // or waits for capacity. Drain before releasing the registration.
  // Steps must not wait for I/O, capacity or dependencies. stop and fail must not throw.
  // The pool does not select a CUDA device for its workers. Control hooks do host-side work;
  // device work runs in an execution domain that explicitly selects the required device.
  std::shared_ptr<registration> add(std::function<step_result()> step,
                                    std::function<void()> stop,
                                    std::function<void(std::exception_ptr)> fail);
  // Reject new registrations, request stop, and keep stepping existing retirement.
  // Owners must remain available for provider close until their registrations drain.
  void stop();
  size_t started_workers() const;
  void before_thread_for_testing(std::function<void(size_t)>);

 private:
  void start_workers();
  std::shared_ptr<state> state_;
  std::mutex lifecycle_mutex_;
  std::vector<std::thread> workers_;
  std::function<void(size_t)> before_thread_;
};
}  // namespace sirius::scan_manager
