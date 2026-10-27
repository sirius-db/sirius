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

#include "event/event.hpp"
#include "event/query_event_publisher.hpp"
#include "event/query_event_subscriber.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string_view>
#include <vector>

namespace sirius::test {

/// Records the events a GPU pipeline executor publishes about a task's reserve phase and its
/// completion, and lets a test wait until a condition over them holds.
class executor_event_recorder final : public event::query_event_subscriber {
 public:
  struct stall_event {
    query_id_t query;
    std::size_t operator_id;
    int gpu;
    std::size_t bytes;  ///< bytes needed, shortfall, or bytes granted, by event kind
  };
  struct completion_event {
    query_id_t query;
    std::size_t operator_id;
    int gpu;
    event::task_outcome outcome;
  };
  struct log {
    std::vector<stall_event> waits;
    std::vector<stall_event> downgrades;
    std::vector<stall_event> ends;
    std::vector<completion_event> completions;

    [[nodiscard]] std::size_t count(event::task_outcome outcome) const
    {
      return static_cast<std::size_t>(
        std::count_if(completions.begin(), completions.end(), [outcome](auto const& c) {
          return c.outcome == outcome;
        }));
    }
  };

  explicit executor_event_recorder(event::query_event_publisher& publisher)
    : query_event_subscriber(publisher,
                             {event::event_type::wait_for_memory_for_task,
                              event::event_type::memory_downgrade_for_task,
                              event::event_type::memory_wait_ended_for_task,
                              event::event_type::task_completed},
                             /*drain_on_stop=*/true)
  {
    start();
  }

  ~executor_event_recorder() override { stop(); }

  [[nodiscard]] std::string_view name() const noexcept override { return "executor-recorder"; }

  /// Wait up to @p limit for @p done to hold over the events received so far.
  bool wait_for(std::function<bool(log const&)> const& done,
                std::chrono::milliseconds limit = std::chrono::seconds(10))
  {
    std::unique_lock lock(_mutex);
    return _cv.wait_for(lock, limit, [&] { return done(_log); });
  }

  [[nodiscard]] log snapshot() const
  {
    std::lock_guard lock(_mutex);
    return _log;
  }

 private:
  template <typename Fn>
  void record(Fn&& fn) noexcept
  {
    try {
      {
        std::lock_guard lock(_mutex);
        fn(_log);
      }
      _cv.notify_all();
    } catch (...) {
    }
  }

  void on_wait_for_memory_for_task(event::event_id_t,
                                   event::timestamp_t,
                                   query_id_t query_id,
                                   std::size_t operator_id,
                                   int gpu_id,
                                   std::size_t bytes_needed) noexcept override
  {
    record([&](log& l) { l.waits.push_back({query_id, operator_id, gpu_id, bytes_needed}); });
  }

  void on_memory_downgrade_for_task(event::event_id_t,
                                    event::timestamp_t,
                                    query_id_t query_id,
                                    std::size_t operator_id,
                                    int gpu_id,
                                    std::size_t shortfall_bytes) noexcept override
  {
    record(
      [&](log& l) { l.downgrades.push_back({query_id, operator_id, gpu_id, shortfall_bytes}); });
  }

  void on_memory_wait_ended_for_task(event::event_id_t,
                                     event::timestamp_t,
                                     query_id_t query_id,
                                     std::size_t operator_id,
                                     int gpu_id,
                                     std::size_t granted_bytes) noexcept override
  {
    record([&](log& l) { l.ends.push_back({query_id, operator_id, gpu_id, granted_bytes}); });
  }

  void on_task_completed(event::event_id_t,
                         event::timestamp_t,
                         query_id_t query_id,
                         std::size_t operator_id,
                         int gpu_id,
                         event::task_outcome outcome) noexcept override
  {
    record([&](log& l) { l.completions.push_back({query_id, operator_id, gpu_id, outcome}); });
  }

  mutable std::mutex _mutex;
  std::condition_variable _cv;
  log _log;
};

}  // namespace sirius::test
