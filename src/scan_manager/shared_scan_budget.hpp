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
#include "query_id.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stop_token>
#include <vector>
namespace sirius::scan_manager {
// Runtime-wide IO admission. A split may touch several backends: reserve them
// atomically, avoiding lock-order cycles. Demand never waits; its debt prevents
// speculative requests from using more of the same backend's budget.
class shared_scan_budget {
 public:
  struct request {
    std::uintptr_t backend;
    std::size_t limit;
  };

 private:
  struct usage {
    std::size_t active{}, limit{};
  };
  struct state {
    std::mutex mutex;
    std::condition_variable cv;
    std::map<std::uintptr_t, usage> backends;
    std::list<query_id_t> waiting;
  };

 public:
  class ticket {
   public:
    ticket(std::shared_ptr<state> owner, std::vector<request> requests)
      : owner(std::move(owner)), requests(std::move(requests))
    {
    }
    ~ticket()
    {
      {
        std::lock_guard lock(owner->mutex);
        for (auto const& r : requests)
          --owner->backends.at(r.backend).active;
      }
      owner->cv.notify_all();
    }
    ticket(ticket const&) = delete;

   private:
    friend class shared_scan_budget;
    std::shared_ptr<state> owner;
    std::vector<request> requests;
  };
  using token = std::shared_ptr<ticket>;

  token acquire(std::vector<request> requests, query_id_t query, std::stop_token stop, bool demand)
  {
    std::sort(requests.begin(), requests.end(), [](auto const& a, auto const& b) {
      return a.backend < b.backend;
    });
    requests.erase(std::unique(requests.begin(),
                               requests.end(),
                               [](auto const& a, auto const& b) { return a.backend == b.backend; }),
                   requests.end());
    auto result = std::make_shared<ticket>(shared, std::vector<request>{});
    std::unique_lock lock(shared->mutex);
    for (auto const& r : requests)
      shared->backends[r.backend].limit = r.limit;
    auto waiter = shared->waiting.insert(shared->waiting.end(), query);
    struct pending_guard {
      state& owner;
      std::list<query_id_t>::iterator position;
      ~pending_guard()
      {
        owner.waiting.erase(position);
        owner.cv.notify_all();
      }
    } pending{*shared, waiter};
    while (!demand) {
      if (stop.stop_requested()) { return {}; }
      bool oldest    = std::none_of(shared->waiting.begin(),
                                 shared->waiting.end(),
                                 [&](auto const& other) { return other < query; });
      bool available = std::all_of(requests.begin(), requests.end(), [&](auto const& r) {
        auto const& u = shared->backends.at(r.backend);
        return u.active < u.limit;
      });
      if (oldest && available) break;
      shared->cv.wait_for(lock, std::chrono::milliseconds(20));
    }
    result->requests = std::move(requests);
    for (auto const& r : result->requests)
      ++shared->backends.at(r.backend).active;
    return result;
  }
  std::size_t pending() const
  {
    std::lock_guard lock(shared->mutex);
    return shared->waiting.size();
  }
  std::size_t active(std::uintptr_t backend) const
  {
    std::lock_guard lock(shared->mutex);
    auto it = shared->backends.find(backend);
    return it == shared->backends.end() ? 0 : it->second.active;
  }

 private:
  std::shared_ptr<state> shared = std::make_shared<state>();
};
}  // namespace sirius::scan_manager
