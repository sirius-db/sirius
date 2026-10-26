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
#include <condition_variable>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
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
    std::size_t active{}, limit{}, waiters{};
  };
  struct state {
    std::mutex mutex;
    std::condition_variable_any cv;
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
      if (requests.empty()) return;
      bool wake = false;
      {
        std::lock_guard lock(owner->mutex);
        for (auto const& r : requests) {
          auto it = owner->backends.find(r.backend);
          auto& u = it->second;
          --u.active;
          wake |= u.active < u.limit;
          if (u.active == 0 && u.waiters == 0) owner->backends.erase(it);
        }
        wake &= !owner->waiting.empty();
      }
      if (wake) owner->cv.notify_all();
    }
    ticket(ticket const&)            = delete;
    ticket& operator=(ticket const&) = delete;

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
    for (std::size_t i = 0; i < requests.size(); ++i) {
      if (requests[i].limit == 0)
        throw std::invalid_argument("shared scan budget requires a positive backend limit");
      if (i && requests[i - 1].backend == requests[i].backend &&
          requests[i - 1].limit != requests[i].limit)
        throw std::invalid_argument("conflicting limits for one scan backend");
    }
    requests.erase(std::unique(requests.begin(),
                               requests.end(),
                               [](auto const& a, auto const& b) { return a.backend == b.backend; }),
                   requests.end());
    if (!demand && stop.stop_requested()) return {};
    auto result = std::make_shared<ticket>(shared, std::vector<request>{});
    if (requests.empty()) return result;
    std::unique_lock lock(shared->mutex);
    // Validate the whole request before changing any accounting. Runtime configuration
    // supplies one stable limit per backend, including demand reads.
    for (auto const& r : requests) {
      auto it = shared->backends.find(r.backend);
      if (it != shared->backends.end() && it->second.limit != r.limit)
        throw std::invalid_argument("scan backend limit changed while in use");
    }
    struct pending_guard {
      state& owner;
      // The span remains valid when the vector transfers into the returned ticket.
      std::span<const request> requests;
      std::size_t registered{0};
      std::list<query_id_t>::iterator position;
      ~pending_guard()
      {
        for (std::size_t i = 0; i < registered; ++i) {
          auto it = owner.backends.find(requests[i].backend);
          if (--it->second.waiters == 0 && it->second.active == 0) owner.backends.erase(it);
        }
        if (position != owner.waiting.end()) {
          owner.waiting.erase(position);
          // Removing a waiter can unblock a younger query, even on another backend:
          // this baseline intentionally retains global oldest-query ordering.
          if (!owner.waiting.empty()) owner.cv.notify_all();
        }
      }
    } pending{*shared, requests, 0, shared->waiting.end()};
    for (auto const& r : requests) {
      auto [it, inserted] = shared->backends.try_emplace(r.backend, usage{0, r.limit, 0});
      ++it->second.waiters;
      ++pending.registered;
    }
    auto available = [&] {
      bool oldest = std::none_of(shared->waiting.begin(),
                                 shared->waiting.end(),
                                 [&](auto const& other) { return other < query; });
      return oldest && std::all_of(requests.begin(), requests.end(), [&](auto const& r) {
               auto const& u = shared->backends.at(r.backend);
               return u.active < u.limit;
             });
    };
    // Demand accrues debt without joining the speculative wait queue. An immediately
    // admissible speculative request likewise needs no queue insertion or notification.
    if (!demand) {
      if (!available()) {
        pending.position = shared->waiting.insert(shared->waiting.end(), query);
        if (!shared->cv.wait(lock, stop, available)) return {};
      }
      if (stop.stop_requested()) return {};
    }
    for (auto const& r : requests)
      ++shared->backends.at(r.backend).active;
    result->requests = std::move(requests);
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
