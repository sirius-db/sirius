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
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace sirius::exec {

/// One monitor coordinates bounded execution, shared planning and exclusive maintenance.
/// Tokens are transferable: no mutex remains locked across user work or across threads.
class query_admission {
 public:
  enum class access { query, planning, maintenance };
  struct counts {
    std::size_t queued_queries{}, active_queries{}, planners{}, maintenance_waiters{};
    bool maintenance_active{}, closing{};
    std::uint64_t completed_queries{};
    std::chrono::steady_clock::time_point last_queued{}, last_admitted{}, last_completed{};
  };

 private:
  struct waiter {
    access kind;
    std::uint64_t ticket;
  };
  struct state {
    std::mutex mutex;
    std::condition_variable changed;
    std::list<waiter> waiting;
    counts current;
    std::size_t limit{1};
    std::uint64_t next_ticket{1};
  };

 public:
  class permit {
   public:
    permit() = default;
    permit(permit&& other) noexcept
      : _state(std::move(other._state)), _kind(other._kind), _ticket(other._ticket)
    {
    }
    permit& operator=(permit&& other) noexcept
    {
      if (this != &other) {
        reset();
        _state  = std::move(other._state);
        _kind   = other._kind;
        _ticket = other._ticket;
      }
      return *this;
    }
    permit(const permit&)            = delete;
    permit& operator=(const permit&) = delete;
    ~permit() { reset(); }
    explicit operator bool() const noexcept { return bool(_state); }
    std::uint64_t ticket() const noexcept { return _ticket; }
    void reset() noexcept
    {
      auto s = std::move(_state);
      if (!s) return;
      {
        std::lock_guard lock(s->mutex);
        if (_kind == access::query) {
          --s->current.active_queries;
          ++s->current.completed_queries;
          s->current.last_completed = std::chrono::steady_clock::now();
        } else if (_kind == access::planning)
          --s->current.planners;
        else
          s->current.maintenance_active = false;
      }
      s->changed.notify_all();
    }

   private:
    friend class query_admission;
    permit(std::shared_ptr<state> s, access kind, std::uint64_t ticket)
      : _state(std::move(s)), _kind(kind), _ticket(ticket)
    {
    }
    std::shared_ptr<state> _state;
    access _kind{access::query};
    std::uint64_t _ticket{};
  };

  void configure(std::size_t limit)
  {
    if (limit == 0) throw std::invalid_argument("query admission limit must be positive");
    std::lock_guard lock(_state->mutex);
    auto const& c = _state->current;
    if (c.active_queries || c.planners || c.maintenance_active || !_state->waiting.empty())
      throw std::logic_error("cannot reconfigure active query admission");
    _state->limit = limit;
  }

  /// check_cancel throws the caller's cancellation/health error. Polling is bounded because
  /// DuckDB interruption does not currently offer a condition-variable notification hook.
  template <class CheckCancel>
  permit acquire(access kind, CheckCancel check_cancel)
  {
    auto s = _state;
    std::unique_lock lock(s->mutex);
    check_cancel();
    if (s->current.closing) throw std::runtime_error("Sirius admission is closed");
    if (s->next_ticket > max_query_id)
      throw std::overflow_error("Sirius admission ID exhausted; restart the runtime");
    auto it = s->waiting.insert(s->waiting.end(), waiter{kind, s->next_ticket++});
    if (kind == access::query) {
      ++s->current.queued_queries;
      s->current.last_queued = std::chrono::steady_clock::now();
    }
    if (kind == access::maintenance) ++s->current.maintenance_waiters;
    auto remove = [&] {
      if (kind == access::query) --s->current.queued_queries;
      if (kind == access::maintenance) --s->current.maintenance_waiters;
      s->waiting.erase(it);
    };
    try {
      for (;;) {
        check_cancel();
        if (s->current.closing) throw std::runtime_error("Sirius admission is closed");
        auto const& c = s->current;
        bool ready    = !c.maintenance_active;
        if (kind == access::maintenance) {
          ready = ready && !c.active_queries && !c.planners &&
                  std::none_of(s->waiting.begin(), it, [](auto const& w) {
                    return w.kind == access::maintenance;
                  });
        } else {
          ready = ready && !c.maintenance_waiters;
          if (kind == access::query) {
            ready =
              ready && c.active_queries < s->limit &&
              std::none_of(
                s->waiting.begin(), it, [](auto const& w) { return w.kind == access::query; });
          }
        }
        if (ready) break;
        s->changed.wait_for(lock, std::chrono::milliseconds(20));
      }
      const auto ticket = it->ticket;
      remove();
      if (kind == access::query) {
        ++s->current.active_queries;
        s->current.last_admitted = std::chrono::steady_clock::now();
      } else if (kind == access::planning)
        ++s->current.planners;
      else
        s->current.maintenance_active = true;
      lock.unlock();
      s->changed.notify_all();
      return permit(std::move(s), kind, ticket);
    } catch (...) {
      remove();
      lock.unlock();
      s->changed.notify_all();
      throw;
    }
  }

  counts snapshot() const
  {
    std::lock_guard lock(_state->mutex);
    return _state->current;
  }
  void close() noexcept
  {
    {
      std::lock_guard lock(_state->mutex);
      _state->current.closing = true;
    }
    _state->changed.notify_all();
  }
  void wait_until_idle()
  {
    std::unique_lock lock(_state->mutex);
    _state->changed.wait(lock, [&] {
      auto const& c = _state->current;
      return !c.active_queries && !c.planners && !c.maintenance_active && _state->waiting.empty();
    });
  }

 private:
  std::shared_ptr<state> _state = std::make_shared<state>();
};
}  // namespace sirius::exec
