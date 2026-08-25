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
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

namespace sirius::exec {

/**
 * Tracks in-flight work with move-only RAII slots.
 *
 * Slots may outlive the tracker and be acquired while another thread waits. close() rejects new
 * slots, making the next zero permanent. All operations are thread-safe.
 */

class work_tracker {
  struct state {
    std::mutex m;
    std::condition_variable cv;
    std::size_t count{0};
    bool closed{false};
  };

 public:
  /// Move-only handle for one unit of work. Releasing the last slot wakes waiters.
  class slot {
   public:
    slot() = default;

    slot(slot&&) noexcept = default;
    slot& operator=(slot&& other) noexcept
    {
      if (this != &other) {
        release();
        _state = std::move(other._state);
      }
      return *this;
    }

    slot(const slot&)            = delete;
    slot& operator=(const slot&) = delete;

    ~slot() { release(); }

    /// True if this slot holds a live count.
    explicit operator bool() const noexcept { return _state != nullptr; }

   private:
    friend class work_tracker;
    explicit slot(std::shared_ptr<state> s) noexcept : _state(std::move(s)) {}

    void release() noexcept
    {
      if (!_state) { return; }
      bool drained = false;
      {
        std::lock_guard<std::mutex> lock(_state->m);
        drained = (--_state->count == 0);
      }
      // The shared state remains alive while notifying outside the lock.
      if (drained) { _state->cv.notify_all(); }
      _state.reset();
    }

    std::shared_ptr<state> _state;
  };

  work_tracker() : _state(std::make_shared<state>()) {}

  work_tracker(const work_tracker&)            = delete;
  work_tracker& operator=(const work_tracker&) = delete;

  /// Acquire a slot, or return an empty slot after close().
  [[nodiscard]] slot acquire()
  {
    {
      std::lock_guard<std::mutex> lock(_state->m);
      if (_state->closed) { return slot{}; }
      ++_state->count;
    }
    return slot{_state};
  }

  /// Permanently reject new slots. Trackers are per-query and are never reopened.
  void close()
  {
    std::lock_guard<std::mutex> lock(_state->m);
    _state->closed = true;
  }

  /// Return a snapshot of the number of live slots.
  [[nodiscard]] std::size_t outstanding() const
  {
    std::lock_guard<std::mutex> lock(_state->m);
    return _state->count;
  }

  /// Wait for the count to reach zero. Returns false on timeout.
  [[nodiscard]] bool wait_quiescent(std::chrono::milliseconds timeout)
  {
    std::unique_lock<std::mutex> lock(_state->m);
    return _state->cv.wait_for(lock, timeout, [this] { return _state->count == 0; });
  }

 private:
  std::shared_ptr<state> _state;
};

}  // namespace sirius::exec
