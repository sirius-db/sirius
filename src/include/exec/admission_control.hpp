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

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <stop_token>

namespace sirius::exec {

// ---------------------------------------------------------------------------
// admission_control — blocking budget-based backpressure
// ---------------------------------------------------------------------------
//
// Hands out RAII `slot`s that reserve a portion of a fixed budget.  Callers
// block in acquire() until their request fits, and release automatically on
// slot destruction.
//
// Deadlock-avoidance: if a request is larger than the remaining budget but
// there are no outstanding slots (nobody to wait on), the request is granted
// and reserves the entire budget.  This keeps a single oversized request
// making progress instead of stalling forever.

class admission_control {
 public:
  class slot {
   public:
    slot() = default;
    ~slot();

    slot(slot&& o) noexcept;
    slot& operator=(slot&& o) noexcept;

    slot(slot const&)            = delete;
    slot& operator=(slot const&) = delete;

    /// Amount actually reserved from the budget.  May exceed the requested
    /// size in the deadlock-avoidance case.
    [[nodiscard]] size_t size() const noexcept { return _reserved; }

    /// True if this slot holds a live reservation.
    explicit operator bool() const noexcept { return _ctrl != nullptr; }

   private:
    friend class admission_control;
    slot(admission_control* ctrl, size_t reserved) noexcept : _ctrl(ctrl), _reserved(reserved) {}

    admission_control* _ctrl{nullptr};
    size_t _reserved{0};
  };

  explicit admission_control(size_t budget) noexcept;
  ~admission_control() = default;

  admission_control(admission_control const&)            = delete;
  admission_control& operator=(admission_control const&) = delete;

  /// Reserve @p size units of budget and return a slot.  Blocks until either:
  ///   - normal:   in_use + size <= budget  →  reserves exactly @p size
  ///   - fallback: no slots outstanding but request still can't fit
  ///               →  reserves the full budget (prevents deadlock for a
  ///                   request larger than the total budget)
  /// If @p stop fires during the wait, returns a disengaged slot.
  [[nodiscard]] slot acquire(size_t size, std::stop_token stop = {});

  /// Block until every slot handed out by @c acquire has been destroyed.
  /// Used by callers that need to drain in-flight work before tearing
  /// down state captured inside outstanding slot-bearing closures.  Safe
  /// to call concurrently with @c acquire and slot destruction; the
  /// condition is observed under the same mutex that @c release uses to
  /// decrement @c _active_slots.
  void wait_for_idle();

  [[nodiscard]] size_t budget() const noexcept { return _budget; }

 private:
  void release(size_t reserved) noexcept;

  const size_t _budget;
  size_t _in_use{0};
  size_t _active_slots{0};
  std::mutex _mtx;
  std::condition_variable_any _cv;
};

}  // namespace sirius::exec
