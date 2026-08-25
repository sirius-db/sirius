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
#include <mutex>
#include <tuple>

namespace sirius::scan_manager {

/**
 * @brief Rations one budget of concurrent device IO between the readahead and
 *        the reads the executor does for itself.
 *
 * Both compete for the same request queue, so both spend from the same purse. A
 * prefetch takes a ticket and returns it when its IO settles. A scan the
 * readahead never covered is doing that same IO itself, so it takes one too --
 * but it can never be made to wait for one, because it is the query's critical
 * path. It BORROWS instead, and the debt is repaid out of the next tickets to
 * come back rather than those going to a new prefetch.
 *
 * The effect is that the readahead throttles itself in exact proportion to how
 * much the executor is competing with it, and recovers on its own as the
 * competition ends -- with no pause flag to get stuck in, and no signal needed
 * to say when to resume.
 *
 * Counted rather than semaphored because the count has to be able to go
 * NEGATIVE, and a semaphore cannot. A negative count IS the debt: the readahead
 * cannot take a ticket until it is back above zero, so returning tickets cover
 * the over-subscription before they let anything new start.
 */
class gatekeeper {
 public:
  /// @p budget is how many concurrent device IOs are worth driving at once.  It
  /// describes the backend rather than the query, so it is fixed for the
  /// gatekeeper's life and taken here instead of at every arming.
  ///
  /// Starts with NO tickets: prefetching stays off until @ref reload arms it.
  /// That is what lets the worker just time out on @ref acquire_for rather than
  /// needing a separate "not started yet" flag to test.
  explicit gatekeeper(int budget) : _budget(budget) {}

  gatekeeper(gatekeeper const&)            = delete;
  gatekeeper& operator=(gatekeeper const&) = delete;

  /// Hand out the full budget, and undo any earlier @ref stop.
  ///
  /// Outstanding debt is cleared with it: it describes how the executor was
  /// competing before, which says nothing about now.
  void reload()
  {
    {
      std::lock_guard lock{_mutex};
      _available = _budget;
      _stopped   = false;
    }
    _cv.notify_all();
  }

  /// Wake everything parked in @ref acquire_for or @ref wait_for_all, and keep it
  /// awake.  Teardown must not sit out a timeout waiting on tickets that are
  /// never coming back.
  void stop()
  {
    {
      std::lock_guard lock{_mutex};
      _stopped = true;
    }
    _cv.notify_all();
  }

  /// Take a ticket for a prefetch, waiting up to @p timeout for one to come
  /// free. False means none did and the caller must not issue.
  ///
  /// Bounded on purpose: the caller owns the stop check, and a prefetch that
  /// cannot be paid for right now is one that should not be waited on.
  [[nodiscard]] bool acquire_for(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock{_mutex};
    if (!_cv.wait_for(lock, timeout, [this] { return _stopped || _available > 0; })) {
      return false;
    }
    if (_stopped) { return false; }
    --_available;
    return true;
  }

  /// Take a ticket for a read the readahead did not cover. Never blocks and
  /// never fails -- holding up the executor to protect read-ahead would be
  /// exactly backwards -- so this is the one caller that may push the count
  /// negative.
  ///
  /// @return true when it had to borrow, i.e. there was no ticket to take and
  ///         the readahead is now over-subscribed by one.
  bool acquire_or_borrow()
  {
    std::lock_guard lock{_mutex};
    bool const borrowed = _available <= 0;
    --_available;
    return borrowed;
  }

  /// Give a ticket back, whoever took it.  A negative count absorbs it as debt
  /// repayment; only once the count is positive again can @ref acquire_for take
  /// one, which is what keeps the throttle proportional.
  void release()
  {
    {
      std::lock_guard lock{_mutex};
      ++_available;
    }
    _cv.notify_one();
  }

  /// Block until every ticket is back.
  ///
  /// Bounded, and reports whether it got there: teardown must not hang on an IO
  /// that never completes.  @ref stop cuts the wait short, and then this returns
  /// false unless the tickets happened to be back anyway.
  [[nodiscard]] bool wait_for_all(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock{_mutex};
    std::ignore = _cv.wait_for(lock, timeout, [this] { return _stopped || _available >= _budget; });
    // Reports on the tickets, not on why the wait ended, so that a stopped wait
    // is not mistaken for a drained one.
    return _available >= _budget;
  }

  [[nodiscard]] int available() const
  {
    std::lock_guard lock{_mutex};
    return _available;
  }

  /// How over-subscribed the readahead is, for reporting.  Derived: it is just
  /// the negative side of the same counter.
  [[nodiscard]] int deficit() const
  {
    std::lock_guard lock{_mutex};
    return _available < 0 ? -_available : 0;
  }

 private:
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  /// Tickets this budget was armed with; the target @ref wait_for_all drains to.
  int _budget{0};
  /// Set by @ref stop: makes every wait give up at once, and stay given up until
  /// the next @ref reload.
  bool _stopped{false};
  /// Tickets free to be taken.  Negative means the executor took ones the budget
  /// could not cover, and that many must come back before the readahead may
  /// take another.
  int _available{0};
};

}  // namespace sirius::scan_manager
