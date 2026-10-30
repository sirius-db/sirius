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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace sirius::memory {

/**
 * @brief Per-GPU counts of tasks that are stalled on memory, read synchronously by components
 *        that allocate speculatively and must back off while a task is starved.
 *
 * Two counts are kept per GPU:
 * - **waiting**: tasks whose reserve phase is stalled. The GPU pipeline executor enters a
 *   @ref waiter before it announces the stall and before it blocks, and releases it when the
 *   whole reserve phase (blocking wait, partial grant, downgrade, final attempt) is over.
 * - **oom pending**: tasks that were rescheduled after running out of memory and have not run
 *   to completion since. The @ref oom_token travels in the rescheduled task's local state, so it is
 *   released exactly when that task is destroyed: on success, on a terminal failure, or when the
 *   query's tasks are discarded.
 *
 * Both are RAII holds, so a count can neither leak nor be released twice. A hold shares
 * ownership of the board, so it may outlive every other owner.
 *
 * The counts are read with @ref waiting and @ref oom_pending rather than learned from events,
 * because event delivery is asynchronous and lags the stall it reports. Every count update and
 * read is sequentially consistent: a reader that reserves memory and then sees no waiter, and an
 * executor that enters a wait and then fails to reserve, cannot both miss each other. This holds
 * only from @ref enter_wait onward: the executor enters the wait after its first, non-blocking
 * reservation attempt has failed, so a reservation that lands between that attempt and
 * @ref enter_wait can still see no waiter. The counts are advisory; a reader that must not take
 * memory a stalled task needs enforces that with its own reservation limit.
 *
 * Devices outside @c [0, max_device_id] are ignored: holds on them are inactive and their counts
 * read as zero.
 *
 * The board must be owned by a @c std::shared_ptr; @ref enter_wait and @ref hold_oom throw
 * @c std::bad_weak_ptr otherwise.
 */
class memory_wait_board : public std::enable_shared_from_this<memory_wait_board> {
  enum class counter : std::uint8_t { waiting, oom };

  /// Move-only hold on one count of one device. A default-constructed or moved-from hold is
  /// inactive and releases nothing.
  template <counter Counter>
  class hold {
   public:
    hold() noexcept = default;
    ~hold() { release(); }

    hold(hold const&)            = delete;
    hold& operator=(hold const&) = delete;

    hold(hold&& other) noexcept
      : _board(std::move(other._board)), _device(std::exchange(other._device, -1))
    {
    }

    hold& operator=(hold&& other) noexcept
    {
      if (this != &other) {
        release();
        _board  = std::move(other._board);
        _device = std::exchange(other._device, -1);
      }
      return *this;
    }

    /// Whether this hold contributes to a count.
    [[nodiscard]] bool active() const noexcept { return _board != nullptr; }

    /// The device this hold counts against, or -1 when inactive.
    [[nodiscard]] int device() const noexcept { return _device; }

   private:
    friend class memory_wait_board;

    hold(std::shared_ptr<memory_wait_board> board, int device) noexcept
      : _board(std::move(board)), _device(device)
    {
    }

    void release() noexcept
    {
      if (_board) {
        _board->slot(Counter, _device).fetch_sub(1, std::memory_order_seq_cst);
        _board.reset();
        _device = -1;
      }
    }

    std::shared_ptr<memory_wait_board> _board;
    int _device{-1};
  };

 public:
  /// Counts one stalled reserve phase on a device while alive.
  using waiter = hold<counter::waiting>;
  /// Counts one OOM-rescheduled task pending on a device while alive.
  using oom_token = hold<counter::oom>;

  /// Board with one slot per device id in @c [0, max_device_id]. A negative @p max_device_id
  /// gives a board with no slots, on which every hold is inactive.
  explicit memory_wait_board(int max_device_id)
    : _n(max_device_id >= 0 ? max_device_id + 1 : 0),
      _waiting(std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(_n))),
      _oom(std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(_n)))
  {
  }

  memory_wait_board(memory_wait_board const&)            = delete;
  memory_wait_board& operator=(memory_wait_board const&) = delete;
  memory_wait_board(memory_wait_board&&)                 = delete;
  memory_wait_board& operator=(memory_wait_board&&)      = delete;

  /// Count a stalled reserve phase on @p device until the returned hold is destroyed.
  [[nodiscard]] waiter enter_wait(int device) { return acquire<counter::waiting>(device); }

  /// Count an OOM-rescheduled task on @p device until the returned hold is destroyed.
  [[nodiscard]] oom_token hold_oom(int device) { return acquire<counter::oom>(device); }

  /// Number of stalled reserve phases on @p device.
  [[nodiscard]] int waiting(int device) const noexcept { return load(counter::waiting, device); }

  /// Number of OOM-rescheduled tasks pending on @p device.
  [[nodiscard]] int oom_pending(int device) const noexcept { return load(counter::oom, device); }

  /// Number of device slots, i.e. @c max_device_id + 1.
  [[nodiscard]] int device_count() const noexcept { return _n; }

 private:
  [[nodiscard]] bool in_range(int device) const noexcept { return device >= 0 && device < _n; }

  [[nodiscard]] std::atomic<int>& slot(counter which, int device) const noexcept
  {
    return (which == counter::waiting ? _waiting : _oom)[static_cast<std::size_t>(device)];
  }

  [[nodiscard]] int load(counter which, int device) const noexcept
  {
    return in_range(device) ? slot(which, device).load(std::memory_order_seq_cst) : 0;
  }

  template <counter Counter>
  [[nodiscard]] hold<Counter> acquire(int device)
  {
    if (!in_range(device)) { return {}; }
    // Take ownership first: if it throws, no count has been raised.
    auto self = shared_from_this();
    slot(Counter, device).fetch_add(1, std::memory_order_seq_cst);
    return hold<Counter>(std::move(self), device);
  }

  int _n{0};
  std::unique_ptr<std::atomic<int>[]> _waiting;
  std::unique_ptr<std::atomic<int>[]> _oom;
};

}  // namespace sirius::memory
