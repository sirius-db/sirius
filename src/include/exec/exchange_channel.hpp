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

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace sirius::exec {

/// Lightweight handle pushed through the channel. The shared_data_repository is the owner of
/// record; queued batches therefore sit idle where the downgrade sweep can see and spill them.
struct exchange_batch_handle {
  uint64_t batch_id;       // repository batch id — repo is the owner of record
  std::size_t size_bytes;  // size estimate captured at registration (byte accounting)
};

/// Bounded MPMC queue of batch handles with close-then-drain end-of-stream:
/// close() forbids further pushes, already-queued items remain poppable, and
/// drained() (= closed() && empty()) is the terminal EOS predicate.
///
/// Capacity is bounded by item count and optionally by cumulative bytes. Engine workers use
/// the non-blocking try_push / try_pop; blocking push / pop serve the wrapper / test side.
/// Callbacks (on_push / on_pop / on_close) fire outside the lock and must not capture raw
/// pointers to objects the channel can outlive.
class exchange_channel {
 public:
  struct config {
    std::size_t capacity_items;      // required, > 0
    std::size_t capacity_bytes = 0;  // 0 = no byte bound
  };

  explicit exchange_channel(config cfg) : _cfg(std::move(cfg))
  {
    if (_cfg.capacity_items == 0) {
      throw std::invalid_argument("exchange_channel: capacity_items must be > 0");
    }
  }

  // -----------------------------------------------------------------------
  // Producer side
  // -----------------------------------------------------------------------

  /// Non-blocking push. Returns false when the handle can't be admitted (item/byte bound would
  /// be crossed) or the channel is closed; never blocks.
  [[nodiscard]] bool try_push(exchange_batch_handle h)
  {
    std::function<void()> cb;
    {
      std::unique_lock<std::mutex> lock(_mutex);
      if (_closed || !can_push_unlocked(h)) return false;
      cb = enqueue_unlocked(h);
    }
    _cv.notify_all();
    if (cb) cb();
    return true;
  }

  /// Blocking push. Blocks while the handle can't be admitted; returns false once closed (and
  /// stays false).
  bool push(exchange_batch_handle h)
  {
    std::function<void()> cb;
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _cv.wait(lock, [&] { return can_push_unlocked(h) || _closed; });
      if (_closed) return false;
      cb = enqueue_unlocked(h);
    }
    _cv.notify_all();
    if (cb) cb();
    return true;
  }

  /// Idempotent close. Queued items remain poppable after close. Fires the on-close callback
  /// exactly once, on the first successful call, outside the lock.
  void close()
  {
    std::function<void()> cb;
    {
      std::unique_lock<std::mutex> lock(_mutex);
      if (_closed) return;
      _closed = true;
      cb      = _on_close;
    }
    _cv.notify_all();
    if (cb) cb();
  }

  // -----------------------------------------------------------------------
  // Consumer side
  // -----------------------------------------------------------------------

  /// Non-blocking pop. Returns nullopt when empty (including when closed+drained); never blocks.
  std::optional<exchange_batch_handle> try_pop()
  {
    std::function<void()> cb;
    std::optional<exchange_batch_handle> result;
    {
      std::unique_lock<std::mutex> lock(_mutex);
      if (_queue.empty()) return std::nullopt;
      result = dequeue_unlocked(cb);
    }
    _cv.notify_all();
    if (cb) cb();
    return result;
  }

  /// Blocking pop. Blocks until an item is available or the channel is drained.
  /// Returns nullopt only when closed && empty (EOS reached).
  std::optional<exchange_batch_handle> pop()
  {
    std::function<void()> cb;
    std::optional<exchange_batch_handle> result;
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _cv.wait(lock, [&] { return !_queue.empty() || _closed; });
      if (_queue.empty()) return std::nullopt;  // closed && drained
      result = dequeue_unlocked(cb);
    }
    _cv.notify_all();
    if (cb) cb();
    return result;
  }

  // -----------------------------------------------------------------------
  // State queries (mutex-guarded — safe for task-admission decisions)
  // -----------------------------------------------------------------------

  [[nodiscard]] bool full() const
  {
    std::unique_lock<std::mutex> lock(_mutex);
    return full_unlocked();
  }

  [[nodiscard]] bool empty() const
  {
    std::unique_lock<std::mutex> lock(_mutex);
    return _queue.empty();
  }

  [[nodiscard]] std::size_t size() const
  {
    std::unique_lock<std::mutex> lock(_mutex);
    return _queue.size();
  }

  [[nodiscard]] std::size_t size_bytes() const
  {
    std::unique_lock<std::mutex> lock(_mutex);
    return _total_bytes;
  }

  [[nodiscard]] bool closed() const
  {
    std::unique_lock<std::mutex> lock(_mutex);
    return _closed;
  }

  /// Terminal EOS predicate: closed() && empty().
  [[nodiscard]] bool drained() const
  {
    std::unique_lock<std::mutex> lock(_mutex);
    return _closed && _queue.empty();
  }

  // -----------------------------------------------------------------------
  // Re-arm hooks (single-slot — last setter wins; fired outside the lock).
  // Wired by the stream session in #839; tests poll instead.
  // A firing operation snapshots the callback under the lock and invokes the copy after
  // unlocking, so replacing a callback does not synchronize with an in-flight invocation:
  // callbacks must not capture raw pointers to objects the channel can outlive (capture a
  // weak reference instead).
  // -----------------------------------------------------------------------

  void set_on_push(std::function<void()> cb)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _on_push = std::move(cb);
  }

  void set_on_pop(std::function<void()> cb)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _on_pop = std::move(cb);
  }

  /// Fired exactly once, after the first successful close() (repeated close() calls are a
  /// no-op and do not re-fire it), outside the lock. Distinct from on_push/on_pop: "the stream
  /// ended" is a different event than "data is available", with a different intended consumer
  /// (pipeline-completion re-evaluation vs. task-creation re-scheduling).
  void set_on_close(std::function<void()> cb)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _on_close = std::move(cb);
  }

 private:
  /// Shared tail of try_push/push: admit `h` and snapshot the push callback for the caller to
  /// fire outside the lock. Caller must hold _mutex and have checked can_push_unlocked().
  std::function<void()> enqueue_unlocked(const exchange_batch_handle& h)
  {
    _queue.push_back(h);
    _total_bytes += h.size_bytes;
    return _on_push;
  }

  /// Shared tail of try_pop/pop: dequeue the front handle and snapshot the pop callback into
  /// `cb` for the caller to fire outside the lock. Caller must hold _mutex; queue non-empty.
  exchange_batch_handle dequeue_unlocked(std::function<void()>& cb)
  {
    exchange_batch_handle result = _queue.front();
    _queue.pop_front();
    _total_bytes -= result.size_bytes;
    cb = _on_pop;
    return result;
  }

  /// full() without taking the lock. Caller must hold _mutex.
  [[nodiscard]] bool full_unlocked() const
  {
    if (_queue.size() >= _cfg.capacity_items) return true;
    if (_cfg.capacity_bytes > 0 && !_queue.empty() && _total_bytes >= _cfg.capacity_bytes)
      return true;
    return false;
  }

  /// True when the specific candidate handle `h` may be admitted right now. Caller must hold
  /// _mutex. Unlike full_unlocked() (a state-only query), this inspects the incoming handle so
  /// a push that would cross capacity_bytes is rejected even when the queue isn't yet at/over
  /// the bound (e.g. 40 queued + a 40-byte handle against a 50-byte bound).
  [[nodiscard]] bool can_push_unlocked(const exchange_batch_handle& h) const
  {
    if (_queue.size() >= _cfg.capacity_items) return false;
    // Reject a handle whose size would overflow the byte accounting; without this a wrapped
    // _total_bytes makes size_bytes()/full() wrong. Only reachable on a byte-unbounded
    // channel — a bounded non-empty channel already rejects below, and an empty channel has
    // _total_bytes == 0.
    if (h.size_bytes > std::numeric_limits<std::size_t>::max() - _total_bytes) return false;
    if (_cfg.capacity_bytes == 0) return true;
    // Oversized-batch rule: always admit into an empty channel so the stream never wedges.
    if (_queue.empty()) return true;
    if (_total_bytes >= _cfg.capacity_bytes) return false;
    // Subtraction (rather than _total_bytes + h.size_bytes) avoids overflow when the incoming
    // handle reports an implausibly large size.
    return h.size_bytes <= _cfg.capacity_bytes - _total_bytes;
  }

  config _cfg;
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<exchange_batch_handle> _queue;
  std::size_t _total_bytes{0};
  bool _closed{false};
  std::function<void()> _on_push;
  std::function<void()> _on_pop;
  std::function<void()> _on_close;
};

}  // namespace sirius::exec
