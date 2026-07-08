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
#include <mutex>
#include <optional>
#include <stdexcept>

namespace sirius::exec {

/// Lightweight handle pushed through the channel. The shared_data_repository is the owner of
/// record; queued batches therefore sit idle where the downgrade sweep can see and spill them
/// (design §3/§7). A channel that owned batches would make them spill-invisible.
struct exchange_batch_handle {
  uint64_t batch_id;       // repository batch id — repo is the owner of record
  std::size_t size_bytes;  // size estimate captured at registration (byte accounting)
};

/// Bounded MPMC queue of batch handles with close-then-drain end-of-stream.
///
/// Semantics (design §3/§7):
///   - full()    ≡  items == capacity_items  OR  (byte_bound set AND bytes ≥ bound AND non-empty)
///   - Oversized-batch rule: a handle whose size_bytes > capacity_bytes is admitted into an
///     *empty* channel so the stream never wedges.
///   - close()   forbids further pushes; already-queued items remain poppable.
///   - drained() ≡  closed() && empty()  — terminal EOS predicate.
///   - Engine workers use try_push / try_pop only (non-blocking). Blocking push / pop are
///     provided for the wrapper / test side.
///   - Callbacks (on_push / on_pop) fire outside the lock; single-slot — last setter wins.
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

  /// Non-blocking push. Returns false when full or closed; never blocks.
  [[nodiscard]] bool try_push(exchange_batch_handle h)
  {
    std::function<void()> cb;
    {
      std::unique_lock<std::mutex> lock(_mutex);
      if (_closed || full_unlocked()) return false;
      _queue.push_back(h);
      _total_bytes += h.size_bytes;
      cb = _on_push;
    }
    _cv.notify_all();
    if (cb) cb();
    return true;
  }

  /// Blocking push. Blocks while full; returns false once closed (and stays false).
  bool push(exchange_batch_handle h)
  {
    std::function<void()> cb;
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _cv.wait(lock, [&] { return !full_unlocked() || _closed; });
      if (_closed) return false;
      _queue.push_back(h);
      _total_bytes += h.size_bytes;
      cb = _on_push;
    }
    _cv.notify_all();
    if (cb) cb();
    return true;
  }

  /// Idempotent close. Queued items remain poppable after close.
  void close()
  {
    {
      std::unique_lock<std::mutex> lock(_mutex);
      if (_closed) return;
      _closed = true;
    }
    _cv.notify_all();
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
      result = _queue.front();
      _queue.pop_front();
      _total_bytes -= result->size_bytes;
      cb = _on_pop;
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
      result = _queue.front();
      _queue.pop_front();
      _total_bytes -= result->size_bytes;
      cb = _on_pop;
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
  // Re-arm hooks (single-slot; fired outside the lock).
  // Wired by the stream session in #839; tests poll instead.
  // The owner must clear callbacks before the callee dies.
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

 private:
  /// full() without taking the lock. Caller must hold _mutex.
  [[nodiscard]] bool full_unlocked() const
  {
    if (_queue.size() >= _cfg.capacity_items) return true;
    if (_cfg.capacity_bytes > 0 && !_queue.empty() && _total_bytes >= _cfg.capacity_bytes)
      return true;
    return false;
  }

  config _cfg;
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<exchange_batch_handle> _queue;
  std::size_t _total_bytes{0};
  bool _closed{false};
  std::function<void()> _on_push;
  std::function<void()> _on_pop;
};

}  // namespace sirius::exec
