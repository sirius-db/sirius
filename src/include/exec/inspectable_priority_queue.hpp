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

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <vector>

namespace sirius::exec {

/// The priority type used to order items in inspectable_priority_queue. Smaller values are
/// scheduled first (popped from the front first), so a task's priority correlates with its
/// execution order. Signed so callers can express "lower than default" priorities without
/// underflow tricks.
using queue_priority = std::int64_t;

/**
 * @brief A thread-safe, inspectable priority queue backed by a contiguous vector.
 *
 * Drop-in replacement for inspectable_mpsc that orders items by an explicit priority rather
 * than by insertion order. The backing storage is a std::vector kept sorted by priority in
 * ascending order (lowest priority value at the front), so lower-numbered tasks run first:
 *   - pop() / try_pop() / pop_front() return the lowest-priority (first-to-run) item,
 *   - pop_back() returns the highest-priority (last-to-run) item.
 * Ties (equal priority) preserve insertion order (FIFO among equals), matching the legacy
 * queue's behavior when every item shares the default priority.
 *
 * The priority of an item is obtained through a caller-supplied extractor at construction
 * time. When no extractor is provided every item gets priority 0, which reduces to a plain
 * FIFO queue. Priority is captured once at push time and never re-read, so mutating an item's
 * priority after it is enqueued has no effect on its position.
 *
 * A vector backing (rather than a heap) is deliberate: pop_if() / mutable_pop_if() and the
 * downgrade inspector scan the whole queue, and callers benefit from cheap in-order iteration.
 */
template <typename T>
class inspectable_priority_queue {
 public:
  /// Extracts the priority of an item. Returning a smaller value ranks the item closer to the
  /// front (popped sooner / runs earlier).
  using priority_fn = std::function<queue_priority(const T&)>;

 private:
  struct entry {
    queue_priority priority;
    std::unique_ptr<T> item;
  };

  std::vector<entry> _queue;  ///< Sorted by priority ascending; ties keep insertion order.
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  bool _active{true};
  priority_fn _priority_fn;

 public:
  inspectable_priority_queue() : _priority_fn([](const T&) -> queue_priority { return 0; }) {}
  explicit inspectable_priority_queue(priority_fn fn) : _priority_fn(std::move(fn))
  {
    if (!_priority_fn) {
      _priority_fn = [](const T&) -> queue_priority { return 0; };
    }
  }
  ~inspectable_priority_queue() = default;

  inspectable_priority_queue(const inspectable_priority_queue&)            = delete;
  inspectable_priority_queue& operator=(const inspectable_priority_queue&) = delete;
  inspectable_priority_queue(inspectable_priority_queue&&)                 = delete;
  inspectable_priority_queue& operator=(inspectable_priority_queue&&)      = delete;

  /**
   * \brief Pushes an item into the queue at its priority-sorted position.
   * \return Returns false if the queue has been interrupted.
   */
  [[nodiscard]] bool push(std::unique_ptr<T> item)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (!_active) { return false; }
    // Compute priority before the move: argument evaluation order is unspecified, so passing
    // _priority_fn(*item) and std::move(item) to one call risks reading a moved-from pointer.
    const queue_priority priority = _priority_fn(*item);
    insert_unlocked(priority, std::move(item));
    lock.unlock();
    _cv.notify_one();
    return true;
  }

  /**
   * \brief Constructs an item in-place and enqueues it at its priority-sorted position.
   * \return Returns false if the queue has been interrupted.
   */
  template <typename... Args>
  [[nodiscard]] bool emplace(Args&&... args)
  {
    auto item = std::make_unique<T>(std::forward<Args>(args)...);
    std::unique_lock<std::mutex> lock(_mutex);
    if (!_active) { return false; }
    const queue_priority priority = _priority_fn(*item);
    insert_unlocked(priority, std::move(item));
    lock.unlock();
    _cv.notify_one();
    return true;
  }

  /**
   * \brief Blocks waiting for the lowest-priority (first-to-run) item.
   * \return Returns nullptr if the queue is interrupted and empty.
   *
   * If the queue is interrupted but still has items, those items are returned (lowest
   * priority first) before nullptr. Uses condition_variable::wait for true blocking.
   */
  std::unique_ptr<T> pop()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return !_queue.empty() || !_active; });
    if (_queue.empty()) { return nullptr; }
    return pop_front_unlocked();
  }

  /**
   * \brief Attempts to pop the lowest-priority (first-to-run) item without blocking.
   * \return Returns nullptr if the queue is empty.
   */
  std::unique_ptr<T> try_pop()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_queue.empty()) { return nullptr; }
    return pop_front_unlocked();
  }

  /**
   * \brief Removes and returns the lowest-priority (first-to-run) item, or nullptr if empty.
   *        Non-blocking.
   */
  std::unique_ptr<T> pop_front() { return try_pop(); }

  /**
   * \brief Removes and returns the highest-priority (last-to-run) item, or nullptr if empty.
   *        Non-blocking.
   */
  std::unique_ptr<T> pop_back()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_queue.empty()) { return nullptr; }
    auto item = std::move(_queue.back().item);
    _queue.pop_back();
    return item;
  }

  /**
   * \brief Interrupts the queue, causing blocked pop() calls to return nullptr.
   */
  void interrupt()
  {
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _active = false;
    }
    _cv.notify_all();
  }

  /**
   * \brief Reactivates the queue after an interrupt, restoring normal operation.
   */
  void reactivate()
  {
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _active = true;
    }
    _cv.notify_all();
  }

  /**
   * \brief Removes all queued items.
   */
  void drain()
  {
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _queue.clear();
    }
    _cv.notify_all();
  }

  /**
   * \brief Returns true if the queue is active (not interrupted).
   */
  [[nodiscard]] bool is_open() const
  {
    std::unique_lock<std::mutex> lock(_mutex);
    return _active;
  }

  /**
   * \brief Returns true if the queue contains no items.
   */
  [[nodiscard]] bool is_empty() const
  {
    std::unique_lock<std::mutex> lock(_mutex);
    return _queue.empty();
  }

  /**
   * \brief Returns the number of items currently in the queue.
   */
  [[nodiscard]] std::size_t size() const
  {
    std::unique_lock<std::mutex> lock(_mutex);
    return _queue.size();
  }

  /**
   * \brief Removes and returns the first element matching the predicate.
   * \param predicate Callable receiving const T& and returning bool.
   * \param front_to_back If true, searches lowest-to-highest priority (first-to-run first); if
   *        false, highest-to-lowest.
   * \return The matching element, or nullptr if no match found.
   *
   * Holds the mutex for the entire scan. Predicate should be lightweight.
   */
  std::unique_ptr<T> pop_if(std::function<bool(const T&)> predicate, bool front_to_back)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    return pop_if_unlocked([&](T& t) { return predicate(t); }, front_to_back);
  }

  /**
   * \brief Removes and returns the first element matching the mutable predicate.
   * \param predicate Callable receiving T& (mutable) and returning bool.
   * \param front_to_back If true, searches lowest-to-highest priority (first-to-run first); if
   *        false, highest-to-lowest.
   * \return The matching element, or nullptr if no match found.
   */
  std::unique_ptr<T> mutable_pop_if(std::function<bool(T&)> predicate, bool front_to_back)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    return pop_if_unlocked([&](T& t) { return predicate(t); }, front_to_back);
  }

 private:
  /// Insert item keeping the vector sorted by priority ascending, with FIFO order among
  /// equal priorities. Caller must hold _mutex.
  void insert_unlocked(queue_priority priority, std::unique_ptr<T> item)
  {
    // Find the first entry with strictly higher priority; insert before it so all
    // equal-priority entries pushed earlier stay ahead (FIFO among equals).
    auto pos = std::find_if(
      _queue.begin(), _queue.end(), [priority](const entry& e) { return e.priority > priority; });
    _queue.insert(pos, entry{priority, std::move(item)});
  }

  /// Pop the front (lowest-priority, first-to-run) element. Caller must hold _mutex and ensure
  /// non-empty.
  std::unique_ptr<T> pop_front_unlocked()
  {
    auto item = std::move(_queue.front().item);
    _queue.erase(_queue.begin());
    return item;
  }

  template <typename Pred>
  std::unique_ptr<T> pop_if_unlocked(Pred predicate, bool front_to_back)
  {
    if (front_to_back) {
      for (auto it = _queue.begin(); it != _queue.end(); ++it) {
        if (predicate(*it->item)) {
          auto item = std::move(it->item);
          _queue.erase(it);
          return item;
        }
      }
    } else {
      for (auto rit = _queue.rbegin(); rit != _queue.rend(); ++rit) {
        if (predicate(*rit->item)) {
          auto item = std::move(rit->item);
          _queue.erase(std::next(rit).base());
          return item;
        }
      }
    }
    return nullptr;
  }
};

}  // namespace sirius::exec
