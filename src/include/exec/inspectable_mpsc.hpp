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
#include <deque>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
namespace sirius::exec {

/// Pop order for inspectable_mpsc::pop() / try_pop().
///   FIFO: first-in-first-out (default; preserves legacy queue behavior).
///   LIFO: last-in-first-out (newest item is popped first; useful as a depth-first scheduler
///         hint so freshly produced tasks are processed before older queued ones).
/// Note: push() / emplace() always append to the back; pop_if() / mutable_pop_if() honor their
/// own `front_to_back` argument and ignore this ordering.
enum class queue_ordering { FIFO, LIFO };

/// ADL-discoverable string conversion so yaml_reader can parse queue_ordering values.
inline bool string_to_enum(std::string_view sv, queue_ordering& out)
{
  static const std::unordered_map<std::string_view, queue_ordering> map = {
    {"fifo", queue_ordering::FIFO},
    {"lifo", queue_ordering::LIFO},
  };
  auto it = map.find(sv);
  if (it == map.end()) { return false; }
  out = it->second;
  return true;
}

inline bool enum_to_string(queue_ordering ordering, std::string& s)
{
  switch (ordering) {
    case queue_ordering::FIFO: s = "fifo"; return true;
    case queue_ordering::LIFO: s = "lifo"; return true;
  }
  return false;
}

template <typename T>
class inspectable_mpsc {
 private:
  std::deque<std::unique_ptr<T>> _queue;
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  bool _active{true};
  queue_ordering _ordering{queue_ordering::FIFO};

 public:
  inspectable_mpsc() = default;
  explicit inspectable_mpsc(queue_ordering ordering) : _ordering(ordering) {}
  ~inspectable_mpsc() = default;

  inspectable_mpsc(const inspectable_mpsc&)            = delete;
  inspectable_mpsc& operator=(const inspectable_mpsc&) = delete;
  inspectable_mpsc(inspectable_mpsc&&)                 = delete;
  inspectable_mpsc& operator=(inspectable_mpsc&&)      = delete;

  /**
   * \brief Pushes an item into the queue.
   * \return Returns false if the queue has been interrupted.
   */
  [[nodiscard]] bool push(std::unique_ptr<T> item)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (!_active) { return false; }
    _queue.push_back(std::move(item));
    lock.unlock();
    _cv.notify_one();
    return true;
  }

  /**
   * \brief Constructs an item in-place and enqueues it.
   * \return Returns false if the queue has been interrupted.
   */
  template <typename... Args>
  [[nodiscard]] bool emplace(Args&&... args)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (!_active) { return false; }
    _queue.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    lock.unlock();
    _cv.notify_one();
    return true;
  }

  /**
   * \brief Blocks waiting for an item.
   * \return Returns nullptr if the queue is interrupted and empty.
   *
   * If the queue is interrupted but still has items, those items are returned
   * before nullptr (drain remaining). Uses condition_variable::wait for true
   * blocking -- no polling. The end of the deque popped from depends on the
   * configured queue_ordering (front for FIFO, back for LIFO).
   */
  std::unique_ptr<T> pop()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return !_queue.empty() || !_active; });
    if (_queue.empty()) { return nullptr; }
    return pop_one_unlocked();
  }

  /**
   * \brief Attempts to pop without blocking.
   * \return Returns nullptr if the queue is empty.
   *
   * Honors the configured queue_ordering (front for FIFO, back for LIFO).
   */
  std::unique_ptr<T> try_pop()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_queue.empty()) { return nullptr; }
    return pop_one_unlocked();
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

  /// Returns the configured pop ordering. Not lock-guarded — the value is fixed at construction.
  [[nodiscard]] queue_ordering get_ordering() const noexcept { return _ordering; }

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
   * \param front_to_back If true, searches oldest-to-newest; if false,
   *        newest-to-oldest.
   * \return The matching element, or nullptr if no match found.
   *
   * Holds the mutex for the entire scan. Predicate should be lightweight.
   */
  std::unique_ptr<T> pop_if(std::function<bool(const T&)> predicate, bool front_to_back)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (front_to_back) {
      for (auto it = _queue.begin(); it != _queue.end(); ++it) {
        if (predicate(**it)) {
          auto item = std::move(*it);
          _queue.erase(it);
          return item;
        }
      }
    } else {
      for (auto rit = _queue.rbegin(); rit != _queue.rend(); ++rit) {
        if (predicate(**rit)) {
          auto item = std::move(*rit);
          _queue.erase(std::next(rit).base());
          return item;
        }
      }
    }
    return nullptr;
  }

  /**
   * \brief Removes and returns the first element matching the mutable
   *        predicate.
   * \param predicate Callable receiving T& (mutable) and returning bool.
   * \param front_to_back If true, searches oldest-to-newest; if false,
   *        newest-to-oldest.
   * \return The matching element, or nullptr if no match found.
   *
   * Same as pop_if but the predicate receives a mutable reference,
   * allowing state inspection that requires non-const access. Holds
   * the mutex for the full scan duration.
   */
  std::unique_ptr<T> mutable_pop_if(std::function<bool(T&)> predicate, bool front_to_back)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (front_to_back) {
      for (auto it = _queue.begin(); it != _queue.end(); ++it) {
        if (predicate(**it)) {
          auto item = std::move(*it);
          _queue.erase(it);
          return item;
        }
      }
    } else {
      for (auto rit = _queue.rbegin(); rit != _queue.rend(); ++rit) {
        if (predicate(**rit)) {
          auto item = std::move(*rit);
          _queue.erase(std::next(rit).base());
          return item;
        }
      }
    }
    return nullptr;
  }

 private:
  /// Pop a single element from the end of the deque selected by _ordering. Caller must hold
  /// _mutex and ensure the queue is non-empty.
  std::unique_ptr<T> pop_one_unlocked()
  {
    if (_ordering == queue_ordering::LIFO) {
      auto item = std::move(_queue.back());
      _queue.pop_back();
      return item;
    }
    auto item = std::move(_queue.front());
    _queue.pop_front();
    return item;
  }
};
}  // namespace sirius::exec
