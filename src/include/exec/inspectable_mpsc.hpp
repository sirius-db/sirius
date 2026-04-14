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
#include <memory>
#include <mutex>

namespace sirius::exec {

template <typename T>
class inspectable_mpsc {
 private:
  std::deque<std::unique_ptr<T>> _queue;
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  bool _active{true};

 public:
  inspectable_mpsc()  = default;
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
   * blocking -- no polling.
   */
  std::unique_ptr<T> pop()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return !_queue.empty() || !_active; });
    if (_queue.empty()) { return nullptr; }
    auto item = std::move(_queue.front());
    _queue.pop_front();
    return item;
  }

  /**
   * \brief Attempts to pop without blocking.
   * \return Returns nullptr if the queue is empty.
   */
  std::unique_ptr<T> try_pop()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_queue.empty()) { return nullptr; }
    auto item = std::move(_queue.front());
    _queue.pop_front();
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
  [[nodiscard]] bool is_open() const noexcept
  {
    std::unique_lock<std::mutex> lock(_mutex);
    return _active;
  }

  /**
   * \brief Returns true if the queue contains no items.
   */
  [[nodiscard]] bool is_empty() const noexcept
  {
    std::unique_lock<std::mutex> lock(_mutex);
    return _queue.empty();
  }

  /**
   * \brief Returns the number of items currently in the queue.
   */
  [[nodiscard]] std::size_t size() const noexcept
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
  std::unique_ptr<T> pop_if(std::function<bool(const T&)> predicate,
                             bool front_to_back)
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
   * \brief Returns a pointer to the first element matching the predicate
   *        without removing it.
   * \param predicate Callable receiving const T& and returning bool.
   * \param front_to_back If true, searches oldest-to-newest; if false,
   *        newest-to-oldest.
   * \return Raw pointer to the matching element, or nullptr if no match.
   *
   * The returned pointer is valid only while the caller holds no other
   * mutating reference. It is invalidated by any subsequent pop(),
   * pop_if(), drain(), or other mutating queue operation. Safe under
   * MPSC: only the single consumer calls inspection methods.
   */
  T* get_if(std::function<bool(const T&)> predicate, bool front_to_back)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (front_to_back) {
      for (auto it = _queue.begin(); it != _queue.end(); ++it) {
        if (predicate(**it)) { return it->get(); }
      }
    } else {
      for (auto rit = _queue.rbegin(); rit != _queue.rend(); ++rit) {
        if (predicate(**rit)) { return rit->get(); }
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
  std::unique_ptr<T> mutable_pop_if(std::function<bool(T&)> predicate,
                                     bool front_to_back)
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
   * \brief Returns a pointer to the first element matching the mutable
   *        predicate without removing it.
   * \param predicate Callable receiving T& (mutable) and returning bool.
   * \param front_to_back If true, searches oldest-to-newest; if false,
   *        newest-to-oldest.
   * \return Raw pointer to the matching element, or nullptr if no match.
   *
   * Same as get_if but the predicate receives a mutable reference.
   * Returned pointer is invalidated by any subsequent pop(), pop_if(),
   * drain(), or other mutating queue operation. Safe under MPSC
   * single-consumer.
   */
  T* mutable_get_if(std::function<bool(T&)> predicate,
                     bool front_to_back)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (front_to_back) {
      for (auto it = _queue.begin(); it != _queue.end(); ++it) {
        if (predicate(**it)) { return it->get(); }
      }
    } else {
      for (auto rit = _queue.rbegin(); rit != _queue.rend(); ++rit) {
        if (predicate(**rit)) { return rit->get(); }
      }
    }
    return nullptr;
  }
};

}  // namespace sirius::exec
