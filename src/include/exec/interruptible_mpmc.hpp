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

#include <blockingconcurrentqueue.h>

#include <atomic>
#include <concepts>
#include <memory>
#include <optional>

namespace sirius::exec {

// Type trait to detect std::shared_ptr
template <typename T>
struct is_shared_ptr : std::false_type {};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

// Type trait to detect std::unique_ptr
template <typename T>
struct is_unique_ptr : std::false_type {};

template <typename T, typename D>
struct is_unique_ptr<std::unique_ptr<T, D>> : std::true_type {};

// Concept requiring T to be either shared_ptr or unique_ptr
template <typename T>
concept smart_pointer = is_shared_ptr<T>::value || is_unique_ptr<T>::value;

template <smart_pointer T>
class interruptible_mpmc {
  using value_type   = typename T::element_type;
  using pointer_type = T;

 private:
  // The underlying high-performance queue
  duckdb_moodycamel::BlockingConcurrentQueue<pointer_type> queue;

  // Atomic flag to manage the shutdown state
  std::atomic<bool> _is_active{true};

 public:
  interruptible_mpmc() = default;
  // Delete copy/move to prevent unsafe duplication of the internal queue
  interruptible_mpmc(const interruptible_mpmc&)            = delete;
  interruptible_mpmc& operator=(const interruptible_mpmc&) = delete;

  [[nodiscard]] bool is_open() const noexcept { return _is_active.load(std::memory_order_relaxed); }

  /**
   * \brief Pushes an item into the queue.
   * \return Returns false if the queue has been stopped/interrupted.
   */
  template <typename... Args>
  [[nodiscard]] bool emplace(Args&&... args)
  {
    if (!_is_active.load(std::memory_order_relaxed)) { return false; }
    queue.enqueue(std::make_unique<value_type>(std::forward<Args>(args)...));
    return true;
  }

  bool push(pointer_type item)
  {
    assert(item != nullptr);
    if (!_is_active.load(std::memory_order_relaxed)) { return false; }
    queue.enqueue(std::move(item));
    return true;
  }

  /**
   * \brief Blocks waiting for an item.
   * \return Returns std::nullopt if the queue is interrupted (shutdown).
   */
  pointer_type pop()
  {
    pointer_type item = nullptr;
    while (_is_active.load(std::memory_order_relaxed)) {
      if (queue.wait_dequeue_timed(item, 10000)) { return std::move(item); }
    }
    return nullptr;
  }

  /**
   * \brief Attempts to pop without blocking.
   * \return Returns nullptr if the queue is empty.
   */
  pointer_type try_pop()
  {
    pointer_type item = nullptr;
    if (queue.try_dequeue(item)) { return std::move(item); }
    return nullptr;
  }

  /**
   * Interrupts the queue.
   * \brief Sets the active flag to false.
   * Consumer threads will see this flag on their next loop cycle (max 10ms delay).
   */
  void interrupt() { _is_active.store(false); }

  void drain()
  {
    pointer_type item = nullptr;
    while (queue.try_dequeue(item)) {}
  }

  /**
   * Resets the queue state to active (useful for restarting workers).
   */
  void reset() { _is_active.store(true, std::memory_order_relaxed); }
};

}  // namespace sirius::exec
