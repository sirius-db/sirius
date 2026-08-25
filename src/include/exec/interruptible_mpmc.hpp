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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

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

  // Pushes that passed the active check but have not finished enqueueing.
  std::atomic<std::size_t> _pending_pushes{0};

  // Backstop poll interval (us) for wait_dequeue_timed. interrupt() wakes consumers directly, so
  // this only bounds the (not expected) case of a missed sentinel.
  static constexpr std::int64_t kPollBackstopUs = 10000;
  // Number of null sentinels enqueued per interrupt, one per potential blocked consumer.
  static constexpr int kWakeSentinels = 4;

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
    return push(std::make_unique<value_type>(std::forward<Args>(args)...));
  }

  bool push(pointer_type item)
  {
    assert(item != nullptr);
    // Register before checking the flag so interrupt() can wait for this enqueue.
    _pending_pushes.fetch_add(1, std::memory_order_seq_cst);
    if (!_is_active.load(std::memory_order_seq_cst)) {
      _pending_pushes.fetch_sub(1, std::memory_order_release);
      return false;
    }
    // Preserve enqueue allocation failures as rejected pushes.
    const bool enqueued = queue.enqueue(std::move(item));
    _pending_pushes.fetch_sub(1, std::memory_order_release);
    return enqueued;
  }

  /**
   * \brief Blocks waiting for an item.
   * \return Returns std::nullopt if the queue is interrupted (shutdown).
   */
  pointer_type pop()
  {
    pointer_type item = nullptr;
    while (_is_active.load(std::memory_order_relaxed)) {
      // A null item is the interrupt sentinel enqueued by interrupt(); treat it as "interrupted"
      // rather than as work. The timeout remains as a backstop for any missed wake-up.
      if (queue.wait_dequeue_timed(item, kPollBackstopUs)) {
        if (item == nullptr) { return nullptr; }
        return std::move(item);
      }
    }
    return nullptr;
  }

  /**
   * \brief Attempts to pop without blocking.
   * \return Returns nullptr if the queue is empty.
   *
   * Interrupt sentinels are skipped, not returned: try_pop reports work or
   * emptiness, never interruption — otherwise a real item queued behind a
   * sentinel would be unreachable to drain/cancel loops.
   */
  pointer_type try_pop()
  {
    pointer_type item = nullptr;
    while (queue.try_dequeue(item)) {
      if (item != nullptr) { return std::move(item); }
    }
    return nullptr;
  }

  /**
   * \brief Clears the active flag AND wakes any blocked consumer immediately.
   *
   * Null sentinels wake blocked consumers immediately; pop() treats them as interruption.
   *
   * This is also a producer barrier: all pushes that observed an active queue finish before
   * return. A subsequent drain therefore remains empty until reactivate().
   */
  void interrupt()
  {
    _is_active.store(false, std::memory_order_seq_cst);
    // This load must join the seq_cst order: acquire alone could miss an accepted producer's
    // increment. Observing its release decrement also makes the enqueue visible to drain().
    while (_pending_pushes.load(std::memory_order_seq_cst) != 0) {
      std::this_thread::yield();
    }
    for (int i = 0; i < kWakeSentinels; ++i) {
      queue.enqueue(pointer_type{});
    }
  }

  void drain()
  {
    pointer_type item = nullptr;
    while (queue.try_dequeue(item)) {}
  }

  /**
   * \brief Returns true if the queue is approximately empty.
   *
   * Uses size_approx() from the underlying concurrent queue, which may
   * transiently over- or under-count in the presence of concurrent producers
   * and consumers. Safe for assertions in quiescent states (e.g. after drain).
   */
  [[nodiscard]] bool is_empty() const noexcept { return queue.size_approx() == 0; }

  /**
   * Resets the queue state to active (useful for restarting workers).
   */
  void reactivate()
  {
    // Discard every unconsumed interrupt sentinel so it cannot be mistaken for an interrupt
    // after the queue is live again. try_dequeue does not honor FIFO across producer
    // sub-queues, so a single scan that stops at the first real item can leave sentinels
    // behind it — drain to exhaustion, then re-enqueue the surviving real items.
    std::vector<pointer_type> kept;
    pointer_type item = nullptr;
    while (queue.try_dequeue(item)) {
      if (item != nullptr) { kept.push_back(std::move(item)); }
    }
    for (auto& survivor : kept) {
      queue.enqueue(std::move(survivor));
    }
    _is_active.store(true, std::memory_order_relaxed);
  }
};

}  // namespace sirius::exec
