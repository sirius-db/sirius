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

#include "interruptible_mpmc.hpp"

#include <memory>

namespace sirius::exec {

// Forward declaration
template <smart_pointer T>
class channel;

/**
 * @brief A publisher that can send items to a channel.
 *
 * Publishers hold a shared_ptr to the underlying queue, allowing multiple
 * publishers to send to the same channel. Publishers are lightweight and
 * can be copied freely.
 *
 * @tparam T The smart pointer type (must satisfy smart_pointer concept)
 */
template <smart_pointer T>
class publisher {
  friend class channel<T>;

 public:
  publisher(const publisher&)            = default;
  publisher& operator=(const publisher&) = default;
  publisher(publisher&&)                 = default;
  publisher& operator=(publisher&&)      = default;

  /**
   * @brief Send a value to the channel.
   *
   * @param value The value to send (will be moved)
   * @return true if the value was successfully sent, false if the channel is closed
   */
  [[nodiscard]] bool send(T value) { return _queue->push(std::move(value)); }

  /**
   * @brief Check if the channel is still open for sending.
   *
   * @return true if the channel is open, false otherwise
   */
  [[nodiscard]] bool is_open() const noexcept { return _queue->is_open(); }

 private:
  explicit publisher(std::shared_ptr<interruptible_mpmc<T>> queue) : _queue(std::move(queue)) {}

  std::shared_ptr<interruptible_mpmc<T>> _queue;
};

/**
 * @brief A channel for producer-consumer communication using smart pointers.
 *
 * The channel is the subscriber (consumer) side that provides get() and close()
 * methods. It can create multiple publishers that share access to the underlying
 * queue.
 *
 * @tparam T The smart pointer type (must satisfy smart_pointer concept)
 */
template <smart_pointer T>
class channel {
 public:
  channel() : _queue(std::make_shared<interruptible_mpmc<T>>()) {}

  // Non-copyable (only one consumer allowed)
  channel(const channel&)            = delete;
  channel& operator=(const channel&) = delete;

  // Movable
  channel(channel&&)            = default;
  channel& operator=(channel&&) = default;

  /**
   * @brief Get (receive) a value from the channel.
   *
   * Blocks until a value is available or the channel is closed.
   *
   * @return The received value, or nullptr if the channel is closed
   */
  T get() { return _queue->pop(); }

  /**
   * @brief Try to get a value without blocking.
   *
   * @return The received value, or nullptr if no value is available
   */
  T try_get() { return _queue->try_pop(); }

  /**
   * @brief Close the channel.
   *
   * After closing, get() will return nullptr and send() will return false.
   */
  void close() { _queue->interrupt(); }

  /**
   * @brief Check if the channel is still open.
   *
   * @return true if the channel is open, false otherwise
   */
  [[nodiscard]] bool is_open() const noexcept { return _queue->is_open(); }

  /**
   * @brief Create a new publisher for this channel.
   *
   * Multiple publishers can be created, all sharing the same underlying queue.
   *
   * @return A new publisher instance
   */
  [[nodiscard]] publisher<T> make_publisher() { return publisher<T>(_queue); }

 private:
  std::shared_ptr<interruptible_mpmc<T>> _queue;
};

}  // namespace sirius::exec
