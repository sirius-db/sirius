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

#include <atomic>
#include <exception>
#include <future>

namespace sirius::pipeline {

/**
 * @brief Handles query completion signaling with thread-safe state management.
 *
 * This class manages a promise/future pair for signaling query completion.
 * It ensures that the completion state is only set once using atomic operations,
 * allowing safe concurrent access from multiple executor threads.
 */
class completion_handler {
 public:
  completion_handler()  = default;
  ~completion_handler() = default;

  // Non-copyable and non-movable
  completion_handler(const completion_handler&)            = delete;
  completion_handler& operator=(const completion_handler&) = delete;
  completion_handler(completion_handler&&)                 = delete;
  completion_handler& operator=(completion_handler&&)      = delete;

  /**
   * @brief Report an error that occurred during query execution.
   *
   * Sets the exception on the promise. Only the first call has effect;
   * subsequent calls are ignored.
   *
   * @param error The exception pointer to report.
   */
  void report_error(std::exception_ptr error) noexcept
  {
    bool expected = false;
    if (_completed.compare_exchange_strong(expected, true)) {
      try {
        _promise.set_exception(error);
      } catch (...) {
        // Promise already satisfied or other error - ignore
      }
    }
  }

  /**
   * @brief Report an error that occurred during query execution.
   *
   * Sets the exception on the promise. Only the first call has effect;
   * subsequent calls are ignored.
   *
   * @param error The exception pointer to report.
   */
  void report_error(std::string_view error) noexcept
  {
    bool expected = false;
    if (_completed.compare_exchange_strong(expected, true)) {
      try {
        _promise.set_exception(std::make_exception_ptr(std::runtime_error(error.data())));
      } catch (...) {
        // Promise already satisfied or other error - ignore
      }
    }
  }

  /**
   * @brief Mark the query as successfully completed.
   *
   * Sets the promise value to signal completion. Only the first call has effect;
   * subsequent calls are ignored.
   */
  void mark_completed() noexcept
  {
    bool expected = false;
    if (_completed.compare_exchange_strong(expected, true)) {
      try {
        _promise.set_value();
      } catch (...) {
        // Promise already satisfied or other error - ignore
      }
    }
  }

  /**
   * @brief Get the future to await query completion.
   *
   * @return A future that will be satisfied when the query completes or errors.
   */
  [[nodiscard]] std::future<void> get_awaitable() { return _promise.get_future(); }

  /**
   * @brief Check if the handler has already been completed or errored.
   *
   * @return True if completion has been signaled, false otherwise.
   */
  [[nodiscard]] bool is_completed() const noexcept { return _completed.load(); }

 private:
  std::promise<void> _promise;
  std::atomic<bool> _completed{false};
};

}  // namespace sirius::pipeline
