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

#include "exec/work_tracker.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace sirius::pipeline {

/**
 * @brief Thread-safe query completion signaling and work accounting.
 *
 * The promise signals a result; the work tracker establishes teardown-safe quiescence.
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
   * @brief Report an exception.
   *
   * The first completion signal satisfies the promise; a losing error is retained for
   * take_late_error().
   *
   * @param error The exception pointer to report.
   * @return True if the future will carry this error, false if it was preserved instead.
   */
  bool report_error(std::exception_ptr error) noexcept
  {
    bool expected = false;
    if (_completed.compare_exchange_strong(expected, true)) {
      try {
        _has_error.store(true);
        _promise.set_exception(error);
      } catch (...) {
        // Promise already satisfied or other error - ignore
      }
      return true;
    }
    keep_late_error(std::move(error));
    return false;
  }

  /**
   * @brief Report an error message with the same completion-race semantics.
   *
   * @param error The message to report.
   * @return True if the future will carry this error, false if it was preserved instead.
   */
  bool report_error(std::string_view error) noexcept
  {
    // Build the exception before racing the CAS: error.data() need not be null-terminated,
    // and a construction failure after winning would leave the future pending forever.
    std::exception_ptr exception;
    try {
      exception = std::make_exception_ptr(std::runtime_error(std::string(error)));
    } catch (...) {
      exception = std::current_exception();
    }
    return report_error(std::move(exception));
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

  /**
   * @brief Check if the handler has already been completed with an error.
   *
   * @return True if an error has been reported, false otherwise.
   */
  [[nodiscard]] bool has_error() const noexcept { return _has_error.load(); }

  /**
   * @brief Take the error that lost the completion race, if any.
   *
   * Success and failure share one CAS. Call this only after all potential reporters have
   * released their work slots or been joined.
   *
   * @return The preserved exception, or nullptr. The call clears it, so it is delivered once.
   */
  [[nodiscard]] std::exception_ptr take_late_error() noexcept
  {
    std::lock_guard<std::mutex> lock(_late_error_mutex);
    auto error  = std::move(_late_error);
    _late_error = nullptr;
    return error;
  }

  /**
   * @brief Count one new unit of this query's work.
   *
   * The returned slot must cover the full lifetime of the work it counts. Returns an empty slot
   * after close_work(); self-stamping producers must then abort.
   */
  [[nodiscard]] exec::work_tracker::slot acquire_work() { return _work.acquire(); }

  /**
   * @brief Permanently reject new work slots.
   */
  void close_work() { _work.close(); }

  /**
   * @brief Block until every counted unit of work has been destroyed, or @p timeout elapses.
   *
   * @return True if all counted query work has been destroyed.
   */
  [[nodiscard]] bool wait_quiescent(std::chrono::milliseconds timeout)
  {
    return _work.wait_quiescent(timeout);
  }

  /**
   * @brief Return a snapshot of the number of live work slots.
   */
  [[nodiscard]] std::size_t outstanding_work() const { return _work.outstanding(); }

 private:
  //! Preserve the first error that loses the completion race.
  void keep_late_error(std::exception_ptr error) noexcept
  {
    if (!error) { return; }
    try {
      std::lock_guard<std::mutex> lock(_late_error_mutex);
      if (!_late_error) { _late_error = std::move(error); }
    } catch (...) {
      // Ignore - this reporter is noexcept
      return;
    }
    // Set even though the future carries a value: the reschedule path reads this to stop
    // retrying work for a query that has already failed.
    _has_error.store(true);
  }

  std::promise<void> _promise;
  std::atomic<bool> _completed{false};
  std::atomic<bool> _has_error{false};
  mutable std::mutex _late_error_mutex;
  std::exception_ptr _late_error;
  exec::work_tracker _work;
};

}  // namespace sirius::pipeline
