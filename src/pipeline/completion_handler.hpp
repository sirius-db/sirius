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

#include "transparent/replay_admission.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>

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
  completion_handler() = default;
  explicit completion_handler(std::shared_ptr<std::atomic<uint64_t>> tasks)
    : tasks_started_(std::move(tasks))
  {
  }
  void record_task_started() noexcept
  {
    if (tasks_started_) tasks_started_->fetch_add(1, std::memory_order_relaxed);
  }
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
  void report_error(
    std::exception_ptr error,
    transparent::late_failure_cause fallback = transparent::late_failure_cause::other) noexcept
  {
    bool expected = false;
    if (_completed.compare_exchange_strong(expected, true)) {
      try {
        {
          std::lock_guard lock(failure_mutex_);
          // Save only the terminal winner, before waking the query thread. Allocation failure
          // in diagnostics must never prevent delivery of the original error.
          try {
            failure_ = transparent::classify_failure(error, fallback);
          } catch (...) {
            failure_.cause = fallback;
          }
        }
        {
          std::lock_guard lock(publication_mutex_);
          _has_error.store(true);
        }
        published_changed_.notify_all();
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
    try {
      report_error(std::make_exception_ptr(std::runtime_error(std::string(error))));
    } catch (...) {
      report_error(std::current_exception());
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

  /**
   * @brief Check if the handler has already been completed with an error.
   *
   * @return True if an error has been reported, false otherwise.
   */
  [[nodiscard]] bool has_error() const noexcept { return _has_error.load(); }

  // Test rendezvous: publication, footer release, and terminal failure share one predicate lock.
  void record_publication_for_testing()
  {
    std::lock_guard lock(publication_mutex_);
    ++publications_;
    published_changed_.notify_all();
  }
  bool wait_for_publication_for_testing(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock(publication_mutex_);
    return published_changed_.wait_for(lock, timeout, [&] {
      return publications_ > 0 || has_error();
    }) && publications_ > 0;
  }
  void release_footer_for_testing(uint64_t file)
  {
    std::lock_guard lock(publication_mutex_);
    released_footer_ = file;
    published_changed_.notify_all();
  }
  void hold_footer_for_testing(uint64_t file)
  {
    std::unique_lock lock(publication_mutex_);
    if (!published_changed_.wait_for(
          lock, std::chrono::seconds(20), [&] { return released_footer_ == file || has_error(); }))
      throw std::runtime_error("footer publication rendezvous timed out");
  }
  [[nodiscard]] transparent::failure_cause failure() const
  {
    std::lock_guard lock(failure_mutex_);
    return failure_;
  }
  std::shared_ptr<op::scan::test_injections const> injections;
  std::atomic<uint64_t> injected_oom_attempts{0};
  std::atomic<uint64_t> injected_launch_attempts{0};
  bool non_rollbackable_state = false;  // Latched by the query thread before task submission.

 private:
  std::mutex publication_mutex_;
  std::condition_variable published_changed_;
  uint64_t publications_    = 0;
  uint64_t released_footer_ = 0;
  mutable std::mutex failure_mutex_;
  transparent::failure_cause failure_;
  std::shared_ptr<std::atomic<uint64_t>> tasks_started_;
  std::promise<void> _promise;
  std::atomic<bool> _completed{false};
  std::atomic<bool> _has_error{false};
};

}  // namespace sirius::pipeline
