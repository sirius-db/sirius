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

#include "scan_manager/preparation.hpp"
#include "transparent/replay_admission.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

namespace sirius::pipeline {

/**
 * @brief Handles query completion signaling with thread-safe state management.
 *
 * This class manages a promise/future pair for signaling query completion.
 * Terminal decisions are serialized; atomic flags allow executor threads to check
 * completion without taking the terminal lock.
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
    finish_error(std::move(error), fallback, nullptr);
  }

  void report_error(scan_manager::preparation_failure const& error) noexcept
  {
    try {
      error.validate();
      auto original = error.original;
      if (!original)
        original = std::make_exception_ptr(
          transparent::classified_execution_error(error.cause, error.detail));
      finish_error(std::move(original), error.cause, &error);
    } catch (...) {
      report_error(std::current_exception());
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

  // Register once before any worker/GPU submission. Late registration cannot retract success.
  void set_preparation_stop_callback(std::function<void(bool)> callback)
  {
    auto observer = std::make_shared<std::function<void(bool)> const>(std::move(callback));
    bool terminal = false, error = false;
    {
      std::lock_guard lock(terminal_mutex_);
      if (preparation_stop_) throw std::logic_error("preparation observer already bound");
      preparation_stop_ = observer;
      terminal          = gpu_done_ || _completed.load();
      error             = _has_error.load();
    }
    if (terminal) (*observer)(error);
  }
  void begin_preparation()
  {
    std::lock_guard lock(terminal_mutex_);
    if (preparation_active_ || gpu_done_ || _completed.load())
      throw std::logic_error("preparation must register before query work starts");
    preparation_active_    = true;
    inputs_closed_         = false;
    preparation_quiescent_ = false;
  }
  void close_preparation_inputs() noexcept
  {
    std::lock_guard lock(terminal_mutex_);
    inputs_closed_ = true;
    maybe_complete();
  }
  void preparation_quiescent() noexcept
  {
    std::lock_guard lock(terminal_mutex_);
    preparation_quiescent_ = true;
    maybe_complete();
  }
  /**
   * @brief Mark GPU work completed; registered preparation must also finish for success.
   *
   * Sets the promise value to signal completion. Only the first call has effect;
   * subsequent calls are ignored.
   */
  void mark_completed() noexcept
  {
    std::shared_ptr<std::function<void(bool)> const> stop;
    {
      std::lock_guard lock(terminal_mutex_);
      gpu_done_ = true;
      stop      = preparation_stop_;
      maybe_complete();
    }
    if (stop) {
      try {
        (*stop)(false);
      } catch (...) {
        report_error(std::current_exception());
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
  bool wait_for_error_for_testing(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock(publication_mutex_);
    return published_changed_.wait_for(lock, timeout, [&] { return has_error(); });
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
  [[nodiscard]] std::optional<op::scan::verdict_reason> failure_reason() const
  {
    std::lock_guard lock(failure_mutex_);
    return failure_reason_;
  }
  std::shared_ptr<op::scan::test_injections const> injections;
  std::atomic<uint64_t> injected_oom_attempts{0};
  std::atomic<uint64_t> injected_launch_attempts{0};
  std::thread::id execution_owner;      // Set by execute before scan preparation is armed.
  bool non_rollbackable_state = false;  // Latched by the query thread before task submission.

 private:
  // terminal -> failure -> publication. Never hold these locks while draining workers.
  void finish_error(std::exception_ptr error,
                    transparent::late_failure_cause fallback,
                    scan_manager::preparation_failure const* preparation) noexcept
  {
    std::unique_lock terminal_lock(terminal_mutex_);
    if (_completed.load()) return;
    {
      std::lock_guard lock(failure_mutex_);
      // Save only the terminal winner, before waking the query thread. Allocation failure
      // in diagnostics must never prevent delivery of the original error.
      failure_reason_ = preparation ? preparation->reason : std::nullopt;
      try {
        failure_ = preparation ? scan_manager::classify_failure(*preparation)
                               : transparent::classify_failure(error, fallback);
      } catch (...) {
        failure_.cause = fallback;
      }
    }
    {
      std::lock_guard lock(publication_mutex_);
      _has_error.store(true);
    }
    _completed.store(true);
    published_changed_.notify_all();
    try {
      _promise.set_exception(error);
    } catch (...) {
    }
    auto stop = preparation_stop_;
    terminal_lock.unlock();
    if (stop) {
      try {
        (*stop)(true);
      } catch (...) {
      }
    }
  }
  // Called with terminal_mutex_. Errors and cancellation do not wait for preparation closure.
  void maybe_complete() noexcept
  {
    if (_completed.load() || !gpu_done_ ||
        (preparation_active_ && (!inputs_closed_ || !preparation_quiescent_)))
      return;
    _completed.store(true);
    try {
      _promise.set_value();
    } catch (...) {
    }
  }
  std::shared_ptr<std::function<void(bool)> const> preparation_stop_;
  std::mutex terminal_mutex_;
  bool gpu_done_              = false;
  bool preparation_active_    = false;
  bool inputs_closed_         = false;
  bool preparation_quiescent_ = false;
  std::mutex publication_mutex_;
  std::condition_variable published_changed_;
  uint64_t publications_    = 0;
  uint64_t released_footer_ = 0;
  mutable std::mutex failure_mutex_;
  transparent::failure_cause failure_;
  std::optional<op::scan::verdict_reason> failure_reason_;
  std::shared_ptr<std::atomic<uint64_t>> tasks_started_;
  std::promise<void> _promise;
  std::atomic<bool> _completed{false};
  std::atomic<bool> _has_error{false};
};

}  // namespace sirius::pipeline
