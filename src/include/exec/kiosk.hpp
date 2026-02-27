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

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <utility>

namespace sirius::exec {

class kiosk;

class ticket {
 public:
  ticket() = default;
  ticket(ticket&& other) noexcept;
  ticket& operator=(ticket&& other) noexcept;
  ~ticket();

  // Non-copyable
  ticket(const ticket&)            = delete;
  ticket& operator=(const ticket&) = delete;

  [[nodiscard]] bool is_valid() const noexcept { return agent_ != nullptr; }

  // Check if ticket is valid
  explicit operator bool() const noexcept { return agent_ != nullptr; }

  // Manually release the ticket
  void release();

 private:
  friend class kiosk;
  explicit ticket(kiosk& agent);

  kiosk* agent_{nullptr};
};

class kiosk {
 public:
  /// \brief Create unbounded kiosk (no max concurrent tickets)
  kiosk();

  /// \brief Create bounded kiosk with max concurrent tickets
  kiosk(size_t max_tickets);

  ~kiosk() = default;

  void stop()
  {
    std::lock_guard lock(mutex_);
    stopped_ = true;
    cv_.notify_all();
  }

  // Blocking: acquire a ticket (waits if bounded and at capacity)
  [[nodiscard]] ticket acquire();

  // Non-blocking: try to acquire a ticket
  [[nodiscard]] std::optional<ticket> try_acquire();

  // Try to acquire with timeout
  template <typename Rep, typename Period>
  [[nodiscard]] std::optional<ticket> try_acquire_for(
    const std::chrono::duration<Rep, Period>& timeout)
  {
    std::unique_lock lock(mutex_);

    if (max_tickets_ == 0) {
      // Unbounded mode
      ++active_tickets_;
      ++total_issued_;
      return ticket(*this);
    }

    // Bounded mode - wait with timeout
    if (cv_.wait_for(lock, timeout, [this] { return active_tickets_ < max_tickets_; })) {
      ++active_tickets_;
      ++total_issued_;
      return ticket(*this);
    }

    return std::nullopt;
  }

  // Wait for all tickets to be released/checked out
  void wait_all();

  // Wait for all tickets with timeout
  template <typename Rep, typename Period>
  bool wait_all_for(const std::chrono::duration<Rep, Period>& timeout)
  {
    std::unique_lock lock(mutex_);
    return wait_cv_.wait_for(lock, timeout, [this] { return active_tickets_ == 0; });
  }

  // Query current state
  size_t active_count() const;
  size_t total_issued() const;
  size_t max_capacity() const;
  bool is_bounded() const;

 private:
  friend class ticket;

  void release_ticket();

  mutable std::mutex mutex_;
  std::condition_variable cv_;       // For acquire waiting
  std::condition_variable wait_cv_;  // For wait_all

  bool stopped_{false};
  const size_t max_tickets_;  // 0 = unbounded
  size_t active_tickets_{0};
  size_t total_issued_{0};
};

inline ticket::ticket(kiosk& agent) : agent_(&agent) {}

inline ticket::ticket(ticket&& other) noexcept : agent_(other.agent_) { other.agent_ = nullptr; }

inline ticket& ticket::operator=(ticket&& other) noexcept
{
  if (this != &other) {
    release();
    agent_       = other.agent_;
    other.agent_ = nullptr;
  }
  return *this;
}

inline ticket::~ticket() { release(); }

inline void ticket::release()
{
  if (auto agent = std::exchange(this->agent_, nullptr); agent != nullptr) {
    agent->release_ticket();
  }
}

inline kiosk::kiosk() : max_tickets_(0) {}

inline kiosk::kiosk(size_t max_tickets) : max_tickets_(max_tickets) {}

inline ticket kiosk::acquire()
{
  std::unique_lock lock(mutex_);
  if (stopped_) return ticket();

  if (max_tickets_ == 0) {
    // Unbounded mode - always succeed
    ++active_tickets_;
    ++total_issued_;
    return ticket(*this);
  }

  // Bounded mode - wait until ticket available
  cv_.wait(lock, [this] { return active_tickets_ < max_tickets_ || stopped_; });

  if (stopped_) return ticket();
  ++total_issued_;
  ++active_tickets_;
  return ticket(*this);
}

inline std::optional<ticket> kiosk::try_acquire()
{
  std::unique_lock lock(mutex_);

  if (max_tickets_ == 0) {
    // Unbounded mode - always succeed
    ++active_tickets_;
    ++total_issued_;
    return ticket(*this);
  }

  // Bounded mode - only succeed if tickets available
  if (active_tickets_ < max_tickets_) {
    ++active_tickets_;
    ++total_issued_;
    return ticket(*this);
  }

  return std::nullopt;
}

inline void kiosk::wait_all()
{
  std::unique_lock lock(mutex_);
  wait_cv_.wait(lock, [this] { return active_tickets_ == 0; });
}

inline size_t kiosk::active_count() const
{
  std::lock_guard lock(mutex_);
  return active_tickets_;
}

inline size_t kiosk::total_issued() const
{
  std::lock_guard lock(mutex_);
  return total_issued_;
}

inline size_t kiosk::max_capacity() const { return max_tickets_; }

inline bool kiosk::is_bounded() const { return max_tickets_ > 0; }

inline void kiosk::release_ticket()
{
  std::unique_lock lock(mutex_);

  if (active_tickets_ == 0) {
    return;  // Already released
  }

  --active_tickets_;

  // Notify waiters
  if (max_tickets_ > 0 && active_tickets_ < max_tickets_) {
    cv_.notify_one();  // Notify one waiting acquire()
  }

  if (active_tickets_ == 0) {
    wait_cv_.notify_all();  // Notify all waiting for completion
  }
}

}  // namespace sirius::exec
