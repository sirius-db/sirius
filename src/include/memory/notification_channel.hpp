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
#include <memory>
#include <mutex>
#include <utility>
namespace cucascade {
namespace memory {

struct notification_channel : std::enable_shared_from_this<notification_channel> {
  struct event_notifier {
    explicit event_notifier(notification_channel& channel) : _channel(channel.shared_from_this()) {}

    ~event_notifier() { _channel->release_notifier(); }

    void post() { _channel->notify(); }

   private:
    std::shared_ptr<notification_channel> _channel;
  };

  enum class wait_status { IDLE, NOTIFIED, SHUTDOWN };

  ~notification_channel() { shutdown(); }

  wait_status wait()
  {
    std::unique_lock lock(mutex_);
    bool notified = false;
    cv_.wait(lock, [&, self = shared_from_this()] {
      notified = std::exchange(has_been_notified_, false);
      return notified || (n_active_notifiers_ == 0) || not is_running_;
    });
    return !is_running_ ? wait_status::SHUTDOWN
           : (notified) ? wait_status::NOTIFIED
                        : wait_status::IDLE;
  }

  std::unique_ptr<event_notifier> get_notifier() { return std::make_unique<event_notifier>(*this); }

  void shutdown()
  {
    std::lock_guard lock(mutex_);
    is_running_ = false;
    cv_.notify_one();
  }

 private:
  void notify()
  {
    std::lock_guard lock(mutex_);
    has_been_notified_ = true;
    cv_.notify_one();
  }

  void release_notifier()
  {
    std::lock_guard lock(mutex_);
    n_active_notifiers_--;
    if (n_active_notifiers_ == 0) { cv_.notify_all(); }
  }

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool has_been_notified_{false};
  std::size_t n_active_notifiers_{0};
  bool is_running_{true};
};

using event_notifier = notification_channel::event_notifier;

struct notify_on_exit {
  explicit notify_on_exit(std::unique_ptr<event_notifier> notifier) : notifer_(std::move(notifier))
  {
  }

  ~notify_on_exit() noexcept
  {
    try {
      if (notifer_) notifer_->post();
    } catch (...) {
    }
  }

 private:
  std::unique_ptr<event_notifier> notifer_;
};

}  // namespace memory
}  // namespace cucascade
