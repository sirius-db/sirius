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

#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <semaphore>

namespace sirius {

/**
 * @brief Enumeration of task sources for completion tracking
 */
enum class Source {
  SCAN,     ///< Task originated from a scan operation
  PIPELINE  ///< Task originated from a pipeline operation
};

/**
 * @brief Message structure for notifying task completion events
 *
 * This structure carries information about completed tasks to notify the task creator
 * that it should check if new dependent tasks can be scheduled for execution.
 */
struct task_completion_message {
  uint64_t task_id;      ///< Unique identifier of the completed task
  uint64_t pipeline_id;  ///< Identifier of the pipeline associated with the completed task
  Source source;         ///< Source type of the completed task (scan or pipeline)
};

/**
 * @brief Thread-safe queue for task completion message passing
 *
 * This class provides a thread-safe mechanism for tasks to notify the task creator
 * about their completion. It uses a semaphore-based blocking queue to ensure
 * efficient communication between producer (tasks) and consumer (task creator) threads.
 */
class task_completion_message_queue {
 public:
  /**
   * @brief Constructs a new task_completion_message_queue
   */
  task_completion_message_queue() = default;

  /**
   * @brief Destructor for task_completion_message_queue
   */
  ~task_completion_message_queue() = default;

  /**
   * @brief Adds a completion message to the queue
   *
   * This method safely enqueues a task completion message and signals waiting
   * consumers that a new message is available.
   *
   * @param message The completion message to enqueue (ownership is transferred)
   */
  void EnqueueMessage(std::unique_ptr<task_completion_message> message)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    message_queue_.push(std::move(message));
    sem_.release();  // signal that one item is available
  }

  /**
   * @brief Removes and returns a completion message from the queue
   *
   * This method blocks until a message becomes available, then safely dequeues
   * and returns it to the caller.
   *
   * @return std::unique_ptr<task_completion_message> The dequeued message, or nullptr if queue
   * is empty
   */
  std::unique_ptr<task_completion_message> DequeueMessage()
  {
    sem_.acquire();  // wait until there's something
    std::lock_guard<std::mutex> lock(mutex_);
    if (message_queue_.empty()) { return nullptr; }
    auto message = std::move(message_queue_.front());
    message_queue_.pop();
    return message;
  }

  /**
   * @brief Alias for DequeueMessage for consistent interface
   *
   * @return std::unique_ptr<task_completion_message> The dequeued message
   */
  std::unique_ptr<task_completion_message> PullMessage() { return DequeueMessage(); }

 private:
  std::mutex mutex_;  ///< Mutex for thread-safe queue access
  std::queue<std::unique_ptr<task_completion_message>>
    message_queue_;                   ///< Underlying message queue
  std::counting_semaphore<> sem_{0};  ///< Semaphore for blocking/signaling (starts with 0 permits)
};

}  // namespace sirius
