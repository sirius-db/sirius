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
#include "helper/helper.hpp"

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
  void enqueue_message(sirius::unique_ptr<task_completion_message> message)
  {
    sirius::lock_guard<sirius::mutex> lock(_mutex);
    _message_queue.push(std::move(message));
    _sem.release();  // signal that one item is available
  }

  /**
   * @brief Removes and returns a completion message from the queue
   *
   * This method blocks until a message becomes available, then safely dequeues
   * and returns it to the caller.
   *
   * @return sirius::unique_ptr<task_completion_message> The dequeued message, or nullptr if queue
   * is empty
   */
  sirius::unique_ptr<task_completion_message> dequeue_message()
  {
    _sem.acquire();  // wait until there's something
    sirius::lock_guard<sirius::mutex> lock(_mutex);
    if (_message_queue.empty()) { return nullptr; }
    auto message = std::move(_message_queue.front());
    _message_queue.pop();
    return message;
  }

  /**
   * @brief Alias for dequeue_message for consistent interface
   *
   * @return sirius::unique_ptr<task_completion_message> The dequeued message
   */
  sirius::unique_ptr<task_completion_message> pull_message() { return dequeue_message(); }

 private:
  sirius::mutex _mutex;  ///< Mutex for thread-safe queue access
  sirius::queue<sirius::unique_ptr<task_completion_message>>
    _message_queue;                   ///< Underlying message queue
  std::counting_semaphore<> _sem{0};  ///< Semaphore for blocking/signaling (starts with 0 permits)
};

}  // namespace sirius
