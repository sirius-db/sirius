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

// sirius
#include <parallel/task.hpp>
#include <parallel/task_queue.hpp>

// duckdb
#include <blockingconcurrentqueue.h>

// standard library
#include <atomic>

namespace sirius::parallel {

//===----------------------------------------------------------------------===//
// DuckDB Scan Task Queue
//===----------------------------------------------------------------------===//

/**
 * @brief A concurrent scan task queue for duckdb scan tasks.
 *
 * This class implements a concurrent task queue simply by wrapping moodycamel's
 * BlockingConcurrentQueue, provided as an external dependency by DuckDB. The constructor uses the
 * default configuration for the underlying BlockingConcurrentQueue.
 *
 */
class duckdb_scan_task_queue : public itask_queue {
 public:
  //===----------Constructor & Destructor----------===//
  explicit duckdb_scan_task_queue(size_t num_threads) : _num_threads(num_threads), _queue() {}
  ~duckdb_scan_task_queue() override = default;

  //===----------Methods----------===//
  void open() override { _is_open.store(true, std::memory_order_release); }

  void close() override
  {
    _is_open.store(false, std::memory_order_release);
    // Wake up all threads blocked in wait_dequeue by pushing nullptr sentinels
    for (size_t i = 0; i < _num_threads; ++i) {
      _queue.enqueue(nullptr);
    }
  }

  void push(std::unique_ptr<itask> task) override { _queue.enqueue(std::move(task)); }

  std::unique_ptr<itask> pull() override
  {
    std::unique_ptr<itask> task;
    while (true) {
      if (_queue.try_dequeue(task)) { return task; }

      // If the queue is closed and empty, return nullptr to indicate no more tasks.
      if (!_is_open.load(std::memory_order_acquire)) { return nullptr; }

      // Otherwise, wait for a task to become available.
      _queue.wait_dequeue(task);
      if (task) { return task; }
    }
  }

 private:
  //===----------Fields----------===//
  size_t _num_threads;                ///< Number of worker threads (for proper cleanup on close)
  std::atomic<bool> _is_open{false};  ///< Whether the queue is open for pushing/pulling tasks
  duckdb_moodycamel::BlockingConcurrentQueue<std::unique_ptr<itask>>
    _queue;  ///< The underlying concurrent queue
};

}  // namespace sirius::parallel
