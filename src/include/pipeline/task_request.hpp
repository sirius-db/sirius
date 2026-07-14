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
#include "parallel/task_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"

#include <blockingconcurrentqueue.h>
#include <chrono>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

namespace sirius {
namespace pipeline {

enum class task_request_kind {
  // A GPU pipeline executor has reserved a worker thread and is ready to accept a task.
  // device_id is the device that is ready.
  device_ready,
  // A new task was pushed into task_scheduler::_task_queue. Wakes the management
  // event loop so it re-runs the matcher against currently-ready devices.
  // device_id is unused.
  task_available,
};

struct task_request {
  int device_id;
  bool is_scan{false};
  task_request_kind kind{task_request_kind::device_ready};
  std::chrono::high_resolution_clock::time_point created_at{std::chrono::high_resolution_clock::now()};
};

class task_request_queue {
 public:
  task_request_queue(size_t num_threads) : _num_threads(num_threads) {};
  void open();
  void close();
  void push(std::unique_ptr<task_request> request);
  std::unique_ptr<task_request> pull();

 private:
  size_t _num_threads;
  duckdb_moodycamel::BlockingConcurrentQueue<std::unique_ptr<task_request>> _request_queue;
  std::atomic<bool> _is_open{false};  ///< Whether the queue is open for pushing/pulling tasks
};

}  // namespace pipeline
}  // namespace sirius
