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

#include "gpu_pipeline_hashmap.hpp"
#include "gpu_physical_operator.hpp"
#include "gpu_pipeline.hpp"
#include "data/data_batch_view.hpp"
#include "parallel/task_executor.hpp"
#include "helper/helper.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <variant>

namespace sirius {

using task_creation_hint = std::variant<std::monostate, ::duckdb::GPUPhysicalOperator*, ::duckdb::GPUPipeline*>;

struct task_creation_info {
  std::shared_ptr<::duckdb::GPUPipeline> pipeline;
  std::vector<sirius::data_batch_view> input_batches;
};

class task_creation_queue {
 public:
  virtual ~task_creation_queue() = default;

  // Open the queue and start accepting new tasks.
  virtual void open() = 0;

  // Close the queue and stop processing new tasks.
  virtual void close() = 0;

  // Add a task to the queue.
  virtual void push(std::unique_ptr<task_creation_hint> task) = 0;

  // Pull a task from the queue. Wait until a task available or the queue is closed.
  virtual std::unique_ptr<task_creation_hint> pull() = 0;
};

class task_creator {
    task_creator(std::unique_ptr<task_creation_queue> task_creation_queue, size_t num_threads, gpu_pipeline_hashmap& gpu_pipeline_map)
      : _task_creation_queue(std::move(task_creation_queue)), _num_threads(num_threads), _running(false)
    {
      for (int i = 0; i < gpu_pipeline_map._vec.size(); ++i) {
        if (gpu_pipeline_map._vec[i]->GetSource() == PhysicalOperatorType::TABLE_SCAN) {
          priority_scans.push(gpu_pipeline_map._vec[i]);
        }
      }
    }

    virtual ~task_creator() { stop_thread_pool(); }

    // Non-copyable and movable
    task_creator(const task_creator&)            = delete;
    task_creator& operator=(const task_creator&) = delete;
    task_creator(task_creator&&)                 = default;
    task_creator& operator=(task_creator&&)      = default;

    void process_next_task(const duckdb::GPUPhysicalOperator* node) {
        auto hint = node->get_next_task_hint();
        if (std::holds_alternative<duckdb::GPUPhysicalOperator*>(hint)) {
            schedule(hint);
        } else if (std::holds_alternative<duckdb::GPUPipeline*>(hint)) {
            auto* pipeline = std::get<duckdb::GPUPipeline*>(hint);
            auto* node = pipeline->GetOperator()[0];
            process_next_task(node);
        } else {
            duckdb::GPUPhysicalOperator* node = priority_scans.front();
            schedule(std::make_unique<task_creation_hint>(node, nullptr));
        }
    }

    void start() {
        while (!priority_scans.empty()) {
          auto* node = priority_scans.front();
          schedule(std::make_unique<task_creation_hint>(node, nullptr));
          priority_scans.pop();
        }
    }

    void start_thread_pool() {
      bool expected = false;
      if (!_running.compare_exchange_strong(expected, true)) { return; }
      on_start();
      _threads.reserve(_num_threads);
      for (int i = 0; i < _num_threads; ++i) {
          _threads.push_back(std::make_unique<std::thread>(
          &task_creator::worker_function, this, i));
      }
    }

    void stop_thread_pool() {
      bool expected = true;
      if (!_running.compare_exchange_strong(expected, false)) { return; }
      on_stop();
      for (auto& thread : _threads) {
        if (thread->joinable()) { thread->join(); }
      }
      _threads.clear();
    }

    void schedule(std::unique_ptr<task_creation_hint> hint) {
        _task_creation_queue->push(std::move(hint));
    }

    void worker_function(int worker_id) {
      while (true) {
        if (!_running.load()) {
          // Executor is stopped.
          break;
        }
        auto hint = _task_creation_queue->pull();
        if (hint == nullptr) {
          // Task queue is closed.
          break;
        }
        try {
          auto node = std::get<duckdb::GPUPhysicalOperator*>(hint);
          node->create_tasks();
        } catch (const std::exception& e) {
          schedule(std::move(hint));
        }
      }
    }

    void on_start() { _task_creation_queue->open(); }

    void on_stop() { _task_creation_queue->close(); }

    private:
      size_t _num_threads;
      std::atomic<bool> _running;
      std::vector<std::unique_ptr<std::thread>> _threads;
      std::queue<duckdb::GPUPhysicalOperator*> priority_scans;
      std::unique_ptr<task_creation_queue> _task_creation_queue;
};

}  // namespace sirius
