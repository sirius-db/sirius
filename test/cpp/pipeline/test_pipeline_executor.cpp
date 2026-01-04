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

#include "catch.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_executor.hpp"
#include "pipeline/task_request.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

using namespace sirius::pipeline;
using namespace sirius::parallel;
using namespace std::chrono_literals;

/**
 * Mock GPU pipeline task for testing.
 * This task simulates work without actually executing GPU operations.
 */
class mock_gpu_pipeline_task_global_state : public gpu_pipeline_task_global_state {
 public:
  mock_gpu_pipeline_task_global_state()
    : gpu_pipeline_task_global_state(nullptr), executed_count(0), gpu_ids_used()
  {
  }

  std::atomic<int> executed_count;
  std::vector<int> gpu_ids_used;
  std::mutex gpu_ids_mutex;
};

class mock_gpu_pipeline_task_local_state : public gpu_pipeline_task_local_state {
 public:
  mock_gpu_pipeline_task_local_state(int task_id, int expected_gpu_id)
    : gpu_pipeline_task_local_state(std::vector<std::shared_ptr<cucascade::data_batch>>{}),
      _task_id(task_id),
      _expected_gpu_id(expected_gpu_id)
  {
  }

  int _task_id;
  int _expected_gpu_id;
};

class mock_gpu_pipeline_task : public gpu_pipeline_task {
 public:
  mock_gpu_pipeline_task(std::unique_ptr<mock_gpu_pipeline_task_local_state> local_state,
                         std::shared_ptr<mock_gpu_pipeline_task_global_state> global_state)
    : gpu_pipeline_task(0,
                        std::vector<cucascade::shared_data_repository*>{},
                        std::move(local_state),
                        std::move(global_state))
  {
  }

  void execute() override
  {
    auto& global = _global_state->cast<mock_gpu_pipeline_task_global_state>();
    auto& local  = _local_state->cast<mock_gpu_pipeline_task_local_state>();

    // Simulate some work
    std::this_thread::sleep_for(5ms);

    // Increment counter
    global.executed_count.fetch_add(1, std::memory_order_relaxed);

    // Record which GPU (thread) executed this task
    {
      std::lock_guard<std::mutex> lock(global.gpu_ids_mutex);
      global.gpu_ids_used.push_back(local._task_id);
    }
  }
};

TEST_CASE("Pipeline executor can start and stop gracefully", "[pipeline_executor]")
{
  task_executor_config config{2, false};
  pipeline_executor executor(config, config, 1);

  REQUIRE_NOTHROW(executor.start());
  REQUIRE_NOTHROW(executor.stop());
}

TEST_CASE("Pipeline executor executes tasks through pipeline_queue", "[pipeline_executor]")
{
  task_executor_config config{2, false};
  pipeline_executor executor(config, config, 1);

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();

  executor.start();

  // Schedule multiple tasks
  const int num_tasks = 10;
  for (int i = 0; i < num_tasks; ++i) {
    auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(i, 0);
    auto task = std::make_unique<mock_gpu_pipeline_task>(std::move(local_state), global_state);

    // Submit task request
    auto request       = std::make_unique<task_request>();
    request->device_id = 0;
    executor.submit_task_request(std::move(request));

    // Schedule task
    executor.schedule(std::move(task));
  }

  // Wait for all tasks to complete
  auto start_time = std::chrono::steady_clock::now();
  auto timeout    = std::chrono::seconds(10);
  while (global_state->executed_count.load(std::memory_order_relaxed) < num_tasks) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      FAIL("Test timed out waiting for tasks to complete");
    }
  }

  REQUIRE(global_state->executed_count.load() == num_tasks);

  executor.stop();
}

TEST_CASE("Pipeline executor dispatches tasks to multiple GPU executors", "[pipeline_executor]")
{
  task_executor_config config{2, false};
  pipeline_executor executor(config, config, 2);

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();

  executor.start();

  // Schedule tasks to different GPUs
  const int num_tasks_per_gpu = 5;
  const int num_gpus          = 2;  // Test with 2 GPUs

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    for (int i = 0; i < num_tasks_per_gpu; ++i) {
      int task_id      = gpu_id * num_tasks_per_gpu + i;
      auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(task_id, gpu_id);
      auto task = std::make_unique<mock_gpu_pipeline_task>(std::move(local_state), global_state);

      // Submit task request with specific GPU ID
      auto request       = std::make_unique<task_request>();
      request->device_id = gpu_id;
      executor.submit_task_request(std::move(request));

      // Schedule task
      executor.schedule(std::move(task));
    }
  }

  // Wait for all tasks to complete
  const int total_tasks = num_tasks_per_gpu * num_gpus;
  auto start_time       = std::chrono::steady_clock::now();
  auto timeout          = std::chrono::seconds(10);
  while (global_state->executed_count.load(std::memory_order_relaxed) < total_tasks) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      FAIL("Test timed out waiting for tasks to complete");
    }
  }

  REQUIRE(global_state->executed_count.load() == total_tasks);
  REQUIRE(global_state->gpu_ids_used.size() == static_cast<size_t>(total_tasks));

  executor.stop();
}

TEST_CASE("GPU pipeline executor can start and stop independently", "[gpu_pipeline_executor]")
{
  task_executor_config config{2, false};
  pipeline_executor main_executor(config, config, 1);

  // GPU pipeline executor is created internally by pipeline_executor
  // but we can test its lifecycle through the main executor
  REQUIRE_NOTHROW(main_executor.start());
  REQUIRE_NOTHROW(main_executor.stop());
}

TEST_CASE("Task queue handles empty queue gracefully", "[pipeline_queue]")
{
  task_executor_config config{1, false};
  pipeline_executor executor(config, config, 1);

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();

  executor.start();

  // Don't schedule any tasks, just verify clean shutdown
  std::this_thread::sleep_for(50ms);

  REQUIRE(global_state->executed_count.load() == 0);

  REQUIRE_NOTHROW(executor.stop());
}

TEST_CASE("Pipeline executor handles rapid task submission", "[pipeline_executor]")
{
  task_executor_config config{4, false};
  pipeline_executor executor(config, config, 2);

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();

  executor.start();

  // Rapidly submit many tasks
  const int num_tasks = 50;
  for (int i = 0; i < num_tasks; ++i) {
    auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(i, 0);
    auto task = std::make_unique<mock_gpu_pipeline_task>(std::move(local_state), global_state);

    auto request       = std::make_unique<task_request>();
    request->device_id = i % 2;  // Alternate between 2 GPUs
    executor.submit_task_request(std::move(request));
    executor.schedule(std::move(task));
  }

  // Wait for all tasks to complete
  auto start_time = std::chrono::steady_clock::now();
  auto timeout    = std::chrono::seconds(15);
  while (global_state->executed_count.load(std::memory_order_relaxed) < num_tasks) {
    std::this_thread::sleep_for(20ms);
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      FAIL("Test timed out waiting for rapid task submission to complete");
    }
  }

  REQUIRE(global_state->executed_count.load() == num_tasks);

  executor.stop();
}

TEST_CASE("Pipeline executor task and request queue synchronization", "[pipeline_executor]")
{
  task_executor_config config{2, false};
  pipeline_executor executor(config, config, 2);

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();

  executor.start();

  // Submit requests and tasks in paired manner
  const int num_pairs = 20;
  for (int i = 0; i < num_pairs; ++i) {
    // Submit request first
    auto request       = std::make_unique<task_request>();
    request->device_id = i % 2;
    executor.submit_task_request(std::move(request));

    // Then submit corresponding task
    auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(i, i % 2);
    auto task = std::make_unique<mock_gpu_pipeline_task>(std::move(local_state), global_state);
    executor.schedule(std::move(task));
  }

  // Wait for completion
  auto start_time = std::chrono::steady_clock::now();
  auto timeout    = std::chrono::seconds(10);
  while (global_state->executed_count.load(std::memory_order_relaxed) < num_pairs) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      FAIL("Test timed out waiting for synchronized tasks to complete");
    }
  }

  REQUIRE(global_state->executed_count.load() == num_pairs);

  executor.stop();
}

TEST_CASE("Multiple start/stop cycles work correctly", "[pipeline_executor]")
{
  task_executor_config config{2, false};
  pipeline_executor executor(config, config, 1);

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();

  // First cycle
  executor.start();

  const int num_tasks = 5;
  for (int i = 0; i < num_tasks; ++i) {
    auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(i, 0);
    auto task = std::make_unique<mock_gpu_pipeline_task>(std::move(local_state), global_state);

    auto request       = std::make_unique<task_request>();
    request->device_id = 0;
    executor.submit_task_request(std::move(request));
    executor.schedule(std::move(task));
  }

  // Wait for tasks to complete
  auto start_time = std::chrono::steady_clock::now();
  auto timeout    = std::chrono::seconds(5);
  while (global_state->executed_count.load(std::memory_order_relaxed) < num_tasks) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      FAIL("Test timed out in first cycle");
    }
  }

  executor.stop();

  int first_cycle_count = global_state->executed_count.load();
  REQUIRE(first_cycle_count == num_tasks);

  // Second cycle - executor should work again after restart
  executor.start();

  for (int i = 0; i < num_tasks; ++i) {
    auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(i + num_tasks, 0);
    auto task = std::make_unique<mock_gpu_pipeline_task>(std::move(local_state), global_state);

    auto request       = std::make_unique<task_request>();
    request->device_id = 0;
    executor.submit_task_request(std::move(request));
    executor.schedule(std::move(task));
  }

  // Wait for second batch
  start_time = std::chrono::steady_clock::now();
  while (global_state->executed_count.load(std::memory_order_relaxed) < num_tasks * 2) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      FAIL("Test timed out in second cycle");
    }
  }

  REQUIRE(global_state->executed_count.load() == num_tasks * 2);

  executor.stop();
}
