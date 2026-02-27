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
#include "exec/config.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_executor.hpp"
#include "scan/test_utils.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

using namespace sirius::pipeline;
using namespace sirius::parallel;
using namespace std::chrono_literals;
using namespace sirius::op;

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
    : gpu_pipeline_task_local_state(
        std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{})),
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

  void execute(rmm::cuda_stream_view stream) override
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
  auto manager = initialize_memory_manager(1);
  sirius::exec::thread_pool_config gpu_config{2};
  sirius::exec::thread_pool_config scan_config{2};
  pipeline_executor executor(gpu_config, scan_config, *manager);

  REQUIRE_NOTHROW(executor.start());
  REQUIRE_NOTHROW(executor.stop());
}

TEST_CASE("Pipeline executor executes tasks through pipeline_queue", "[pipeline_executor]")
{
  auto manager = initialize_memory_manager(1);
  sirius::exec::thread_pool_config gpu_config{2};
  sirius::exec::thread_pool_config scan_config{2};
  pipeline_executor executor(gpu_config, scan_config, *manager);

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();

  executor.start();

  // Schedule multiple tasks
  const int num_tasks = 10;
  for (int i = 0; i < num_tasks; ++i) {
    auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(i, 0);
    auto task = std::make_unique<mock_gpu_pipeline_task>(std::move(local_state), global_state);
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

TEST_CASE("Task queue handles empty queue gracefully", "[pipeline_queue]")
{
  auto manager = initialize_memory_manager(1);
  sirius::exec::thread_pool_config gpu_config{2};
  sirius::exec::thread_pool_config scan_config{2};
  pipeline_executor executor(gpu_config, scan_config, *manager);

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();

  executor.start();

  // Don't schedule any tasks, just verify clean shutdown
  std::this_thread::sleep_for(50ms);

  REQUIRE(global_state->executed_count.load() == 0);

  REQUIRE_NOTHROW(executor.stop());
}
