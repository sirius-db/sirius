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
#include "data/convertible_gpu_pipeline_task.hpp"
#include "exec/config.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_scheduler.hpp"
#include "scan/test_utils.hpp"
#include "utils/telemetry_utils.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

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
    : gpu_pipeline_task_global_state(nullptr, sirius::test::make_test_telemetry_context()),
      executed_count(0),
      gpu_ids_used()
  {
  }

  std::atomic<int> executed_count;
  std::vector<int> gpu_ids_used;
  std::mutex gpu_ids_mutex;
};

class mock_gpu_pipeline_task_local_state : public gpu_pipeline_task_local_state {
 public:
  mock_gpu_pipeline_task_local_state(int task_id, int expected_gpu_id)
    : gpu_pipeline_task_local_state(std::make_unique<pipelineable_operator_data>(
        std::vector<std::shared_ptr<cucascade::data_batch>>{})),
      _task_id(task_id),
      _expected_gpu_id(expected_gpu_id)
  {
  }

  int _task_id;
  int _expected_gpu_id;
};

class mock_gpu_pipeline_task : public gpu_pipeline_task {
 public:
  mock_gpu_pipeline_task(uint64_t task_id,
                         std::unique_ptr<mock_gpu_pipeline_task_local_state> local_state,
                         std::shared_ptr<mock_gpu_pipeline_task_global_state> global_state)
    : gpu_pipeline_task(task_id,
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

TEST_CASE("Task scheduler can start and stop gracefully", "[task_scheduler]")
{
  auto manager = initialize_memory_manager(1);
  sirius::exec::thread_pool_config gpu_config{2};
  task_scheduler executor(gpu_config, *manager, sirius::test::make_test_telemetry_context());

  REQUIRE_NOTHROW(executor.start());
  REQUIRE_NOTHROW(executor.stop());
}

TEST_CASE("Task scheduler executes tasks through pipeline_queue", "[task_scheduler]")
{
  auto manager = initialize_memory_manager(1);
  sirius::exec::thread_pool_config gpu_config{2};
  task_scheduler executor(gpu_config, *manager, sirius::test::make_test_telemetry_context());

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();

  executor.start();

  // Schedule multiple tasks
  const int num_tasks = 10;
  for (int i = 0; i < num_tasks; ++i) {
    auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(i, 0);
    auto task = std::make_unique<mock_gpu_pipeline_task>(i, std::move(local_state), global_state);
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
  task_scheduler executor(gpu_config, *manager, sirius::test::make_test_telemetry_context());

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();

  executor.start();

  // Don't schedule any tasks, just verify clean shutdown
  std::this_thread::sleep_for(50ms);

  REQUIRE(global_state->executed_count.load() == 0);

  REQUIRE_NOTHROW(executor.stop());
}

namespace {

/// Poll until @p done() or @p timeout elapses; FAIL the test on timeout.
template <typename Pred>
void wait_or_fail(Pred done, std::chrono::seconds timeout, const char* what)
{
  const auto start = std::chrono::steady_clock::now();
  while (!done()) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) { FAIL(what); }
  }
}

}  // namespace

// Regression test for the PR #1467 deadlock.
//
// The downgrade executor's TIER-2 pass extracts queued pipeline tasks with
// mutable_pop_if and returns them via convertible_gpu_pipeline_task's RAII
// destructor — a direct _task_queue.push() that emits no task_available event.
// If every executor had already posted device_ready and parked (empty queue)
// by then, no channel event would ever arrive again and the returned tasks
// were never dispatched: a permanent all-workers-idle deadlock (SF1000 TPC-H
// q18, hung 7/7). This stages that interleaving deterministically; on the
// pre-fix event loop it times out.
TEST_CASE("Tasks extracted and RAII-returned while executors are parked still run",
          "[task_scheduler][deadlock-1467]")
{
  auto manager = initialize_memory_manager(1);
  sirius::exec::thread_pool_config gpu_config{2};
  task_scheduler sched(gpu_config, *manager, sirius::test::make_test_telemetry_context());

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();
  auto* queue       = sched.get_pipeline_task_queue();

  // Schedule before start(): the tasks sit in the queue and the
  // task_available events buffer in the channel, so the extraction below
  // cannot race the matcher.
  const int num_tasks = 2;  // the captured failure extracted exactly 2 tasks
  for (int i = 0; i < num_tasks; ++i) {
    auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(i, 0);
    auto task = std::make_unique<mock_gpu_pipeline_task>(i, std::move(local_state), global_state);
    sched.schedule(std::move(task));
  }

  // TIER-2 extraction: pull every task out of the queue, wrapping each so the
  // RAII destructor returns it (convertible_gpu_pipeline_task_provider does the
  // same, with a memory-space predicate).
  std::vector<std::unique_ptr<sirius::convertible_gpu_pipeline_task>> extracted;
  while (auto t = queue->mutable_pop_if([](sirius::parallel::itask&) { return true; },
                                        /*front_to_back=*/false)) {
    extracted.push_back(
      std::make_unique<sirius::convertible_gpu_pipeline_task>(std::move(*t), *queue));
  }
  REQUIRE(extracted.size() == num_tasks);

  // Start the scheduler: the buffered task_available events are consumed
  // against a now-empty queue, every executor posts device_ready and parks.
  sched.start();
  std::this_thread::sleep_for(500ms);
  REQUIRE(global_state->executed_count.load() == 0);  // nothing to run yet

  // "Conversion finishes": destroying the wrappers pushes the tasks back via
  // the RAII destructor — the silent return that deadlocked the pre-fix loop.
  extracted.clear();

  wait_or_fail([&] { return global_state->executed_count.load() == num_tasks; },
               10s,
               "DEADLOCK REGRESSION: RAII-returned tasks were never dispatched "
               "(downgrade return emitted no event and the matcher never re-ran)");

  sched.stop();
}

TEST_CASE("Task scheduler dispatches tasks with device preference", "[task_scheduler]")
{
  // Multi-GPU device-preference dispatch needs a real 2-GPU host; skip on
  // single-GPU machines (mirrors the require_two_gpus() convention used by the
  // MGPU operator tests in mgpu_test_utils.hpp).
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("Task scheduler device-preference test requires >=2 GPUs; single-GPU host — skipping");
    return;
  }

  auto manager = initialize_memory_manager(2);
  sirius::exec::thread_pool_config gpu_config{2};
  task_scheduler sched(gpu_config, *manager, sirius::test::make_test_telemetry_context());

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();
  sched.start();

  // Schedule tasks — pull-signal model ensures tasks stay in the scheduler's
  // queue (downgrade-visible) until a GPU executor is ready.
  const int num_tasks = 10;
  for (int i = 0; i < num_tasks; ++i) {
    auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(i, 0);
    auto task = std::make_unique<mock_gpu_pipeline_task>(i, std::move(local_state), global_state);
    sched.schedule(std::move(task));
  }

  auto start_time = std::chrono::steady_clock::now();
  while (global_state->executed_count.load(std::memory_order_relaxed) < num_tasks) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start_time > 10s) {
      FAIL("Tasks not completed with 2-GPU scheduler");
    }
  }
  REQUIRE(global_state->executed_count.load() == num_tasks);
  sched.stop();
}
