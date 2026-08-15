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
#include "exec/query_lifecycle_registry.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_scheduler.hpp"
#include "query_id.hpp"
#include "scan/test_utils.hpp"
#include "utils/telemetry_utils.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
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
                        std::vector<std::shared_ptr<cucascade::shared_data_repository>>{},
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

TEST_CASE("terminate_query fails only its own query and leaves the scheduler running",
          "[task_scheduler][concurrency]")
{
  // Regression for the "one query's failure hangs the whole engine" class of bug:
  // terminate_query used to call stop(), which closed the request channel, joined the management
  // thread and stopped every GPU executor -- for ALL queries -- with no path that ever calls
  // start() again. Any other in-flight query was left waiting on a completion that could never
  // arrive, as was every subsequent query in the process.
  auto manager = initialize_memory_manager(1);
  sirius::exec::thread_pool_config gpu_config{2};
  task_scheduler executor(gpu_config, *manager, sirius::test::make_test_telemetry_context());

  executor.start();

  // Query A fails.
  auto handler_a = std::make_shared<completion_handler>();
  auto future_a  = handler_a->get_awaitable();
  executor.terminate_query(handler_a,
                           std::make_exception_ptr(std::runtime_error("query A creation failed")));

  // A -- and only A -- is failed.
  REQUIRE_THROWS_AS(future_a.get(), std::runtime_error);

  // The scheduler must still dispatch work. Before the fix this hung until the 10s timeout
  // because the management thread and every GPU executor were already stopped.
  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();

  const int num_tasks = 5;
  for (int i = 0; i < num_tasks; ++i) {
    auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(i, 0);
    auto task = std::make_unique<mock_gpu_pipeline_task>(i, std::move(local_state), global_state);
    executor.schedule(std::move(task));
  }

  auto start_time = std::chrono::steady_clock::now();
  auto timeout    = std::chrono::seconds(10);
  while (global_state->executed_count.load(std::memory_order_relaxed) < num_tasks) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      FAIL("scheduler stopped dispatching after terminate_query on an unrelated query");
    }
  }
  REQUIRE(global_state->executed_count.load() == num_tasks);

  executor.stop();
}

TEST_CASE("the lifecycle gate refuses scheduling for a quiescing query",
          "[task_scheduler][query_lifecycle_gate][concurrency]")
{
  // Wiring check for the gate: task_scheduler::schedule must consult it. A task creation worker
  // can land in schedule() after this query's queue drain already ran, and the task would then
  // sit in the shared queue holding raw repository pointers into a manager about to be erased.
  auto manager = initialize_memory_manager(1);
  sirius::exec::thread_pool_config gpu_config{2};
  // Declared BEFORE the scheduler so it is destroyed AFTER it: the scheduler and every GPU
  // executor hold a raw pointer to the gate, and ~task_scheduler still runs stop().
  sirius::exec::query_lifecycle_registry lifecycle;
  task_scheduler executor(gpu_config, *manager, sirius::test::make_test_telemetry_context());
  executor.set_query_lifecycle_registry(&lifecycle);

  // Mock tasks carry no pipeline, so index_keys_for() reports them as query 0.
  const auto mock_query = sirius::make_query_id(0);
  lifecycle.open_query(mock_query);

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();
  executor.start();

  auto schedule_batch = [&](int count, int id_base) {
    for (int i = 0; i < count; ++i) {
      auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(id_base + i, 0);
      auto task =
        std::make_unique<mock_gpu_pipeline_task>(id_base + i, std::move(local_state), global_state);
      executor.schedule(std::move(task));
    }
  };

  // Open: work flows.
  const int num_tasks = 5;
  schedule_batch(num_tasks, 0);

  auto start_time = std::chrono::steady_clock::now();
  while (global_state->executed_count.load(std::memory_order_relaxed) < num_tasks) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start_time > std::chrono::seconds(10)) {
      FAIL("tasks did not execute while the query was open");
    }
  }
  REQUIRE(global_state->executed_count.load() == num_tasks);

  // Quiescing: further work is dropped rather than enqueued.
  lifecycle.quiesce(mock_query);
  schedule_batch(num_tasks, 100);

  std::this_thread::sleep_for(200ms);
  REQUIRE(global_state->executed_count.load() == num_tasks);

  executor.stop();
}

TEST_CASE("wait_for_completion validates only its own query's queue",
          "[task_scheduler][query_lifecycle_gate][concurrency]")
{
  // Regression for A5: wait_for_completion used the whole-queue size(), so query A completing
  // normally threw "pipeline task queue not empty at query completion" purely because query B had
  // work legitimately queued. It also called _task_creator->stop_thread_pool(), tearing down the
  // SHARED creation pool on every successful completion.
  auto manager = initialize_memory_manager(1);
  sirius::exec::thread_pool_config gpu_config{2};
  sirius::exec::query_lifecycle_registry lifecycle;
  task_scheduler executor(gpu_config, *manager, sirius::test::make_test_telemetry_context());
  executor.set_query_lifecycle_registry(&lifecycle);

  // Mock tasks carry no pipeline, so index_keys_for() reports them as query 0. Completing a
  // DIFFERENT query id must ignore them entirely.
  const auto other_query      = sirius::make_query_id(0);
  const auto completing_query = sirius::make_query_id(42);
  lifecycle.open_query(other_query);
  lifecycle.open_query(completing_query);

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();
  executor.start();

  // Park work belonging to `other_query` in the scheduler queue by keeping every device busy.
  for (int i = 0; i < 8; ++i) {
    auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(i, 0);
    executor.schedule(
      std::make_unique<mock_gpu_pipeline_task>(i, std::move(local_state), global_state));
  }

  // The assertion this test exists for: completing an unrelated query must not throw because a
  // co-tenant has work queued. Before the per-query size() check, this threw
  // "pipeline task queue not empty at query completion" every time.
  REQUIRE_NOTHROW(executor.wait_for_completion(completing_query));

  // NOT asserted here: that every co-tenant task still runs. wait_and_validate_empty() must
  // release the manager thread's pool slot before wait_all() can return (see
  // itask_executor::quiesce_manager), and that interrupt makes push() return false for the
  // duration — so a co-tenant task in transit from the scheduler to a device queue can still be
  // dropped. Step 3 removes the *whole-queue* drain that used to destroy co-tenants' queued work
  // outright; eliminating the in-transit drop needs per-query in-flight accounting from the
  // query-aware bounded_thread_pool, at which point this test should gain a liveness assertion.
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

// Regression test for the #1467 deadlock: the downgrade executor extracts
// queued tasks and returns them via convertible_gpu_pipeline_task's RAII
// destructor — a direct queue push with no task_available event. With every
// executor already parked, the pre-fix loop (blocked on the channel) never
// dispatched them; this stages that interleaving and times out on it.
TEST_CASE("Tasks extracted and RAII-returned while executors are parked still run",
          "[task_scheduler][deadlock-1467]")
{
  auto manager = initialize_memory_manager(1);
  sirius::exec::thread_pool_config gpu_config{2};
  task_scheduler sched(gpu_config, *manager, sirius::test::make_test_telemetry_context());

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();
  auto* queue       = sched.get_pipeline_task_queue();

  // Schedule before start() so the extraction below cannot race the matcher.
  const int num_tasks = 2;
  for (int i = 0; i < num_tasks; ++i) {
    auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(i, 0);
    auto task = std::make_unique<mock_gpu_pipeline_task>(i, std::move(local_state), global_state);
    sched.schedule(std::move(task));
  }

  // Extract every task, wrapped so the RAII destructor returns it (as the
  // downgrade's TIER-2 pass does).
  std::vector<std::unique_ptr<sirius::convertible_gpu_pipeline_task>> extracted;
  while (auto t = queue->mutable_pop_if([](sirius::parallel::itask&) { return true; },
                                        /*front_to_back=*/false)) {
    // Keys are resolved at extraction time, exactly as the TIER-2 provider does.
    const auto keys = sirius::pipeline::index_keys_for(**t);
    extracted.push_back(
      std::make_unique<sirius::convertible_gpu_pipeline_task>(std::move(*t), *queue, keys));
  }
  REQUIRE(extracted.size() == num_tasks);

  // Start with an empty queue; every executor posts device_ready and parks.
  sched.start();
  std::this_thread::sleep_for(500ms);
  REQUIRE(global_state->executed_count.load() == 0);

  // Destroying the wrappers pushes the tasks back — the silent return.
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
