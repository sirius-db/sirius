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
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_scheduler.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/split_connector.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

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

class zero_split_table_info : public sirius::op::scan::ingestible_table_info {
 public:
  [[nodiscard]] std::span<std::string const> column_names() const override { return _columns; }

  [[nodiscard]] std::span<std::string const> file_paths() const override { return _files; }

 private:
  std::vector<std::string> _columns{"only_col"};
  std::vector<std::string> _files{"zero-task://closed-before-dispatch"};
};

class empty_batch_coalescer : public sirius::op::scan::batch_coalescer {
 public:
  std::vector<std::unique_ptr<sirius::op::scan::scan_info>> push(
    std::unique_ptr<sirius::op::scan::scan_info>) override
  {
    return {};
  }

  std::vector<std::unique_ptr<sirius::op::scan::scan_info>> flush() override { return {}; }
};

class zero_split_ingestible : public sirius::op::scan::gpu_ingestible {
 public:
  std::unique_ptr<sirius::op::scan::batch_coalescer> create_batch_coalescer() const override
  {
    return std::make_unique<empty_batch_coalescer>();
  }

  [[nodiscard]] bool has_processed_all_metadata() const override { return true; }

  metadata_scan_task_t next_split_provider(sirius::io::ioctx_resolver) override { return {}; }

  sirius::op::scan::filtered_table materialize_metadata_to_table(
    const sirius::op::scan::scan_info&,
    const cucascade::memory::memory_space&,
    rmm::cuda_stream_view) override
  {
    throw std::logic_error("zero_split_ingestible should not materialize data");
  }

  std::unique_ptr<cudf::table> post_filter_and_project(sirius::op::scan::filtered_table&&,
                                                       const cucascade::memory::memory_space&,
                                                       rmm::cuda_stream_view) override
  {
    throw std::logic_error("zero_split_ingestible should not post-process data");
  }

  [[nodiscard]] const sirius::op::scan::ingestible_table_info& table_info() const noexcept override
  {
    return _table_info;
  }

  [[nodiscard]] std::vector<std::size_t> materialized_column_order() const override { return {}; }

 private:
  zero_split_table_info _table_info;
};

class test_result_collector_sink : public sirius_physical_operator {
 public:
  test_result_collector_sink()
    : sirius_physical_operator(SiriusPhysicalOperatorType::RESULT_COLLECTOR, {}, 0)
  {
  }

  bool is_sink() const override { return true; }

  std::unique_ptr<operator_data> execute(const operator_data&, rmm::cuda_stream_view) override
  {
    throw std::logic_error("zero-task result collector should not execute a task");
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

TEST_CASE("zero-task GPU scan source completes the result-collector query", "[zero-task-protocol]")
{
  if (sirius::test::g_shared_env && sirius::test::g_shared_env->is_active()) {
    sirius::test::g_shared_env->pause();
  }
  if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
    sirius::test::g_integration_env->pause();
  }
  if (sirius::test::g_integration_env_2gpu && sirius::test::g_integration_env_2gpu->is_active()) {
    sirius::test::g_integration_env_2gpu->pause();
  }

  pipeline_build_context build_ctx{};
  auto pipeline = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx);
  pipeline->set_pipeline_id(777);

  auto ingestible = std::make_shared<zero_split_ingestible>();
  sirius::op::scan::sirius_gpu_scan_operator scan(
    {sirius::logical_type::make(sirius::type_id::INTEGER)}, 0, ingestible);
  test_result_collector_sink result_collector;

  sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*pipeline, scan);
  build_state.set_pipeline_sink(*pipeline, &result_collector, 0);
  scan.set_pipeline(pipeline);
  result_collector.set_pipeline(pipeline);

  // This is the zero-task shape: the source has no splits and is already closed
  // before the task-creation request is processed.
  scan.get_split_connector().close();

  auto config_path =
    std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "scan" / "memory.yaml";
  auto db  = std::make_unique<duckdb::DuckDB>(nullptr);
  auto con = duckdb::Connection(*db);
  sirius::sirius_config config;
  config.load_from_file(config_path);
  auto sirius_ctx = duckdb::make_shared_ptr<duckdb::SiriusContext>();
  sirius_ctx->initialize(config);
  REQUIRE(sirius_ctx != nullptr);
  REQUIRE(sirius_ctx->is_initialized());
  auto& sirius_ctx_ref = *sirius_ctx;
  con.context->registered_state->Remove("sirius_state");
  con.context->registered_state->Insert("sirius_state", sirius_ctx);
  sirius_ctx_ref.get_task_creator().set_client_context(*con.context);

  auto telemetry_context = sirius_ctx_ref.get_telemetry_context();
  REQUIRE(telemetry_context != nullptr);
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> pipelines;
  pipelines.push_back(pipeline);
  sirius_ctx_ref.create_query(std::move(pipelines),
                              sirius::telemetry::query_telemetry_info{
                                telemetry_context->engine_id(), telemetry_context->worker_id()});

  auto future = sirius_ctx_ref.get_task_scheduler().start_query();
  auto status = future.wait_for(30s);
  REQUIRE(status == std::future_status::ready);
  REQUIRE_NOTHROW(future.get());
  CHECK(pipeline->get_tasks_created() == 0);
  CHECK(scan.get_split_connector().is_closed());
  REQUIRE_NOTHROW(sirius_ctx_ref.get_task_scheduler().wait_for_completion());
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
