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
#include "exec/channel.hpp"
#include "exec/config.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_request.hpp"
#include "scan/test_utils.hpp"
#include "utils/telemetry_utils.hpp"

#include <cuda_runtime_api.h>

#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kReservationBytes = 20 * 1024 * 1024;
constexpr std::size_t kAllocationBytes  = 10 * 1024 * 1024;

class test_gpu_pipeline_task_global_state
  : public sirius::pipeline::sirius_pipeline_task_global_state {
 public:
  test_gpu_pipeline_task_global_state()
    : sirius_pipeline_task_global_state(nullptr, sirius::test::make_test_telemetry_context())
  {
  }

  void add_error(std::string message)
  {
    std::cerr << message << std::endl;
    error_count.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(error_mutex);
    errors.push_back(std::move(message));
  }

  std::atomic<int> executed_count{0};
  std::atomic<int> error_count{0};
  std::mutex error_mutex;
  std::vector<std::string> errors;

  std::mutex memory_mutex;
  std::vector<std::size_t> memory_consumption;
};

class test_gpu_pipeline_task_local_state : public sirius::pipeline::gpu_pipeline_task_local_state {
 public:
  using sirius::pipeline::gpu_pipeline_task_local_state::gpu_pipeline_task_local_state;
};

class sirius_pipeline_task : public sirius::pipeline::gpu_pipeline_task {
 public:
  sirius_pipeline_task(uint64_t task_id,
                       std::unique_ptr<test_gpu_pipeline_task_local_state> local_state,
                       std::shared_ptr<test_gpu_pipeline_task_global_state> global_state)
    : gpu_pipeline_task(task_id,
                        std::vector<cucascade::shared_data_repository*>{},
                        std::move(local_state),
                        std::move(global_state))
  {
  }

  void execute(rmm::cuda_stream_view stream) override
  {
    auto& global = _global_state->cast<test_gpu_pipeline_task_global_state>();
    auto& local  = _local_state->cast<test_gpu_pipeline_task_local_state>();

    auto reservation = local.release_reservation();
    if (!reservation) {
      global.add_error("Missing GPU memory reservation for task.");
      global.executed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    auto& mem_space = reservation->get_memory_space();
    auto* allocator =
      reservation->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    if (!allocator) {
      global.add_error("Missing reservation-aware allocator for GPU memory space.");
      global.executed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    if (!allocator->attach_reservation_to_tracker(stream, std::move(reservation))) {
      global.add_error("Failed to attach reservation to stream tracker.");
      global.executed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    void* allocation = nullptr;
    try {
      allocation = allocator->allocate(stream, kAllocationBytes, alignof(std::max_align_t));
    } catch (const std::exception& e) {
      global.add_error(std::string("GPU allocation failed: ") + e.what());
      allocator->reset_stream_reservation(stream);
      global.executed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    allocator->deallocate(stream, allocation, kAllocationBytes, alignof(std::max_align_t));

    auto consumed_bytes = mem_space.get_total_reserved_memory();
    {
      std::lock_guard<std::mutex> lock(global.memory_mutex);
      global.memory_consumption.push_back(consumed_bytes);
    }

    allocator->reset_stream_reservation(stream);
    global.executed_count.fetch_add(1, std::memory_order_relaxed);
  }

  sirius::pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space* /*target_space*/) const override
  {
    sirius::pipeline::reservation_size_info info;
    info.reservation_size = kReservationBytes;
    return info;
  }

  std::vector<sirius::op::sirius_physical_operator*> get_output_consumers() override { return {}; }
};

}  // namespace

// Post-v1.0 push-model: tasks are pushed directly to the executor (see commit 90dc104 —
// management_eventloop now pops tasks from _task_queue and routes by preferred_device_id;
// gpu_pipeline_executor no longer publishes task_requests on a pull channel). The test
// keeps the request_channel wiring to validate executor construction, but schedules
// tasks directly instead of waiting on `request_channel.get()`.
TEST_CASE("GPU pipeline executor schedules GPU tasks directly (push-model)",
          "[gpu_pipeline_executor]")
{
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  try {
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(256 * 1024 * 1024)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(1 * 1024 * 1024 * 1024)
      .use_gpu_id_as_host_id()
      .track_reservation_per_stream(false)
      .set_reservation_fraction_per_numa_region(0.75);
    auto space_configs = builder.build();
    manager =
      std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
  } catch (const std::exception& e) {
    WARN("Skipping test due to insufficient GPUs: " << e.what());
    return;
  }

  auto* mem_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  if (!mem_space) {
    WARN("Skipping test because no GPU memory space is available.");
    return;
  }

  sirius::exec::channel<std::unique_ptr<sirius::pipeline::task_request>> request_channel;
  auto request_publisher = request_channel.make_publisher();

  sirius::exec::thread_pool_config config;
  config.num_threads        = 2;
  config.thread_name_prefix = "gpu-pipeline-test";

  sirius::pipeline::gpu_pipeline_executor executor(config,
                                                   mem_space,
                                                   std::move(request_publisher),
                                                   nullptr,
                                                   sirius::test::make_test_telemetry_context());
  auto global_state = std::make_shared<test_gpu_pipeline_task_global_state>();

  const int num_tasks = 10;
  std::atomic<int> dispatched{0};

  executor.start();

  std::thread request_handler([&]() {
    // Push-model: schedule tasks directly onto the executor. The executor's
    // manager_loop handles capacity/reservation internally (bounded_pool->reserve()).
    while (dispatched.load(std::memory_order_relaxed) < num_tasks) {
      auto local_state = std::make_unique<test_gpu_pipeline_task_local_state>(
        std::make_unique<sirius::op::pipelineable_operator_data>(
          std::vector<std::shared_ptr<cucascade::data_batch>>{}));
      auto task = std::make_unique<sirius_pipeline_task>(
        static_cast<uint64_t>(dispatched.load(std::memory_order_relaxed)),
        std::move(local_state),
        global_state);
      executor.schedule(std::move(task));
      dispatched.fetch_add(1, std::memory_order_relaxed);
    }
  });

  auto start_time = std::chrono::steady_clock::now();
  auto timeout    = std::chrono::seconds(20);
  while (global_state->executed_count.load(std::memory_order_relaxed) < num_tasks) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      executor.stop();
      request_channel.close();
      request_handler.join();
      FAIL("Timed out waiting for GPU pipeline tasks to complete.");
    }
  }

  executor.stop();
  request_channel.close();
  request_handler.join();

  if (global_state->error_count.load(std::memory_order_relaxed) > 0) {
    std::lock_guard<std::mutex> lock(global_state->error_mutex);
    for (const auto& error : global_state->errors) {
      INFO(error);
    }
  }

  REQUIRE(global_state->error_count.load(std::memory_order_relaxed) == 0);
  REQUIRE(global_state->executed_count.load(std::memory_order_relaxed) == num_tasks);

  {
    std::lock_guard<std::mutex> lock(global_state->memory_mutex);
    REQUIRE(global_state->memory_consumption.size() == static_cast<size_t>(num_tasks));
    for (auto consumed_bytes : global_state->memory_consumption) {
      REQUIRE(consumed_bytes >= kReservationBytes);
    }
  }
}

namespace {

// Shared observation state for the teardown-quiesce regression test below.
struct teardown_probe {
  std::atomic<bool> stream_work_done{false};         // set by the stream-ordered host func
  std::atomic<bool> destroyed_while_pending{false};  // task destroyed before stream quiesced
  std::atomic<bool> task_destroyed{false};
};

// Stream-ordered "pending work": blocks the stream for a while, then flips the
// flag. Anything that destroys task-owned device state before this has run is
// exactly the freed-while-pending teardown hazard.
void CUDART_CB slow_stream_work(void* userdata)
{
  auto* probe = static_cast<teardown_probe*>(userdata);
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  probe->stream_work_done.store(true, std::memory_order_release);
}

// A task that enqueues slow stream-ordered work and then throws, emulating an
// operator OOM that unwinds with kernels still in flight on exc_stream.
class throwing_pipeline_task : public sirius::pipeline::gpu_pipeline_task {
 public:
  enum class failure_mode { reschedule_exception, generic_exception };

  throwing_pipeline_task(uint64_t task_id,
                         std::unique_ptr<test_gpu_pipeline_task_local_state> local_state,
                         std::shared_ptr<test_gpu_pipeline_task_global_state> global_state,
                         std::shared_ptr<teardown_probe> probe,
                         failure_mode mode)
    : gpu_pipeline_task(task_id,
                        std::vector<cucascade::shared_data_repository*>{},
                        std::move(local_state),
                        std::move(global_state)),
      _probe(std::move(probe)),
      _mode(mode)
  {
  }

  ~throwing_pipeline_task() override
  {
    // REGRESSION GUARD for the staged-refresh corruption: the executor must
    // quiesce exc_stream on every abnormal exit BEFORE the task (and the
    // device memory it owns) is destroyed. If this destructor runs while the
    // stream still has our enqueued work pending, freed buffers could be
    // rebound by the async pool to concurrent queries mid-kernel.
    _probe->destroyed_while_pending.store(!_probe->stream_work_done.load(std::memory_order_acquire),
                                          std::memory_order_release);
    _probe->task_destroyed.store(true, std::memory_order_release);
  }

  void execute(rmm::cuda_stream_view stream) override
  {
    auto err = cudaLaunchHostFunc(stream.value(), slow_stream_work, _probe.get());
    if (err != cudaSuccess) {
      // Never leave the probe unresolvable: report and finish "quiesced".
      _probe->stream_work_done.store(true, std::memory_order_release);
      throw std::runtime_error("cudaLaunchHostFunc failed in throwing_pipeline_task");
    }
    if (_mode == failure_mode::reschedule_exception) {
      throw sirius::pipeline::oom_reschedule_exception(
        nullptr, 0, "test-injected OOM with stream work in flight");
    }
    throw std::runtime_error("test-injected failure with stream work in flight");
  }

  sirius::pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space* /*target_space*/) const override
  {
    sirius::pipeline::reservation_size_info info;
    info.reservation_size = kReservationBytes;
    return info;
  }

  std::vector<sirius::op::sirius_physical_operator*> get_output_consumers() override { return {}; }

 private:
  std::shared_ptr<teardown_probe> _probe;
  failure_mode _mode;
};

}  // namespace

// Regression test for the staged-refresh delta-serving corruption (concurrent
// throughput): a task that threw out of execute() was destroyed — and with it
// the exception's input batches — while its enqueued kernels were still
// running on exc_stream. In particular, the reschedule handler's
// has_error early return skipped the stream sync entirely. The async pool
// then rebound the freed device memory to concurrent queries while the
// orphaned kernels still read/wrote it, surfacing as illegal addresses,
// negative-size (2^64-N) allocations from scribbled string offsets, garbage
// VARCHAR bytes ("Invalid unicode" at the client), and never-terminating
// kernels. The executor must quiesce exc_stream before any teardown.
TEST_CASE("GPU pipeline executor quiesces the task stream before teardown on error paths",
          "[gpu_pipeline_executor]")
{
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  try {
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(256 * 1024 * 1024)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(1 * 1024 * 1024 * 1024)
      .use_gpu_id_as_host_id()
      .track_reservation_per_stream(false)
      .set_reservation_fraction_per_numa_region(0.75);
    auto space_configs = builder.build();
    manager =
      std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
  } catch (const std::exception& e) {
    WARN("Skipping test due to insufficient GPUs: " << e.what());
    return;
  }

  auto* mem_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  if (!mem_space) {
    WARN("Skipping test because no GPU memory space is available.");
    return;
  }

  sirius::exec::channel<std::unique_ptr<sirius::pipeline::task_request>> request_channel;
  auto request_publisher = request_channel.make_publisher();

  sirius::exec::thread_pool_config config;
  config.num_threads        = 2;
  config.thread_name_prefix = "gpu-pipeline-teardown-test";

  sirius::pipeline::gpu_pipeline_executor executor(config,
                                                   mem_space,
                                                   std::move(request_publisher),
                                                   nullptr,
                                                   sirius::test::make_test_telemetry_context());

  // Pre-error the completion handler: this drives the reschedule handler's
  // has_error early return — the exact path that used to skip the sync.
  sirius::pipeline::completion_handler handler;
  handler.report_error("pre-set error state (test)");
  executor.set_completion_handler(&handler);

  executor.start();

  auto const run_one = [&](throwing_pipeline_task::failure_mode mode, uint64_t task_id) {
    auto probe        = std::make_shared<teardown_probe>();
    auto global_state = std::make_shared<test_gpu_pipeline_task_global_state>();
    auto local_state  = std::make_unique<test_gpu_pipeline_task_local_state>(
      std::make_unique<sirius::op::pipelineable_operator_data>(
        std::vector<std::shared_ptr<cucascade::data_batch>>{}));
    executor.schedule(std::make_unique<throwing_pipeline_task>(
      task_id, std::move(local_state), global_state, probe, mode));

    auto const start_time = std::chrono::steady_clock::now();
    while (!probe->task_destroyed.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (std::chrono::steady_clock::now() - start_time > std::chrono::seconds(20)) {
        FAIL("Timed out waiting for the throwing task to be destroyed.");
      }
    }
    INFO("failure mode " << static_cast<int>(mode));
    REQUIRE_FALSE(probe->destroyed_while_pending.load(std::memory_order_acquire));
  };

  run_one(throwing_pipeline_task::failure_mode::reschedule_exception, 1);
  run_one(throwing_pipeline_task::failure_mode::generic_exception, 2);

  executor.stop();
  request_channel.close();
}
