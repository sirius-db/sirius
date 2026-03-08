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
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_request.hpp"
#include "scan/test_utils.hpp"

#include <rmm/mr/device_memory_resource.hpp>

#include <absl/cleanup/cleanup.h>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

// Memory layout:
//   GPU capacity  = 1100 MB (software limit)
//   Reservation   =  50 MB per task (intentionally small)
//   Allocation    = 400 MB per task (much larger than reservation)
//
// With 3 threads, all 3 tasks start concurrently.
// Accounting after 3 reservations: 150 MB
// First  allocation (+350 MB overflow): total =  500 MB ≤ 1100 → OK
// Second allocation (+350 MB overflow): total =  850 MB ≤ 1100 → OK
// Third  allocation (+350 MB overflow): total = 1200 MB > 1100 → OOM!
//
// The OOM'd task gets rescheduled. After the first two complete, enough
// accounting headroom is freed for the rescheduled task to succeed.
// (Note: cucascade's cross-boundary deallocation accounting leaves some
//  residual in _total_allocated_bytes, hence the extra capacity margin.)
constexpr std::size_t kGpuCapacity     = 1100ULL * 1024 * 1024;  // 1100 MB
constexpr std::size_t kReservationSize = 50ULL * 1024 * 1024;    // 50 MB
constexpr std::size_t kAllocationBytes = 400ULL * 1024 * 1024;   // 400 MB
constexpr auto kHoldDuration           = std::chrono::milliseconds(800);

//------------------------------------------------------------------------------
// Test global state — shared across all tasks
//------------------------------------------------------------------------------
class oom_test_global_state : public sirius::pipeline::sirius_pipeline_task_global_state {
 public:
  oom_test_global_state() : sirius_pipeline_task_global_state(nullptr) {}

  void add_error(std::string message)
  {
    error_count.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(error_mutex);
    errors.push_back(std::move(message));
  }

  std::atomic<int> completed_count{0};
  std::atomic<int> oom_count{0};
  std::atomic<int> error_count{0};
  std::mutex error_mutex;
  std::vector<std::string> errors;
};

//------------------------------------------------------------------------------
// Test task — allocates kAllocationBytes of GPU memory, holds it, then frees.
// If OOM occurs, throws oom_reschedule_exception for the executor to handle.
//------------------------------------------------------------------------------
class oom_test_task : public sirius::pipeline::gpu_pipeline_task {
 public:
  oom_test_task(uint64_t task_id,
                std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state> local_state,
                std::shared_ptr<oom_test_global_state> global_state)
    : gpu_pipeline_task(task_id,
                        std::vector<cucascade::shared_data_repository*>{},
                        std::move(local_state),
                        std::move(global_state))
  {
  }

  void execute(rmm::cuda_stream_view stream) override
  {
    auto& global = _global_state->cast<oom_test_global_state>();
    auto& local  = _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>();

    auto reservation = local.release_reservation();
    if (!reservation) {
      global.add_error("Missing GPU memory reservation for OOM test task.");
      global.completed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    auto* allocator =
      reservation->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    if (!allocator) {
      global.add_error("Missing reservation-aware allocator for OOM test task.");
      global.completed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    // Attach with ignore policy — reservations are intentionally small so the
    // actual allocation (kAllocationBytes >> kReservationSize) exceeds the
    // reservation. The ignore policy lets the allocation proceed to the
    // upstream pool, where it will OOM if the software capacity is exceeded.
    if (!allocator->attach_reservation_to_tracker(
          stream,
          std::move(reservation),
          std::make_unique<cucascade::memory::ignore_reservation_limit_policy>(),
          nullptr)) {
      global.add_error("Failed to attach reservation to stream tracker.");
      global.completed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    absl::Cleanup cleanup = [allocator, stream]() { allocator->reset_stream_reservation(stream); };

    void* allocation = nullptr;
    try {
      allocation = allocator->allocate(stream, kAllocationBytes);
    } catch (const rmm::out_of_memory&) {
      global.oom_count.fetch_add(1, std::memory_order_relaxed);
      // Throw oom_reschedule_exception so the executor reschedules this task.
      // Pass the input data through so the rescheduled task can use it.
      throw sirius::pipeline::oom_reschedule_exception(
        std::move(local._input_data), 0, "OOM in test task allocation");
    }

    // Hold the memory for a while to create pressure on concurrent tasks.
    std::this_thread::sleep_for(kHoldDuration);

    allocator->deallocate(stream, allocation, kAllocationBytes);
    global.completed_count.fetch_add(1, std::memory_order_relaxed);
  }

  std::unique_ptr<sirius::op::operator_data> compute_task(rmm::cuda_stream_view) override
  {
    return nullptr;
  }

  void publish_output(sirius::op::operator_data&, rmm::cuda_stream_view) override {}

  std::size_t get_estimated_reservation_size() const override { return kReservationSize; }

  std::vector<sirius::op::sirius_physical_operator*> get_output_consumers() override { return {}; }

  std::unique_ptr<gpu_pipeline_task> create_rescheduled_task(
    uint64_t task_id,
    std::unique_ptr<sirius::pipeline::sirius_pipeline_task_local_state> local_state) override
  {
    auto typed_local = std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state>(
      static_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(local_state.release()));
    return std::make_unique<oom_test_task>(
      task_id,
      std::move(typed_local),
      std::dynamic_pointer_cast<oom_test_global_state>(_global_state));
  }
};

}  // namespace

TEST_CASE("GPU pipeline executor reschedules tasks on OOM", "[gpu_pipeline_executor][oom]")
{
  //--------------------------------------------------------------------------
  // 1. Set up a constrained GPU memory environment (900 MB software limit)
  //--------------------------------------------------------------------------
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  try {
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(kGpuCapacity)
      .set_reservation_fraction_per_gpu(0.95)
      .set_per_host_capacity(1ULL * 1024 * 1024 * 1024)
      .use_host_per_gpu()
      .track_reservation_per_stream(false)
      .set_reservation_fraction_per_host(0.75);
    auto space_configs = builder.build();
    manager =
      std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
  } catch (const std::exception& e) {
    WARN("Skipping OOM reschedule test due to insufficient GPUs: " << e.what());
    return;
  }

  auto* mem_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  if (!mem_space) {
    WARN("Skipping OOM reschedule test because no GPU memory space is available.");
    return;
  }

  //--------------------------------------------------------------------------
  // 2. Create the executor with 3 worker threads so all tasks run concurrently
  //--------------------------------------------------------------------------
  sirius::exec::channel<std::unique_ptr<sirius::pipeline::task_request>> request_channel;
  auto request_publisher = request_channel.make_publisher();

  sirius::exec::thread_pool_config config;
  config.num_threads        = 3;
  config.thread_name_prefix = "oom-test";

  sirius::pipeline::gpu_pipeline_executor executor(config, mem_space, request_publisher);
  auto global_state = std::make_shared<oom_test_global_state>();

  //--------------------------------------------------------------------------
  // 3. Schedule 3 tasks (total demand 3×400MB = 1.2GB > 900MB capacity)
  //--------------------------------------------------------------------------
  const int num_tasks = 3;
  std::atomic<int> dispatched{0};

  executor.start();

  // Thread that responds to task requests from the executor's manager_loop.
  std::thread request_handler([&]() {
    while (dispatched.load(std::memory_order_relaxed) < num_tasks) {
      auto request = request_channel.get();
      if (!request) { break; }

      auto local_state = std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(
        std::make_unique<sirius::op::operator_data>(
          std::vector<std::shared_ptr<cucascade::data_batch>>{}));
      auto task = std::make_unique<oom_test_task>(
        static_cast<uint64_t>(dispatched.load(std::memory_order_relaxed)),
        std::move(local_state),
        global_state);
      executor.schedule(std::move(task));
      dispatched.fetch_add(1, std::memory_order_relaxed);
    }

    // Keep consuming task requests for rescheduled tasks until completion.
    // Rescheduled tasks are already in the executor's queue — we just drain
    // the request channel to prevent the manager_loop from blocking.
    while (global_state->completed_count.load(std::memory_order_relaxed) < num_tasks) {
      auto request = request_channel.get();
      if (!request) { break; }
    }
  });

  //--------------------------------------------------------------------------
  // 4. Wait for all tasks to complete (including rescheduled ones)
  //--------------------------------------------------------------------------
  auto start_time = std::chrono::steady_clock::now();
  auto timeout    = std::chrono::seconds(30);
  while (global_state->completed_count.load(std::memory_order_relaxed) < num_tasks) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      executor.stop();
      request_channel.close();
      request_handler.join();
      FAIL("Timed out waiting for OOM rescheduled tasks to complete. "
           << "Completed: " << global_state->completed_count.load()
           << ", OOM count: " << global_state->oom_count.load());
    }
  }

  executor.stop();
  request_channel.close();
  request_handler.join();

  //--------------------------------------------------------------------------
  // 5. Validate results
  //--------------------------------------------------------------------------
  if (global_state->error_count.load(std::memory_order_relaxed) > 0) {
    std::lock_guard<std::mutex> lock(global_state->error_mutex);
    for (const auto& error : global_state->errors) {
      INFO(error);
    }
  }

  REQUIRE(global_state->error_count.load(std::memory_order_relaxed) == 0);
  REQUIRE(global_state->completed_count.load(std::memory_order_relaxed) == num_tasks);
  // At least one task should have OOM'd and been rescheduled
  REQUIRE(global_state->oom_count.load(std::memory_order_relaxed) >= 1);

  INFO("OOM reschedule test passed: " << global_state->oom_count.load() << " task(s) rescheduled, "
                                      << num_tasks << " completed successfully");
}
