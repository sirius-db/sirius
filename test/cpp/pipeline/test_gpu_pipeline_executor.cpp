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
#include "pipeline/task_request.hpp"
#include "scan/test_utils.hpp"

#include <rmm/mr/device/device_memory_resource.hpp>

#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kReservationBytes = 20 * 1024 * 1024;
constexpr std::size_t kAllocationBytes  = 10 * 1024 * 1024;

class test_gpu_pipeline_task_global_state
  : public sirius::pipeline::gpu_pipeline_task_global_state {
 public:
  test_gpu_pipeline_task_global_state() : gpu_pipeline_task_global_state(nullptr) {}

  void add_error(std::string message)
  {
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

  void execute() override
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

    auto stream = mem_space.acquire_stream();
    if (!allocator->attach_reservation_to_tracker(stream, std::move(reservation))) {
      global.add_error("Failed to attach reservation to stream tracker.");
      global.executed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    void* allocation = nullptr;
    try {
      allocation = allocator->allocate(kAllocationBytes, stream);
    } catch (const std::exception& e) {
      global.add_error(std::string("GPU allocation failed: ") + e.what());
      allocator->reset_stream_reservation(stream);
      global.executed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    allocator->deallocate(allocation, kAllocationBytes, stream);

    auto consumed_bytes = mem_space.get_total_reserved_memory();
    {
      std::lock_guard<std::mutex> lock(global.memory_mutex);
      global.memory_consumption.push_back(consumed_bytes);
    }

    allocator->reset_stream_reservation(stream);
    global.executed_count.fetch_add(1, std::memory_order_relaxed);
  }

  std::size_t get_estimated_reservation_size() const override { return kReservationBytes; }

  std::vector<sirius::op::sirius_physical_operator*> get_output_consumers() override { return {}; }
};

}  // namespace

TEST_CASE("GPU pipeline executor uses task requests to schedule GPU tasks",
          "[gpu_pipeline_executor]")
{
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  try {
    manager = initialize_memory_manager(1);
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

  sirius::pipeline::gpu_pipeline_executor executor(config, mem_space, request_publisher);
  auto global_state = std::make_shared<test_gpu_pipeline_task_global_state>();

  const int num_tasks = 10;
  std::atomic<int> dispatched{0};

  executor.start();

  std::thread request_handler([&]() {
    while (dispatched.load(std::memory_order_relaxed) < num_tasks) {
      auto request = request_channel.get();
      if (!request) { break; }

      auto local_state = std::make_unique<test_gpu_pipeline_task_local_state>(
        std::vector<std::shared_ptr<cucascade::data_batch>>{});
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
