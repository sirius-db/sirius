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
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_request.hpp"
#include "scan/test_utils.hpp"
#include "utils/telemetry_utils.hpp"

#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
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

  std::atomic<int> estimated_count{0};
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

  void execute(::cuda::stream_ref stream) override
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
    _global_state->cast<test_gpu_pipeline_task_global_state>().estimated_count.fetch_add(1);
    sirius::pipeline::reservation_size_info info;
    info.reservation_size = kReservationBytes;
    return info;
  }

  std::vector<sirius::op::sirius_physical_operator*> get_output_consumers() override { return {}; }
};

// Hold the production manager before it can pop any staging task. The gate is test-local;
// no production callback or alternative manager protocol is needed to control the interleaving.
class paused_gpu_executor : public sirius::pipeline::gpu_pipeline_executor {
 public:
  using gpu_pipeline_executor::gpu_pipeline_executor;

  ~paused_gpu_executor() override
  {
    resume_manager_for_test();
    stop();
  }

  void resume_manager_for_test()
  {
    if (!resumed_) {
      resumed_ = true;
      resume_.set_value();
    }
  }

 protected:
  void manager_loop() override
  {
    ready_to_run_.wait();
    gpu_pipeline_executor::manager_loop();
  }

 private:
  std::promise<void> resume_;
  std::future<void> ready_to_run_{resume_.get_future()};
  bool resumed_{false};
};

}  // namespace

TEST_CASE("GPU cancellation preserves staged work until the manager restores readiness",
          "[gpu_pipeline_executor][query_lifecycle_gate][concurrency]")
{
  using namespace std::chrono_literals;
  const auto cancelled_query = sirius::make_query_id(7);
  const auto next_query      = sirius::make_query_id(0);  // Detached test tasks use query 0.
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(cancelled_query);
  lifecycle.open_query(next_query);
  auto manager    = initialize_memory_manager(1);
  auto* mem_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(mem_space);

  // Supply a real query identity and a valid operator for task destruction callbacks.
  sirius::op::sirius_physical_operator source(
    sirius::op::SiriusPhysicalOperatorType::FILTER, {}, 0);
  auto pipe = std::make_shared<sirius::pipeline::sirius_pipeline>(
    sirius::pipeline::pipeline_build_context{sirius::test::make_test_telemetry_context()});
  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*pipe, source);
  build_state.set_pipeline_sink(*pipe, &source, 1);
  pipe->set_query_id(cancelled_query);
  auto cancelled_state = std::make_shared<test_gpu_pipeline_task_global_state>();
  cancelled_state->set_pipeline(pipe);
  auto next_state = std::make_shared<test_gpu_pipeline_task_global_state>();
  auto make_task  = [](auto state) {
    return std::make_unique<sirius_pipeline_task>(
      1,
      std::make_unique<test_gpu_pipeline_task_local_state>(
        std::make_unique<sirius::op::pipelineable_operator_data>(
          std::vector<std::shared_ptr<cucascade::data_batch>>{})),
      std::move(state));
  };

  sirius::exec::channel<std::unique_ptr<sirius::pipeline::task_request>> requests;
  paused_gpu_executor executor(lifecycle,
                               {1, "cancel-staging"},
                               mem_space,
                               requests.make_publisher(),
                               nullptr,
                               sirius::test::make_test_telemetry_context());
  executor.start();
  REQUIRE(executor.schedule(make_task(cancelled_state)));
  REQUIRE(lifecycle.activity(cancelled_query).work == 1);

  // Freeze the queue-insertion-before-pop interleaving. In production the scheduler has
  // consumed readiness for this accepted handoff; cancellation must not remove its task.
  lifecycle.quiesce_and_wait_for_submissions(cancelled_query);
  executor.wait_and_drain_query(cancelled_query);
  CHECK_FALSE(executor.is_task_queue_empty());
  CHECK(lifecycle.activity(cancelled_query).work == 1);
  executor.resume_manager_for_test();

  auto take_readiness = [&] {
    std::unique_ptr<sirius::pipeline::task_request> request;
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!(request = requests.try_get()) && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
    }
    return request;
  };
  // The first announcement belongs to the staged handoff. The second proves the real manager
  // popped and discarded the cancelled task, released its slot, and entered its next iteration.
  for (int i = 0; i < 2; ++i) {
    auto request = take_readiness();
    REQUIRE(request);
    CHECK(request->kind == sirius::pipeline::task_request_kind::device_ready);
    CHECK(request->device_id == 0);
  }
  CHECK(cancelled_state->estimated_count.load() == 0);
  CHECK(cancelled_state->executed_count.load() == 0);
  REQUIRE(lifecycle.activity(cancelled_query).work == 0);
  lifecycle.wait_for_work(cancelled_query);
  lifecycle.close(cancelled_query);

  // Use the restored readiness to run a different query without restarting the executor.
  REQUIRE(executor.schedule(make_task(next_state)));
  auto request = take_readiness();
  REQUIRE(request);
  CHECK(request->kind == sirius::pipeline::task_request_kind::device_ready);
  // With one worker slot, this announcement follows completion of the task and its epilogue.
  CHECK(next_state->executed_count.load() == 1);
  CHECK(next_state->error_count.load() == 0);
  REQUIRE(lifecycle.activity(next_query).work == 0);
  lifecycle.quiesce_and_wait_for_submissions(next_query);
  lifecycle.wait_for_work(next_query);
  lifecycle.close(next_query);
  executor.stop();
}

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

  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(sirius::make_query_id(0));
  sirius::pipeline::gpu_pipeline_executor executor(lifecycle,
                                                   config,
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

TEST_CASE("reservation wait excludes scheduler delay and refreshes on progress", "[memory_wait]")
{
  using namespace std::chrono_literals;
  using wait_type = sirius::memory::reservation_wait;
  wait_type wait;
  auto now = wait_type::clock::time_point{};
  REQUIRE(wait.retry(now, 0, 0, 20ms));
  CHECK(wait.retry_at() == now + 5ms);
  // An hour queued behind other work charges only the requested 5 ms backoff.
  now += 1h;
  REQUIRE(wait.retry(now, 0, 0, 20ms));
  CHECK(wait.retry_at() == now + 10ms);
  now = wait.retry_at();
  REQUIRE(wait.retry(now, 0, 0, 20ms));
  SECTION("persistent exhaustion consumes the retry budget")
  {
    CHECK_FALSE(wait.retry(wait.retry_at(), 0, 0, 20ms));
  }
  SECTION("released capacity refreshes the budget")
  {
    CHECK(wait.retry(wait.retry_at(), 1, 0, 20ms));
  }
  SECTION("completed work refreshes it even if another task took the freed memory")
  {
    CHECK(wait.retry(wait.retry_at(), 0, 1, 20ms));
  }
  SECTION("moving to another device starts a new pressure episode")
  {
    CHECK(wait.retry(wait.retry_at(), 0, 0, 20ms, 1));
  }
  wait.reset();
  CHECK_FALSE(wait.waiting());
  CHECK(wait.retry(now + 2h, 0, 0, 20ms));
}

TEST_CASE("reservation retries back off without exceeding fifty milliseconds", "[memory_wait]")
{
  using namespace std::chrono_literals;
  sirius::memory::reservation_wait wait;
  auto now = sirius::memory::reservation_wait::clock::time_point{};
  for (auto delay : {5ms, 10ms, 20ms, 40ms, 50ms, 50ms}) {
    REQUIRE(wait.retry(now, 0, 0, 1s));
    CHECK(wait.retry_at() - now == delay);
    now = wait.retry_at();
  }
}
