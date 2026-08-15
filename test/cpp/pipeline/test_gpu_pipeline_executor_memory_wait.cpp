/*
 * Copyright 2026, Sirius Contributors.
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

/**
 * @file test_gpu_pipeline_executor_memory_wait.cpp
 * @brief The GPU manager thread never blocks on memory (register issue C4).
 *
 * There is exactly ONE manager thread per GPU executor. It used to perform the blocking
 * make_reservation (and the downgrade .get()) itself while holding a reserved pool slot, so one
 * task's memory wait stalled EVERY query's dispatch to that device — the F1 fair pops rotated
 * the queue, but a fair pop still stalled behind the manager's blocked reserve.
 *
 * After the fix the wait parks on a pool worker (at most one per executor; overflow waiters
 * re-queue with a backoff), and the manager keeps dispatching. These cases drive a real GPU
 * memory space with a test-held reservation so a "hungry" task provably has to wait, and assert
 * that small co-tenant tasks scheduled AFTER it still execute while it waits — the exact
 * sequence that deadlocked dispatch before the fix.
 */

#include "catch.hpp"
#include "exec/channel.hpp"
#include "exec/config.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_request.hpp"
#include "utils/telemetry_utils.hpp"

#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace {

using steady_clock = std::chrono::steady_clock;

constexpr std::size_t kMiB = 1024 * 1024;

//! Records which tasks executed and when. Tasks carry no pipeline, so every failure surfaces
//! through `errors` rather than a completion handler.
class memory_wait_global_state : public sirius::pipeline::sirius_pipeline_task_global_state {
 public:
  memory_wait_global_state()
    : sirius_pipeline_task_global_state(nullptr, sirius::test::make_test_telemetry_context())
  {
  }

  void record(uint64_t task_id)
  {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _completions.emplace_back(task_id, steady_clock::now());
    }
    executed_count.fetch_add(1, std::memory_order_relaxed);
  }

  [[nodiscard]] bool has_executed(uint64_t task_id)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return std::any_of(_completions.begin(), _completions.end(), [&](const auto& entry) {
      return entry.first == task_id;
    });
  }

  [[nodiscard]] std::vector<std::pair<uint64_t, steady_clock::time_point>> completions()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _completions;
  }

  std::atomic<int> executed_count{0};

 private:
  std::mutex _mutex;
  std::vector<std::pair<uint64_t, steady_clock::time_point>> _completions;
};

//! A task whose reservation demand the test controls. execute() drops the reservation (freeing
//! the memory, which wakes any parked waiter) and records its completion.
class memory_wait_task : public sirius::pipeline::gpu_pipeline_task {
 public:
  memory_wait_task(uint64_t task_id,
                   std::size_t reservation_bytes,
                   std::shared_ptr<memory_wait_global_state> global_state)
    : gpu_pipeline_task(task_id,
                        std::vector<std::shared_ptr<cucascade::shared_data_repository>>{},
                        std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(
                          std::make_unique<sirius::op::pipelineable_operator_data>(
                            std::vector<std::shared_ptr<cucascade::data_batch>>{})),
                        std::move(global_state)),
      _reservation_bytes(reservation_bytes)
  {
  }

  void execute(rmm::cuda_stream_view /*stream*/) override
  {
    auto& global = _global_state->cast<memory_wait_global_state>();
    auto& local  = _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>();
    // Dropping the reservation returns its bytes to the space and posts the release
    // notification a parked waiter sleeps on.
    auto reservation = local.release_reservation();
    reservation.reset();
    global.record(get_task_id());
  }

  sirius::pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space* /*target_space*/) const override
  {
    sirius::pipeline::reservation_size_info info;
    info.reservation_size = _reservation_bytes;
    return info;
  }

  std::vector<sirius::op::sirius_physical_operator*> get_output_consumers() override { return {}; }

 private:
  std::size_t _reservation_bytes;
};

//! One GPU memory space (small pool) plus an executor with 2 workers and no downgrade
//! executor, so a reservation that does not fit has to WAIT for a release.
struct memory_wait_fixture {
  bool valid = false;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* mem_space = nullptr;
  sirius::exec::channel<std::unique_ptr<sirius::pipeline::task_request>> request_channel;
  std::unique_ptr<sirius::pipeline::gpu_pipeline_executor> executor;
  std::shared_ptr<memory_wait_global_state> global_state;

  memory_wait_fixture()
  {
    try {
      cucascade::memory::reservation_manager_configurator builder;
      builder.set_number_of_gpus(1)
        .set_gpu_usage_limit(256 * kMiB)
        .set_reservation_fraction_per_gpu(0.75)
        .set_per_numa_region_capacity(1024 * kMiB)
        .use_gpu_id_as_host_id()
        .track_reservation_per_stream(false)
        .set_reservation_fraction_per_numa_region(0.75);
      manager =
        std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    } catch (const std::exception& e) {
      WARN("Skipping memory-wait test (no usable GPU): " << e.what());
      return;
    }
    mem_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
    if (!mem_space) {
      WARN("Skipping memory-wait test: no GPU memory space available.");
      return;
    }

    sirius::exec::thread_pool_config config;
    config.num_threads        = 2;
    config.thread_name_prefix = "gpu-memwait-test";

    executor = std::make_unique<sirius::pipeline::gpu_pipeline_executor>(
      config,
      mem_space,
      request_channel.make_publisher(),
      nullptr,
      sirius::test::make_test_telemetry_context());
    global_state = std::make_shared<memory_wait_global_state>();
    valid        = true;
  }

  ~memory_wait_fixture()
  {
    if (executor) { executor->stop(); }
    request_channel.close();
  }

  void schedule(uint64_t task_id, std::size_t reservation_bytes)
  {
    executor->schedule(
      std::make_unique<memory_wait_task>(task_id, reservation_bytes, global_state));
  }

  [[nodiscard]] bool wait_until(const std::function<bool()>& done, std::chrono::seconds deadline)
  {
    const auto give_up = steady_clock::now() + deadline;
    while (!done()) {
      if (steady_clock::now() > give_up) { return false; }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
  }
};

constexpr uint64_t kHungryTaskId       = 1000;
constexpr uint64_t kSecondHungryTaskId = 1001;

}  // namespace

TEST_CASE("C4: a waiting reservation parks on a worker; co-tenant tasks keep dispatching",
          "[gpu_pipeline_executor][memory_wait][concurrency]")
{
  memory_wait_fixture f;
  if (!f.valid) { return; }
  f.executor->start();

  // Occupy the space so that only a small head room is grantable. The hold's reservation keeps
  // a release-notifier active, so a parked make_reservation waits (no IDLE fallback) until a
  // release actually happens.
  const std::size_t space_max = f.mem_space->get_max_memory();
  const std::size_t head_room = 8 * kMiB;
  REQUIRE(space_max > 4 * head_room);
  auto hold = f.mem_space->make_reservation_or_null(space_max - head_room);
  REQUIRE(hold);
  REQUIRE(hold->size() == space_max - head_room);

  // The hungry task demands more than the head room: it must WAIT until `hold` is released.
  f.schedule(kHungryTaskId, space_max / 2);

  // Give the executor time to pop it and park in the wait. Pre-fix, this is the moment the
  // manager thread blocked in make_reservation while holding its reserved slot.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  REQUIRE_FALSE(f.global_state->has_executed(kHungryTaskId));

  // Short co-tenant tasks scheduled AFTER the hungry one. They fit the head room; the only
  // question is whether anything still dispatches them.
  constexpr int kShortTasks = 4;
  const auto shorts_start   = steady_clock::now();
  for (uint64_t i = 0; i < kShortTasks; ++i) {
    f.schedule(i, 1 * kMiB);
  }

  const bool shorts_done = f.wait_until(
    [&] { return f.global_state->executed_count.load(std::memory_order_relaxed) >= kShortTasks; },
    std::chrono::seconds(30));

  // Dispatch-latency evidence for the report: time from scheduling the shorts to each short's
  // completion, with the hungry task still parked. Pre-fix these never complete (the manager
  // holds the dispatch loop inside a blocked make_reservation), so shorts_done times out.
  double first_short_ms = -1.0;
  double last_short_ms  = -1.0;
  for (const auto& [task_id, when] : f.global_state->completions()) {
    if (task_id >= kHungryTaskId) { continue; }
    const double ms = std::chrono::duration<double, std::milli>(when - shorts_start).count();
    if (first_short_ms < 0.0 || ms < first_short_ms) { first_short_ms = ms; }
    last_short_ms = std::max(last_short_ms, ms);
  }
  INFO("first_short_ms=" << first_short_ms << " last_short_ms=" << last_short_ms << " requeues="
                         << f.executor->get_metrics().tasks_requeued_on_memory_wait);
  REQUIRE(shorts_done);
  // The hungry task must still be waiting — the shorts really did overtake a parked wait.
  REQUIRE_FALSE(f.global_state->has_executed(kHungryTaskId));

  // Release the hold: the parked waiter's reservation becomes grantable and it completes.
  hold.reset();
  REQUIRE(f.wait_until([&] { return f.global_state->has_executed(kHungryTaskId); },
                       std::chrono::seconds(30)));

  f.executor->stop();
}

TEST_CASE("C4: overflow memory waiters re-queue instead of filling every worker slot",
          "[gpu_pipeline_executor][memory_wait][concurrency]")
{
  memory_wait_fixture f;
  if (!f.valid) { return; }
  f.executor->start();

  const std::size_t space_max = f.mem_space->get_max_memory();
  const std::size_t head_room = 8 * kMiB;
  REQUIRE(space_max > 4 * head_room);
  auto hold = f.mem_space->make_reservation_or_null(space_max - head_room);
  REQUIRE(hold);

  // TWO hungry tasks and only ONE memory-wait slot per executor: the first parks, the second
  // must re-queue with a backoff rather than park a wait in the last worker slot. With both
  // parked (no cap), the 2-thread pool would have no slot left and the shorts below would
  // starve exactly as they did pre-fix — one level below the manager.
  f.schedule(kHungryTaskId, space_max / 2);
  f.schedule(kSecondHungryTaskId, space_max / 2);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  REQUIRE_FALSE(f.global_state->has_executed(kHungryTaskId));
  REQUIRE_FALSE(f.global_state->has_executed(kSecondHungryTaskId));

  constexpr int kShortTasks = 4;
  for (uint64_t i = 0; i < kShortTasks; ++i) {
    f.schedule(i, 1 * kMiB);
  }

  REQUIRE(f.wait_until(
    [&] { return f.global_state->executed_count.load(std::memory_order_relaxed) >= kShortTasks; },
    std::chrono::seconds(30)));
  REQUIRE_FALSE(f.global_state->has_executed(kHungryTaskId));
  REQUIRE_FALSE(f.global_state->has_executed(kSecondHungryTaskId));

  // The overflow waiter went through the re-queue path at least once.
  REQUIRE(f.executor->get_metrics().tasks_requeued_on_memory_wait >= 1);

  hold.reset();
  REQUIRE(f.wait_until(
    [&] {
      return f.global_state->has_executed(kHungryTaskId) &&
             f.global_state->has_executed(kSecondHungryTaskId);
    },
    std::chrono::seconds(30)));

  f.executor->stop();
}
