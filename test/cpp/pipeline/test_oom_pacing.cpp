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

// OOM-retry storm pacing tests. A raw "hold" allocation the downgrade executor
// cannot see fills the pool, so every admission is partial, every downgrade
// pass frees 0 bytes, and every execute OOMs until the hold is released.

#include "catch.hpp"
#include "data/data_repository_manager_registry.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "exec/channel.hpp"
#include "exec/config.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_request.hpp"
#include "scan/test_utils.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace {

// Memory layout:
//   GPU capacity        = 1100 MB (software usage limit)
//   Reservation limit   = 0.95 * capacity = 1045 MB
//   Hold allocation     = 1000 MB  -> reservable headroom 45 MB, pool free 100 MB
//   Task reservation    =  200 MB  -> or_null fails; blocking path grants a
//                                     partial (the 45 MB remainder) via the
//                                     IDLE -> reserve_upto escape
//   Task allocation     =  400 MB  -> OOMs while the hold is in place
constexpr std::size_t kGpuCapacity          = 1100ULL * 1024 * 1024;
constexpr std::size_t kHoldBytes            = 1000ULL * 1024 * 1024;
constexpr std::size_t kStormReservationSize = 200ULL * 1024 * 1024;
constexpr std::size_t kStormAllocationBytes = 400ULL * 1024 * 1024;

class pacing_test_global_state : public sirius::pipeline::sirius_pipeline_task_global_state {
 public:
  pacing_test_global_state()
    : sirius_pipeline_task_global_state(nullptr, sirius::test::make_test_telemetry_context())
  {
  }

  void add_error(std::string message)
  {
    error_count.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(error_mutex);
    errors.push_back(std::move(message));
  }

  void record_retry_count(uint32_t retries)
  {
    uint32_t cur = max_retry_count.load(std::memory_order_relaxed);
    while (retries > cur &&
           !max_retry_count.compare_exchange_weak(cur, retries, std::memory_order_relaxed)) {}
  }

  std::atomic<int> completed_count{0};
  std::atomic<int> oom_count{0};
  std::atomic<int> error_count{0};
  std::atomic<uint32_t> max_retry_count{0};
  std::mutex error_mutex;
  std::vector<std::string> errors;
};

struct pacing_test_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* mem_space = nullptr;
  sirius::data::data_repository_manager_registry repo_registry;
  std::unique_ptr<sirius::parallel::downgrade_executor> downgrade;
  sirius::exec::channel<std::unique_ptr<sirius::pipeline::task_request>> request_channel;
  std::unique_ptr<sirius::pipeline::gpu_pipeline_executor> executor;
  std::shared_ptr<sirius::pipeline::completion_handler> completion =
    std::make_shared<sirius::pipeline::completion_handler>();

  // Counts against the pool but holds no reservation and is invisible to the
  // downgrade executor.
  void* hold                                                       = nullptr;
  cucascade::memory::reservation_aware_resource_adaptor* allocator = nullptr;

  bool setup(int num_threads,
             const std::string& thread_name_prefix,
             std::chrono::milliseconds downgrade_cooldown)
  {
    try {
      cucascade::memory::reservation_manager_configurator builder;
      builder.set_number_of_gpus(1)
        .set_gpu_usage_limit(kGpuCapacity)
        .set_reservation_fraction_per_gpu(0.95)
        .set_per_numa_region_capacity(1ULL * 1024 * 1024 * 1024)
        .use_gpu_id_as_host_id()
        .track_reservation_per_stream(false)
        .set_reservation_fraction_per_numa_region(0.75);
      auto space_configs = builder.build();
      manager            = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(
        std::move(space_configs));
    } catch (const std::exception&) {
      return false;
    }

    mem_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
    if (!mem_space) { return false; }
    allocator =
      mem_space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    if (!allocator) { return false; }

    sirius::exec::downgrade_executor_config downgrade_config{
      .thread_pool                 = {.num_threads = 1, .thread_name_prefix = "downgrade"},
      .monitor_period              = std::chrono::milliseconds{0},
      .no_progress_rescan_cooldown = downgrade_cooldown};
    downgrade = std::make_unique<sirius::parallel::downgrade_executor>(
      downgrade_config,
      repo_registry,
      cucascade::memory::memory_space_id(cucascade::memory::Tier::GPU, 0),
      mem_space,
      *manager);
    downgrade->start();

    sirius::exec::thread_pool_config config;
    config.num_threads        = num_threads;
    config.thread_name_prefix = thread_name_prefix;

    executor = std::make_unique<sirius::pipeline::gpu_pipeline_executor>(
      config,
      mem_space,
      request_channel.make_publisher(),
      downgrade.get(),
      sirius::test::make_test_telemetry_context());
    return true;
  }

  void acquire_hold()
  {
    hold = allocator->allocate(cudf::get_default_stream(), kHoldBytes, alignof(std::max_align_t));
  }

  void release_hold()
  {
    if (hold) {
      allocator->deallocate(
        cudf::get_default_stream(), hold, kHoldBytes, alignof(std::max_align_t));
      hold = nullptr;
    }
  }

  void teardown()
  {
    if (executor) { executor->stop(); }
    request_channel.close();
    if (downgrade) { downgrade->stop(); }
    release_hold();
  }
};

// Reserves more than the headroom left by the hold, then allocates more than the
// free pool, so execute OOMs until the hold is released.
class storm_task : public sirius::pipeline::gpu_pipeline_task {
 public:
  storm_task(uint64_t task_id,
             std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state> local_state,
             std::shared_ptr<pacing_test_global_state> global_state)
    : gpu_pipeline_task(task_id,
                        std::vector<cucascade::shared_data_repository*>{},
                        std::move(local_state),
                        std::move(global_state))
  {
  }

  std::unique_ptr<sirius::op::operator_data> compute_task(rmm::cuda_stream_view) override
  {
    return nullptr;
  }

  void publish_output(sirius::op::operator_data&, rmm::cuda_stream_view) override {}

  sirius::pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space* /*target_space*/) const override
  {
    sirius::pipeline::reservation_size_info info;
    info.reservation_size = kStormReservationSize;
    return info;
  }

  std::vector<sirius::op::sirius_physical_operator*> get_output_consumers() override { return {}; }

  void execute(rmm::cuda_stream_view stream) override
  {
    auto& global = _global_state->cast<pacing_test_global_state>();
    auto& local  = _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>();

    auto reservation = local.release_reservation();
    if (!reservation) {
      global.add_error("Missing GPU memory reservation for storm task.");
      global.completed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    auto* allocator =
      reservation->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    if (!allocator) {
      global.add_error("Missing reservation-aware allocator for storm task.");
      global.completed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    // Released at scope exit so backed-off tasks hold no reservation while sleeping.
    auto reservation_guard = std::move(reservation);

    void* allocation = nullptr;
    try {
      allocation = allocator->allocate(stream, kStormAllocationBytes, alignof(std::max_align_t));
    } catch (const rmm::out_of_memory&) {
      global.oom_count.fetch_add(1, std::memory_order_relaxed);
      throw sirius::pipeline::oom_reschedule_exception(
        std::move(local._input_data), 0, "OOM in storm task allocation");
    }

    std::this_thread::sleep_for(10ms);
    allocator->deallocate(stream, allocation, kStormAllocationBytes, alignof(std::max_align_t));
    global.record_retry_count(local.retry_count);
    global.completed_count.fetch_add(1, std::memory_order_relaxed);
  }

  std::unique_ptr<gpu_pipeline_task> create_rescheduled_task(
    uint64_t task_id,
    std::unique_ptr<sirius::pipeline::sirius_pipeline_task_local_state> local_state) override
  {
    auto typed_local = std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state>(
      static_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(local_state.release()));
    return std::make_unique<storm_task>(
      task_id,
      std::move(typed_local),
      std::dynamic_pointer_cast<pacing_test_global_state>(_global_state));
  }
};

std::unique_ptr<storm_task> make_storm_task(
  uint64_t task_id, const std::shared_ptr<pacing_test_global_state>& global_state)
{
  auto local_state = std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(
    std::make_unique<sirius::op::pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{}));
  return std::make_unique<storm_task>(task_id, std::move(local_state), global_state);
}

}  // namespace

TEST_CASE("compute_oom_backoff escalates with the starved streak and caps",
          "[gpu_pipeline_executor][oom_pacing]")
{
  using sirius::pipeline::gpu_pipeline_executor;
  CHECK(gpu_pipeline_executor::compute_oom_backoff(0) == 50ms);  // clean admission: base
  CHECK(gpu_pipeline_executor::compute_oom_backoff(1) == 100ms);
  CHECK(gpu_pipeline_executor::compute_oom_backoff(2) == 200ms);
  CHECK(gpu_pipeline_executor::compute_oom_backoff(3) == 400ms);
  CHECK(gpu_pipeline_executor::compute_oom_backoff(4) == 800ms);
  CHECK(gpu_pipeline_executor::compute_oom_backoff(5) == 800ms);     // capped
  CHECK(gpu_pipeline_executor::compute_oom_backoff(1000) == 800ms);  // no overflow at the cap
}

TEST_CASE("OOM-retry storm is paced: bounded retries, coalesced downgrades, prompt recovery",
          "[gpu_pipeline_executor][oom_pacing]")
{
  pacing_test_fixture f;
  if (!f.setup(2, "oom-pacing", /*downgrade_cooldown=*/250ms)) {
    WARN("Skipping OOM pacing storm test — no GPU available.");
    return;
  }

  auto global_state = std::make_shared<pacing_test_global_state>();
  global_state->set_completion_handler(f.completion);
  f.acquire_hold();
  f.executor->start();

  constexpr int num_tasks = 4;
  for (int i = 0; i < num_tasks; ++i) {
    f.executor->schedule(make_storm_task(static_cast<uint64_t>(i), global_state));
  }

  constexpr auto kStormDuration = 1200ms;
  std::this_thread::sleep_for(kStormDuration);

  auto const mid_stats = f.executor->get_oom_pacing_stats();
  REQUIRE(global_state->completed_count.load() == 0);
  REQUIRE(mid_stats.oom_reschedules >= num_tasks);
  REQUIRE(mid_stats.starved_admissions >= 1);
  REQUIRE(mid_stats.backoff_events >= 1);
  REQUIRE(f.downgrade->no_progress_passes() >= 1);
  REQUIRE(f.downgrade->coalesced_requests() >= 1);

  // Releasing the hold must wake the backed-off tasks promptly.
  auto const release_time = std::chrono::steady_clock::now();
  f.release_hold();

  auto const deadline = release_time + 10s;
  while (global_state->completed_count.load(std::memory_order_relaxed) < num_tasks &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  auto const recovery = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now() - release_time);

  {
    std::lock_guard<std::mutex> lock(global_state->error_mutex);
    for (const auto& error : global_state->errors) {
      INFO(error);
    }
  }
  REQUIRE(global_state->error_count.load() == 0);
  REQUIRE(global_state->completed_count.load() == num_tasks);

  REQUIRE(recovery < 3000ms);

  // ~1.2 s of storm allows ~5 escalated attempts per task; unpaced it would be 20+.
  REQUIRE(global_state->max_retry_count.load() > 0);
  REQUIRE(global_state->max_retry_count.load() <= 10);

  f.teardown();
}

TEST_CASE("clean-admission OOMs keep the base retry interval (no escalation)",
          "[gpu_pipeline_executor][oom_pacing]")
{
  // No hold: reservations are granted in full, so contention OOMs must not escalate.
  pacing_test_fixture f;
  if (!f.setup(3, "oom-clean", /*downgrade_cooldown=*/250ms)) {
    WARN("Skipping clean-admission OOM test — no GPU available.");
    return;
  }

  auto global_state = std::make_shared<pacing_test_global_state>();
  global_state->set_completion_handler(f.completion);
  f.executor->start();

  // 3 workers x 400 MB against an 1100 MB pool: the third allocation OOMs.
  constexpr int num_tasks = 6;
  for (int i = 0; i < num_tasks; ++i) {
    f.executor->schedule(make_storm_task(static_cast<uint64_t>(i), global_state));
  }

  auto const deadline = std::chrono::steady_clock::now() + 30s;
  while (global_state->completed_count.load(std::memory_order_relaxed) < num_tasks &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  REQUIRE(global_state->completed_count.load() == num_tasks);
  REQUIRE(global_state->error_count.load() == 0);

  auto const stats = f.executor->get_oom_pacing_stats();
  REQUIRE(stats.starved_admissions == 0);
  REQUIRE(stats.backoff_events == 0);

  f.teardown();
}

TEST_CASE("executor stop is prompt while tasks are in escalated backoff",
          "[gpu_pipeline_executor][oom_pacing]")
{
  pacing_test_fixture f;
  if (!f.setup(2, "oom-stop", /*downgrade_cooldown=*/250ms)) {
    WARN("Skipping backoff stop test — no GPU available.");
    return;
  }

  auto global_state = std::make_shared<pacing_test_global_state>();
  global_state->set_completion_handler(f.completion);
  f.acquire_hold();
  f.executor->start();

  for (int i = 0; i < 2; ++i) {
    f.executor->schedule(make_storm_task(static_cast<uint64_t>(i), global_state));
  }

  // Wait until at least one task is in an escalated backoff sleep.
  auto const arm_deadline = std::chrono::steady_clock::now() + 10s;
  while (f.executor->get_oom_pacing_stats().backoff_events < 1 &&
         std::chrono::steady_clock::now() < arm_deadline) {
    std::this_thread::sleep_for(10ms);
  }
  REQUIRE(f.executor->get_oom_pacing_stats().backoff_events >= 1);

  // stop() must not wait out the full backoff of each sleeper.
  auto const stop_start = std::chrono::steady_clock::now();
  f.executor->stop();
  auto const stop_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now() - stop_start);
  REQUIRE(stop_ms < 2000ms);

  f.teardown();
}
