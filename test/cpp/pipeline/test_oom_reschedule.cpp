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
#include "data/data_repository_manager_registry.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "exec/channel.hpp"
#include "exec/config.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/retry_futility.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_request.hpp"
#include "scan/test_utils.hpp"
#include "utils/telemetry_utils.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

// Memory layout constants shared across all tests.
//   GPU capacity  = 1100 MB (software limit)
//   Reservation   =  50 MB per task (intentionally small)
constexpr std::size_t kGpuCapacity     = 1100ULL * 1024 * 1024;  // 1100 MB
constexpr std::size_t kReservationSize = 50ULL * 1024 * 1024;    // 50 MB

//------------------------------------------------------------------------------
// Test global state — shared across all tasks
//------------------------------------------------------------------------------
class oom_test_global_state : public sirius::pipeline::sirius_pipeline_task_global_state {
 public:
  oom_test_global_state()
    : sirius_pipeline_task_global_state(nullptr, sirius::test::make_test_telemetry_context())
  {
  }

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
// Test fixture: memory manager + executor + channel, reused by all test cases.
//------------------------------------------------------------------------------
struct oom_test_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* mem_space = nullptr;
  sirius::exec::channel<std::unique_ptr<sirius::pipeline::task_request>> request_channel;
  // Declared before `executor` so they outlive it: the executor keeps raw pointers to the
  // downgrade executor and shares ownership of the progress signals.
  std::unique_ptr<sirius::data::data_repository_manager_registry> repo_registry;
  std::unique_ptr<sirius::parallel::downgrade_executor> downgrade;
  std::shared_ptr<sirius::pipeline::execution_progress> progress =
    std::make_shared<sirius::pipeline::execution_progress>();
  std::unique_ptr<sirius::pipeline::gpu_pipeline_executor> executor;
  sirius::pipeline::completion_handler completion;

  // Returns false if setup failed (no GPU available) — caller should WARN and return.
  // With `with_downgrade_executor` the executor gets a real downgrade_executor over an empty
  // repository registry, so a partial-grant gate runs the production downgrade round trip and
  // sees 0 bytes freed (the analogue of parked fragment output the sweep cannot reach).
  bool setup(int num_threads,
             const std::string& thread_name_prefix,
             bool with_downgrade_executor = false)
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

    auto request_publisher = request_channel.make_publisher();

    sirius::exec::thread_pool_config config;
    config.num_threads        = num_threads;
    config.thread_name_prefix = thread_name_prefix;

    if (with_downgrade_executor) {
      repo_registry = std::make_unique<sirius::data::data_repository_manager_registry>();
      sirius::exec::downgrade_executor_config dg_config{
        .thread_pool    = {.num_threads = 1, .thread_name_prefix = "downgrade"},
        .monitor_period = std::chrono::milliseconds{0}};
      downgrade = std::make_unique<sirius::parallel::downgrade_executor>(
        dg_config,
        *repo_registry,
        cucascade::memory::memory_space_id(cucascade::memory::Tier::GPU, 0),
        mem_space,
        *manager);
      downgrade->start();
    }

    executor = std::make_unique<sirius::pipeline::gpu_pipeline_executor>(
      config,
      mem_space,
      std::move(request_publisher),
      downgrade.get(),
      sirius::test::make_test_telemetry_context(),
      progress);
    executor->set_completion_handler(&completion);
    return true;
  }
};

// Memory the downgrade sweep cannot see: a plain allocation on the space's default allocator,
// outside any task reservation (the analogue of parked STREAMING_SINK output). With the 1100 MB
// capacity and 0.95 reservation fraction (limit 1045 MB), a 900 MB hog leaves an "upto" grant of
// exactly 145 MB and 55 MB of unreservable overflow headroom.
constexpr std::size_t kHogBytes = 900ULL * 1024 * 1024;  // 900 MB

std::unique_ptr<rmm::device_buffer> make_hog(cucascade::memory::memory_space* mem_space)
{
  return std::make_unique<rmm::device_buffer>(
    kHogBytes, rmm::cuda_stream_default, mem_space->get_default_allocator());
}

//------------------------------------------------------------------------------
// Base class for test tasks — handles the common constructor, trivial overrides,
// and the reservation-setup boilerplate shared by all task types.
//------------------------------------------------------------------------------
class oom_test_task_base : public sirius::pipeline::gpu_pipeline_task {
 public:
  oom_test_task_base(uint64_t task_id,
                     std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state> local_state,
                     std::shared_ptr<oom_test_global_state> global_state)
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
    info.reservation_size = kReservationSize;
    return info;
  }

  std::vector<sirius::op::sirius_physical_operator*> get_output_consumers() override { return {}; }

 protected:
  // RAII guard that resets the stream reservation on destruction.
  struct allocator_guard {
    cucascade::memory::reservation_aware_resource_adaptor* allocator;
    rmm::cuda_stream_view stream;

    allocator_guard(cucascade::memory::reservation_aware_resource_adaptor* a,
                    rmm::cuda_stream_view s)
      : allocator(a), stream(s)
    {
    }
    ~allocator_guard() { allocator->reset_stream_reservation(stream); }

    allocator_guard(const allocator_guard&)            = delete;
    allocator_guard& operator=(const allocator_guard&) = delete;
    allocator_guard(allocator_guard&&)                 = delete;
    allocator_guard& operator=(allocator_guard&&)      = delete;
  };

  // Performs the common reservation → allocator → attach → cleanup setup.
  // Returns the allocator guard on success, or nullptr on failure (after
  // recording an error on global_state).
  std::unique_ptr<allocator_guard> setup_allocator(rmm::cuda_stream_view stream,
                                                   const std::string& task_label)
  {
    auto& global = _global_state->cast<oom_test_global_state>();
    auto& local  = _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>();

    auto reservation = local.release_reservation();
    if (!reservation) {
      global.add_error("Missing GPU memory reservation for " + task_label + ".");
      return nullptr;
    }

    auto* allocator =
      reservation->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    if (!allocator) {
      global.add_error("Missing reservation-aware allocator for " + task_label + ".");
      return nullptr;
    }

    if (!allocator->attach_reservation_to_tracker(
          stream,
          std::move(reservation),
          std::make_unique<cucascade::memory::ignore_reservation_limit_policy>(),
          nullptr)) {
      global.add_error("Failed to attach reservation to stream tracker for " + task_label + ".");
      return nullptr;
    }

    return std::make_unique<allocator_guard>(allocator, stream);
  }
};

//------------------------------------------------------------------------------
// Task that allocates kAllocationBytes of GPU memory, holds it, then frees.
// If OOM occurs, throws oom_reschedule_exception for the executor to handle.
//------------------------------------------------------------------------------
constexpr std::size_t kAllocationBytes = 400ULL * 1024 * 1024;  // 400 MB
constexpr auto kHoldDuration           = std::chrono::milliseconds(30);

class oom_test_task : public oom_test_task_base {
 public:
  using oom_test_task_base::oom_test_task_base;

  void execute(rmm::cuda_stream_view stream) override
  {
    auto& global = _global_state->cast<oom_test_global_state>();
    auto& local  = _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>();

    auto guard = setup_allocator(stream, "OOM test task");
    if (!guard) {
      global.completed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    auto* allocator = guard->allocator;

    void* allocation = nullptr;
    try {
      allocation = allocator->allocate(stream, kAllocationBytes, alignof(std::max_align_t));
    } catch (const rmm::out_of_memory&) {
      global.oom_count.fetch_add(1, std::memory_order_relaxed);
      throw sirius::pipeline::oom_reschedule_exception(
        std::move(local._input_data), 0, "OOM in test task allocation");
    }

    // Hold the memory for a while to create pressure on concurrent tasks.
    std::this_thread::sleep_for(kHoldDuration);

    allocator->deallocate(stream, allocation, kAllocationBytes, alignof(std::max_align_t));
    global.completed_count.fetch_add(1, std::memory_order_relaxed);
  }

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

//------------------------------------------------------------------------------
// Small task — allocates a tiny amount of GPU memory and completes immediately.
//------------------------------------------------------------------------------
constexpr std::size_t kSmallAllocationBytes = 1ULL * 1024 * 1024;  // 1 MB

class small_task : public oom_test_task_base {
 public:
  using oom_test_task_base::oom_test_task_base;

  void execute(rmm::cuda_stream_view stream) override
  {
    auto& global = _global_state->cast<oom_test_global_state>();

    auto guard = setup_allocator(stream, "small test task");
    if (!guard) { return; }
    auto* allocator = guard->allocator;

    void* allocation =
      allocator->allocate(stream, kSmallAllocationBytes, alignof(std::max_align_t));
    allocator->deallocate(stream, allocation, kSmallAllocationBytes, alignof(std::max_align_t));
    global.completed_count.fetch_add(1, std::memory_order_relaxed);
  }

  std::unique_ptr<gpu_pipeline_task> create_rescheduled_task(
    uint64_t task_id,
    std::unique_ptr<sirius::pipeline::sirius_pipeline_task_local_state> local_state) override
  {
    auto typed_local = std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state>(
      static_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(local_state.release()));
    return std::make_unique<small_task>(
      task_id,
      std::move(typed_local),
      std::dynamic_pointer_cast<oom_test_global_state>(_global_state));
  }
};

//------------------------------------------------------------------------------
// XL task — always allocates more than the total GPU capacity so it can never
// succeed, guaranteeing it will be rescheduled until the retry limit is hit.
//------------------------------------------------------------------------------
constexpr std::size_t kXlAllocationBytes = 2ULL * 1024 * 1024 * 1024;  // 2 GB (> kGpuCapacity)

class xl_task : public oom_test_task_base {
 public:
  using oom_test_task_base::oom_test_task_base;

  void execute(rmm::cuda_stream_view stream) override
  {
    auto& global = _global_state->cast<oom_test_global_state>();
    auto& local  = _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>();

    auto guard = setup_allocator(stream, "XL test task");
    if (!guard) { return; }

    try {
      guard->allocator->allocate(stream, kXlAllocationBytes, alignof(std::max_align_t));
    } catch (const rmm::out_of_memory&) {
      global.oom_count.fetch_add(1, std::memory_order_relaxed);
      throw sirius::pipeline::oom_reschedule_exception(
        std::move(local._input_data), 0, "OOM in XL test task allocation");
    }

    // Should never reach here — allocation always exceeds capacity.
    global.add_error("XL task allocation unexpectedly succeeded.");
  }

  std::unique_ptr<gpu_pipeline_task> create_rescheduled_task(
    uint64_t task_id,
    std::unique_ptr<sirius::pipeline::sirius_pipeline_task_local_state> local_state) override
  {
    auto typed_local = std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state>(
      static_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(local_state.release()));
    return std::make_unique<xl_task>(
      task_id,
      std::move(typed_local),
      std::dynamic_pointer_cast<oom_test_global_state>(_global_state));
  }
};

//------------------------------------------------------------------------------
// Floor-aware task — models the production retry contract that the fail-fast rule
// depends on: the estimator honours the OOM-derived retry floor (production does
// this in gpu_pipeline_task::get_estimated_reservation_size_info) and the OOM
// handler records live + requested the way gpu_pipeline_task::execute does.
// Allocates kFloorAllocation on every attempt.
//------------------------------------------------------------------------------
constexpr std::size_t kFloorAllocation = 300ULL * 1024 * 1024;  // 300 MB

class floor_aware_task : public oom_test_task_base {
 public:
  using oom_test_task_base::oom_test_task_base;

  sirius::pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space* /*target_space*/) const override
  {
    auto const& ls = _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>();
    sirius::pipeline::reservation_size_info info;
    info.retry_reservation_floor = ls.get_retry_reservation_floor();
    info.reservation_size        = std::max(kReservationSize, info.retry_reservation_floor);
    return info;
  }

  void execute(rmm::cuda_stream_view stream) override
  {
    auto& global = _global_state->cast<oom_test_global_state>();
    auto& local  = _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>();

    auto const reservation_bytes = local.get_reservation_bytes();
    auto guard                   = setup_allocator(stream, "floor-aware test task");
    if (!guard) { return; }
    auto* allocator = guard->allocator;

    void* allocation = nullptr;
    try {
      allocation = allocator->allocate(stream, kFloorAllocation, alignof(std::max_align_t));
    } catch (const rmm::out_of_memory& oom) {
      std::optional<std::size_t> requested;
      if (auto const* cc_oom =
            dynamic_cast<const cucascade::memory::cucascade_out_of_memory*>(&oom)) {
        requested = cc_oom->requested_bytes;
      }
      auto const live = allocator->get_allocated_bytes(stream);
      local.update_retry_reservation_floor_after_oom(reservation_bytes, live, requested);
      global.oom_count.fetch_add(1, std::memory_order_relaxed);
      throw sirius::pipeline::oom_reschedule_exception(
        std::move(local._input_data), 0, "OOM in floor-aware test task allocation");
    }

    allocator->deallocate(stream, allocation, kFloorAllocation, alignof(std::max_align_t));
    global.completed_count.fetch_add(1, std::memory_order_relaxed);
  }

  std::unique_ptr<gpu_pipeline_task> create_rescheduled_task(
    uint64_t task_id,
    std::unique_ptr<sirius::pipeline::sirius_pipeline_task_local_state> local_state) override
  {
    auto typed_local = std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state>(
      static_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(local_state.release()));
    return std::make_unique<floor_aware_task>(
      task_id,
      std::move(typed_local),
      std::dynamic_pointer_cast<oom_test_global_state>(_global_state));
  }
};

//------------------------------------------------------------------------------
// Holder task — reserves most of the space and sleeps without allocating. The
// reservation alone charges the pool and keeps a notifier alive on the space's
// notification channel, so a concurrent make_reservation() blocks instead of
// being handed a partial "upto" grant.
//------------------------------------------------------------------------------
constexpr std::size_t kHolderReservation = 900ULL * 1024 * 1024;  // 900 MB
constexpr auto kHolderDuration           = std::chrono::milliseconds(300);

class holder_task : public oom_test_task_base {
 public:
  using oom_test_task_base::oom_test_task_base;

  sirius::pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space* /*target_space*/) const override
  {
    sirius::pipeline::reservation_size_info info;
    info.reservation_size = kHolderReservation;
    return info;
  }

  void execute(rmm::cuda_stream_view) override
  {
    auto& global = _global_state->cast<oom_test_global_state>();
    auto& local  = _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>();
    if (local.get_reservation_bytes() != kHolderReservation) {
      global.add_error("Holder task did not receive its full reservation.");
    }
    // The reservation stays on the local state until the task is destroyed.
    std::this_thread::sleep_for(kHolderDuration);
    global.completed_count.fetch_add(1, std::memory_order_relaxed);
  }

  std::unique_ptr<gpu_pipeline_task> create_rescheduled_task(
    uint64_t task_id,
    std::unique_ptr<sirius::pipeline::sirius_pipeline_task_local_state> local_state) override
  {
    auto typed_local = std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state>(
      static_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(local_state.release()));
    return std::make_unique<holder_task>(
      task_id,
      std::move(typed_local),
      std::dynamic_pointer_cast<oom_test_global_state>(_global_state));
  }
};

//------------------------------------------------------------------------------
// Big-estimate task — asks for more than the space can grant (the executor clamps
// the request to the space max), then only needs a little. A first attempt on a
// partial grant must run, not fail fast.
//------------------------------------------------------------------------------
constexpr std::size_t kBigEstimateBytes  = 2ULL * 1024 * 1024 * 1024;  // 2 GB (> kGpuCapacity)
constexpr std::size_t kBigTaskAllocation = 50ULL * 1024 * 1024;        // 50 MB

class big_estimate_task : public oom_test_task_base {
 public:
  using oom_test_task_base::oom_test_task_base;

  sirius::pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space* /*target_space*/) const override
  {
    sirius::pipeline::reservation_size_info info;
    info.reservation_size = kBigEstimateBytes;
    return info;
  }

  void execute(rmm::cuda_stream_view stream) override
  {
    auto& global = _global_state->cast<oom_test_global_state>();

    auto guard = setup_allocator(stream, "big-estimate test task");
    if (!guard) { return; }
    auto* allocator = guard->allocator;

    void* allocation = allocator->allocate(stream, kBigTaskAllocation, alignof(std::max_align_t));
    allocator->deallocate(stream, allocation, kBigTaskAllocation, alignof(std::max_align_t));
    global.completed_count.fetch_add(1, std::memory_order_relaxed);
  }

  std::unique_ptr<gpu_pipeline_task> create_rescheduled_task(
    uint64_t task_id,
    std::unique_ptr<sirius::pipeline::sirius_pipeline_task_local_state> local_state) override
  {
    auto typed_local = std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state>(
      static_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(local_state.release()));
    return std::make_unique<big_estimate_task>(
      task_id,
      std::move(typed_local),
      std::dynamic_pointer_cast<oom_test_global_state>(_global_state));
  }
};

std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state> make_empty_local_state()
{
  return std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(
    std::make_unique<sirius::op::pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{}));
}

// Polls `done` every 10 ms until it holds or `timeout` elapses; returns whether it held.
template <typename Pred>
bool wait_until(Pred&& done, std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!done()) {
    if (std::chrono::steady_clock::now() > deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Memory layout for the reschedule test:
//   Allocation    = 400 MB per task (much larger than the 50 MB reservation)
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
// ---------------------------------------------------------------------------
TEST_CASE("GPU pipeline executor reschedules tasks on OOM", "[gpu_pipeline_executor][oom]")
{
  oom_test_fixture f;
  if (!f.setup(3, "oom-test")) {
    WARN("Skipping OOM reschedule test — no GPU available.");
    return;
  }

  auto global_state = std::make_shared<oom_test_global_state>();

  const int num_tasks = 3;
  std::atomic<int> dispatched{0};

  f.executor->start();

  // Post-v1.0 push-model (commit 90dc104): gpu_pipeline_executor no longer publishes
  // task_requests. Schedule tasks directly onto the executor instead of waiting on
  // f.request_channel.get(). The request_channel is kept wired to preserve the
  // fixture API but is not used at runtime.
  std::thread request_handler([&]() {
    while (dispatched.load(std::memory_order_relaxed) < num_tasks) {
      auto local_state = std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(
        std::make_unique<sirius::op::pipelineable_operator_data>(
          std::vector<std::shared_ptr<cucascade::data_batch>>{}));
      auto task = std::make_unique<oom_test_task>(
        static_cast<uint64_t>(dispatched.load(std::memory_order_relaxed)),
        std::move(local_state),
        global_state);
      f.executor->schedule(std::move(task));
      dispatched.fetch_add(1, std::memory_order_relaxed);
    }
  });

  //--------------------------------------------------------------------------
  // Wait for all tasks to complete (including rescheduled ones)
  //--------------------------------------------------------------------------
  auto start_time = std::chrono::steady_clock::now();
  auto timeout    = std::chrono::seconds(30);
  while (global_state->completed_count.load(std::memory_order_relaxed) < num_tasks) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      f.executor->stop();
      f.request_channel.close();
      request_handler.join();
      FAIL("Timed out waiting for OOM rescheduled tasks to complete. "
           << "Completed: " << global_state->completed_count.load()
           << ", OOM count: " << global_state->oom_count.load());
    }
  }

  f.executor->stop();
  f.request_channel.close();
  request_handler.join();

  //--------------------------------------------------------------------------
  // Validate results
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

// ---------------------------------------------------------------------------
// Test: tasks that can never succeed exhaust their retry budget (MAX_OOM_RETRIES=10)
// and cause the query to fail, while small tasks that fit in memory still complete.
//
// Memory layout:
//   GPU capacity  = 1100 MB
//   small_task    =    1 MB allocation  → always succeeds
//   xl_task       = 2048 MB allocation  → always OOMs (exceeds capacity)
//
// We dispatch 5 small tasks + 3 XL tasks = 8 total.
// The XL tasks will be rescheduled up to 10 times each before the executor
// reports a max-retry error on the completion handler.
// The 5 small tasks should all complete regardless.
// After the error, drain_and_wait() should empty the task queue.
// ---------------------------------------------------------------------------
TEST_CASE("GPU pipeline executor fails after max OOM retries",
          "[gpu_pipeline_executor][oom][max_retries]")
{
  oom_test_fixture f;
  if (!f.setup(3, "oom-retry")) {
    WARN("Skipping max-retry OOM test — no GPU available.");
    return;
  }

  auto global_state = std::make_shared<oom_test_global_state>();

  const int num_small = 5;
  const int num_xl    = 3;
  const int num_tasks = num_small + num_xl;
  std::atomic<int> dispatched{0};

  f.executor->start();

  // Post-v1.0 push-model (commit 90dc104): schedule tasks directly instead of
  // waiting on request_channel.get(). First num_small tasks become small_tasks,
  // the rest xl_tasks.
  std::thread request_handler([&]() {
    while (dispatched.load(std::memory_order_relaxed) < num_tasks) {
      auto id          = static_cast<uint64_t>(dispatched.load(std::memory_order_relaxed));
      auto local_state = std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(
        std::make_unique<sirius::op::pipelineable_operator_data>(
          std::vector<std::shared_ptr<cucascade::data_batch>>{}));

      std::unique_ptr<sirius::parallel::itask> task;
      if (id < static_cast<uint64_t>(num_small)) {
        task = std::make_unique<small_task>(id, std::move(local_state), global_state);
      } else {
        task = std::make_unique<xl_task>(id, std::move(local_state), global_state);
      }
      f.executor->schedule(std::move(task));
      dispatched.fetch_add(1, std::memory_order_relaxed);
    }
  });

  //--------------------------------------------------------------------------
  // Wait for the completion handler to report an error (max retries exceeded)
  //--------------------------------------------------------------------------
  auto start_time = std::chrono::steady_clock::now();
  auto timeout    = std::chrono::seconds(60);
  while (!f.completion.has_error()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      f.executor->stop();
      f.request_channel.close();
      request_handler.join();
      FAIL("Timed out waiting for max-retry OOM error. "
           << "Completed: " << global_state->completed_count.load()
           << ", OOM count: " << global_state->oom_count.load());
    }
  }

  //--------------------------------------------------------------------------
  // Drain remaining tasks and shut down
  //--------------------------------------------------------------------------
  f.executor->drain_and_wait();
  f.request_channel.close();
  request_handler.join();
  f.executor->stop();

  //--------------------------------------------------------------------------
  // Validate results
  //--------------------------------------------------------------------------
  if (global_state->error_count.load(std::memory_order_relaxed) > 0) {
    std::lock_guard<std::mutex> lock(global_state->error_mutex);
    for (const auto& error : global_state->errors) {
      INFO(error);
    }
  }

  // All small tasks should have completed successfully.
  REQUIRE(global_state->error_count.load(std::memory_order_relaxed) == 0);
  REQUIRE(global_state->completed_count.load(std::memory_order_relaxed) == num_small);

  // The completion handler should be in an error state from exceeding max retries.
  REQUIRE(f.completion.has_error());

  // XL tasks should have OOM'd many times (at least 10 for the one that hit the limit).
  REQUIRE(global_state->oom_count.load(std::memory_order_relaxed) >= 10);

  // After drain_and_wait(), the task queue should be empty.
  REQUIRE(f.executor->is_task_queue_empty());

  INFO("Max-retry OOM test passed: " << global_state->oom_count.load() << " OOM events, "
                                     << global_state->completed_count.load()
                                     << " small tasks completed, error correctly reported");
}

// ---------------------------------------------------------------------------
// Fail-fast: an OOM retry that provably cannot be granted what its last OOM
// needed fails the query at that retry instead of replaying the OOM 100 times.
//
// Memory layout (capacity 1100 MB, reservation limit 1045 MB, hog 900 MB held
// outside any reservation):
//   attempt 0: gate grants 50 MB in full (900 + 50 <= 1045); the 300 MB allocation
//              overflows by 250 MB -> 1200 > 1100 -> LIMIT_EXCEEDED; required = 300 MB
//   retry 1:   floor 300 MB -> or_null fails -> IDLE -> upto 145 MB (partial); the
//              downgrade sweeps an empty registry and frees 0; the final
//              make_reservation grants 145 MB again; the OOM repeats
//              (145 + 155 overflow -> 1200 > 1100); 300 MB > 145 MB granted, nothing
//              completed, no first attempt in flight -> futile.
// ---------------------------------------------------------------------------
TEST_CASE("GPU pipeline executor fails the query when an OOM retry cannot make progress",
          "[gpu_pipeline_executor][oom][futile]")
{
  oom_test_fixture f;
  if (!f.setup(1, "oom-futile", /*with_downgrade_executor=*/true)) {
    WARN("Skipping futile OOM test — no GPU available.");
    return;
  }

  auto hog = make_hog(f.mem_space);
  REQUIRE(f.mem_space->get_available_memory() <= kGpuCapacity - kHogBytes + (1ULL << 20));

  auto global_state = std::make_shared<oom_test_global_state>();
  auto fut          = f.completion.get_awaitable();

  f.executor->start();
  auto const start_time = std::chrono::steady_clock::now();
  f.executor->schedule(
    std::make_unique<floor_aware_task>(1, make_empty_local_state(), global_state));

  bool const failed = wait_until([&] { return f.completion.has_error(); }, std::chrono::seconds(5));
  auto const elapsed = std::chrono::steady_clock::now() - start_time;

  f.executor->drain_and_wait();
  f.executor->stop();

  if (global_state->error_count.load(std::memory_order_relaxed) > 0) {
    std::lock_guard<std::mutex> lock(global_state->error_mutex);
    for (const auto& error : global_state->errors) {
      INFO(error);
    }
  }
  REQUIRE(global_state->error_count.load(std::memory_order_relaxed) == 0);
  REQUIRE(failed);
  REQUIRE_THROWS_WITH(fut.get(),
                      Catch::Contains("gave up after 1 OOM retry") &&
                        Catch::Contains("held outside any task reservation") &&
                        Catch::Contains("freed 0 bytes"));

  // First attempt + the one retry whose OOM confirmed the gate's verdict.
  REQUIRE(global_state->oom_count.load(std::memory_order_relaxed) == 2);
  REQUIRE(global_state->completed_count.load(std::memory_order_relaxed) == 0);

  auto const metrics = f.executor->get_metrics();
  REQUIRE(metrics.oom_reschedules == 1);
  REQUIRE(metrics.futile_aborts == 1);
  REQUIRE(f.executor->is_task_queue_empty());
  REQUIRE(f.progress->inflight_first_attempts.load() == 0);
  REQUIRE(elapsed < std::chrono::seconds(2));

  hog.reset();  // after the executor is stopped
}

// ---------------------------------------------------------------------------
// Fact A pinned: while another task holds a live reservation the space hands out
// no partial grant, so the retry blocks until the holder finishes and is then
// granted in full. If cucascade ever returned a partial grant while a
// reservation is alive, the floor task would OOM more than once.
// ---------------------------------------------------------------------------
TEST_CASE("GPU pipeline executor keeps retrying an OOM while another task holds a live reservation",
          "[gpu_pipeline_executor][oom][futile]")
{
  oom_test_fixture f;
  if (!f.setup(2, "oom-holder", /*with_downgrade_executor=*/true)) {
    WARN("Skipping holder OOM test — no GPU available.");
    return;
  }

  auto global_state = std::make_shared<oom_test_global_state>();

  f.executor->start();
  f.executor->schedule(std::make_unique<holder_task>(1, make_empty_local_state(), global_state));
  f.executor->schedule(
    std::make_unique<floor_aware_task>(2, make_empty_local_state(), global_state));

  bool const both_done = wait_until(
    [&] {
      return global_state->completed_count.load(std::memory_order_relaxed) >= 2 ||
             f.completion.has_error();
    },
    std::chrono::seconds(10));

  f.executor->drain_and_wait();
  f.executor->stop();

  if (global_state->error_count.load(std::memory_order_relaxed) > 0) {
    std::lock_guard<std::mutex> lock(global_state->error_mutex);
    for (const auto& error : global_state->errors) {
      INFO(error);
    }
  }
  REQUIRE(global_state->error_count.load(std::memory_order_relaxed) == 0);
  REQUIRE(both_done);
  REQUIRE_FALSE(f.completion.has_error());
  REQUIRE(global_state->completed_count.load(std::memory_order_relaxed) == 2);
  REQUIRE(global_state->oom_count.load(std::memory_order_relaxed) == 1);

  auto const metrics = f.executor->get_metrics();
  REQUIRE(metrics.oom_reschedules == 1);
  REQUIRE(metrics.futile_aborts == 0);
  REQUIRE(f.progress->inflight_first_attempts.load() == 0);
}

// ---------------------------------------------------------------------------
// A first attempt on a partial grant runs to completion (rule: retry_count == 0
// never fails fast). No downgrade executor here, covering the gate branch that
// records downgrade_requested == false.
// ---------------------------------------------------------------------------
TEST_CASE("GPU pipeline executor does not fail a first attempt that runs on a partial reservation",
          "[gpu_pipeline_executor][oom][futile]")
{
  oom_test_fixture f;
  if (!f.setup(1, "oom-first", /*with_downgrade_executor=*/false)) {
    WARN("Skipping partial-first-attempt test — no GPU available.");
    return;
  }

  auto hog = make_hog(f.mem_space);
  REQUIRE(f.mem_space->get_available_memory() <= kGpuCapacity - kHogBytes + (1ULL << 20));

  auto global_state = std::make_shared<oom_test_global_state>();

  f.executor->start();
  f.executor->schedule(
    std::make_unique<big_estimate_task>(1, make_empty_local_state(), global_state));

  bool const done = wait_until(
    [&] {
      return global_state->completed_count.load(std::memory_order_relaxed) >= 1 ||
             f.completion.has_error();
    },
    std::chrono::seconds(5));

  f.executor->drain_and_wait();
  f.executor->stop();

  if (global_state->error_count.load(std::memory_order_relaxed) > 0) {
    std::lock_guard<std::mutex> lock(global_state->error_mutex);
    for (const auto& error : global_state->errors) {
      INFO(error);
    }
  }
  REQUIRE(global_state->error_count.load(std::memory_order_relaxed) == 0);
  REQUIRE(done);
  REQUIRE_FALSE(f.completion.has_error());
  REQUIRE(global_state->completed_count.load(std::memory_order_relaxed) == 1);
  REQUIRE(global_state->oom_count.load(std::memory_order_relaxed) == 0);

  auto const metrics = f.executor->get_metrics();
  REQUIRE(metrics.futile_aborts == 0);
  REQUIRE(metrics.oom_reschedules == 0);

  hog.reset();  // after the executor is stopped
}

// ---------------------------------------------------------------------------
// Once the query has failed, queued tasks are dropped at the gate instead of
// each costing the manager a blocking make_reservation() and a dispatch.
// ---------------------------------------------------------------------------
TEST_CASE("GPU pipeline executor drops queued tasks once the query has failed",
          "[gpu_pipeline_executor][oom][futile]")
{
  oom_test_fixture f;
  if (!f.setup(1, "oom-dropped")) {
    WARN("Skipping dropped-task test — no GPU available.");
    return;
  }

  auto global_state = std::make_shared<oom_test_global_state>();
  f.completion.report_error("simulated earlier failure");
  REQUIRE(f.completion.has_error());

  f.executor->start();
  f.executor->schedule(std::make_unique<small_task>(1, make_empty_local_state(), global_state));
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  f.executor->drain_and_wait();
  f.executor->stop();

  REQUIRE(global_state->error_count.load(std::memory_order_relaxed) == 0);
  REQUIRE(global_state->completed_count.load(std::memory_order_relaxed) == 0);
  REQUIRE(f.executor->is_task_queue_empty());
  REQUIRE(f.executor->get_metrics().tasks_executed == 0);
}
