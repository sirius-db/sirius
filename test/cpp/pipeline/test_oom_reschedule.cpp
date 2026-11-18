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
#include "creator/task_creator.hpp"
#include "data/data_batch_utils.hpp"
#include "exec/config.hpp"
#include "op/scan/table_scan/scan_contract.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/query_retirement_test_utils.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_scheduler.hpp"
#include "scan/test_utils.hpp"  // IWYU pragma: keep
#include "utils/telemetry_utils.hpp"
#include "utils/utils.hpp"

#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <limits>
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
  sirius::exec::query_lifecycle_registry lifecycle;
  std::unique_ptr<sirius::pipeline::task_scheduler> executor;
  std::unique_ptr<sirius::creator::task_creator> creator;

  ~oom_test_fixture()
  {
    // GPU workers borrow the creator, so stop them before member destruction.
    if (executor) executor->stop();
  }
  // The query's completion signal now travels on the task's global state rather than on
  // the executor, so it is shared with whatever global state the test builds.
  std::shared_ptr<sirius::pipeline::completion_handler> completion =
    std::make_shared<sirius::pipeline::completion_handler>();

  // Returns false if setup failed (no GPU available) — caller should WARN and return.
  bool setup(int num_threads, const std::string& thread_name_prefix)
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

    sirius::exec::thread_pool_config config;
    config.num_threads        = num_threads;
    config.thread_name_prefix = thread_name_prefix;

    lifecycle.open_query(sirius::make_query_id(0));
    executor = std::make_unique<sirius::pipeline::task_scheduler>(
      lifecycle, config, *manager, sirius::test::make_test_telemetry_context());
    creator = std::make_unique<sirius::creator::task_creator>(
      lifecycle, sirius::creator::task_creator_config{}, *manager);
    creator->set_task_scheduler(*executor);
    executor->set_task_creator(*creator);
    return true;
  }
};

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

  std::unique_ptr<sirius::op::operator_data> compute_task(::cuda::stream_ref) override
  {
    return nullptr;
  }

  void publish_output(sirius::op::operator_data&, ::cuda::stream_ref) override {}

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
    ::cuda::stream_ref stream{cudaStream_t{}};

    allocator_guard(cucascade::memory::reservation_aware_resource_adaptor* a, ::cuda::stream_ref s)
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
  std::unique_ptr<allocator_guard> setup_allocator(::cuda::stream_ref stream,
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

  void execute(::cuda::stream_ref stream) override
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

  void execute(::cuda::stream_ref stream) override
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

  void execute(::cuda::stream_ref stream) override
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
  global_state->set_completion_handler(f.completion);

  const int num_tasks = 3;
  std::atomic<int> dispatched{0};

  f.executor->start();

  // Route initial work and retries through the shared scheduler, as production does.
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
      request_handler.join();
      FAIL("Timed out waiting for OOM rescheduled tasks to complete. "
           << "Completed: " << global_state->completed_count.load()
           << ", OOM count: " << global_state->oom_count.load());
    }
  }

  f.executor->stop();
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
// After the error, per-query retirement should empty the task queue.
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
  global_state->set_completion_handler(f.completion);

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
  while (!f.completion->has_error()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      f.executor->stop();
      request_handler.join();
      FAIL("Timed out waiting for max-retry OOM error. "
           << "Completed: " << global_state->completed_count.load()
           << ", OOM count: " << global_state->oom_count.load());
    }
  }

  //--------------------------------------------------------------------------
  // Drain remaining tasks and shut down
  //--------------------------------------------------------------------------
  sirius::test::retire_scheduler_query(
    f.lifecycle, *f.executor, sirius::make_query_id(0), false, f.creator.get());
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
  REQUIRE(f.completion->has_error());

  // XL tasks should have OOM'd many times (at least 10 for the one that hit the limit).
  REQUIRE(global_state->oom_count.load(std::memory_order_relaxed) >= 10);

  // After retirement, the task queue should be empty.
  REQUIRE(f.executor->get_pipeline_task_queue()->empty());

  INFO("Max-retry OOM test passed: " << global_state->oom_count.load() << " OOM events, "
                                     << global_state->completed_count.load()
                                     << " small tasks completed, error correctly reported");
}

TEST_CASE("GPU retry-handler failures preserve their type and runtime classification",
          "[gpu_pipeline_executor][device_health]")
{
  using namespace std::chrono_literals;
  bool const fatal = GENERATE(false, true);
  oom_test_fixture f;
  if (!f.setup(1, "retry-error")) { SKIP("No GPU available"); }

  class retry_failure_task final : public oom_test_task_base {
   public:
    using oom_test_task_base::oom_test_task_base;
    std::exception_ptr failure;
    void execute(::cuda::stream_ref) override
    {
      throw sirius::pipeline::task_reschedule_exception(nullptr, 0, "retry requested");
    }
    std::unique_ptr<gpu_pipeline_task> create_rescheduled_task(
      uint64_t, std::unique_ptr<sirius::pipeline::sirius_pipeline_task_local_state>) override
    {
      std::rethrow_exception(failure);
    }
  };
  auto global = std::make_shared<oom_test_global_state>();
  global->set_completion_handler(f.completion);
  auto task = std::make_unique<retry_failure_task>(
    0, std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(nullptr), global);
  task->failure =
    fatal
      ? std::make_exception_ptr(cudf::cuda_error("retry factory failed", cudaErrorIllegalAddress))
      : std::make_exception_ptr(std::runtime_error("retry factory input: cudaErrorUnknown"));
  auto result = f.completion->get_awaitable();
  f.executor->start();
  f.executor->schedule(std::move(task));
  REQUIRE(result.wait_for(10s) == std::future_status::ready);
  if (fatal) {
    CHECK_THROWS_AS(result.get(), cudf::cuda_error);
  } else {
    CHECK_THROWS_WITH(result.get(), "retry factory input: cudaErrorUnknown");
  }
  CHECK(f.lifecycle.runtime_failed() == fatal);
  sirius::test::retire_scheduler_query(f.lifecycle, *f.executor, sirius::make_query_id(0));
  f.executor->stop();
}

TEST_CASE("GPU executor reports CCCL and Thrust task failures before completion",
          "[gpu_pipeline_executor][device_health]")
{
  using namespace std::chrono_literals;
  bool const cccl = GENERATE(false, true);
  auto status     = GENERATE(cudaErrorIllegalAddress, cudaErrorInvalidValue);
  oom_test_fixture f;
  if (!f.setup(1, "typed-cuda-error")) { SKIP("No GPU available"); }

  class cuda_failure_task final : public oom_test_task_base {
   public:
    using oom_test_task_base::oom_test_task_base;
    std::exception_ptr failure;
    void execute(::cuda::stream_ref) override { std::rethrow_exception(failure); }
  };
  auto global = std::make_shared<oom_test_global_state>();
  global->set_completion_handler(f.completion);
  auto task = std::make_unique<cuda_failure_task>(
    0, std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(nullptr), global);
  auto failure =
    cccl ? std::make_exception_ptr(::cuda::cuda_error(status, "prepare stream synchronization"))
         : std::make_exception_ptr(thrust::system_error(status, thrust::cuda_category(), "task"));
  task->failure = failure;
  auto result   = f.completion->get_awaitable();
  f.executor->start();
  f.executor->schedule(std::move(task));
  REQUIRE(result.wait_for(10s) == std::future_status::ready);
  try {
    result.get();
    FAIL("task failure must reach completion");
  } catch (std::exception const&) {
    CHECK(std::current_exception() == failure);
  }
  CHECK(f.lifecycle.runtime_failed() == (status == cudaErrorIllegalAddress));
  // This fixture has no completion observer: the worker must record its owner
  // before publishing shared health failure or signaling completion.
  auto diagnostics = f.lifecycle.diagnostics();
  REQUIRE(diagnostics.size() == 1);
  CHECK_FALSE(diagnostics.front().error.empty());
  sirius::test::retire_scheduler_query(f.lifecycle, *f.executor, sirius::make_query_id(0));
  f.executor->stop();
}

// ---------------------------------------------------------------------------
// operator_data::shrink_for_retry() in the reschedule path.
//
// These tasks allocate nothing: each attempt records the retry reservation floor it was
// scheduled with and then follows a script (OOM, CUDA launch failure, or complete), so the
// floor sequence is deterministic.
// ---------------------------------------------------------------------------
namespace {

constexpr std::size_t kMiB = 1024ULL * 1024;

enum class attempt_action { oom, launch_failure, complete };

struct retry_probe {
  std::vector<attempt_action> script;                      ///< One action per attempt
  attempt_action after_script = attempt_action::complete;  ///< Action once the script ends
  std::size_t shrinks_allowed = 0;                         ///< Shrinks before the input's floor
  /// When set, an OOM attempt first read-locks its input here, as the real task flow does.
  const cucascade::memory::memory_space* prepare_space = nullptr;

  std::atomic<int> shrink_calls{0};
  std::atomic<std::size_t> shrinks_done{0};
  std::atomic<std::size_t> window{0};
  std::atomic<int> oom_count{0};
  std::atomic<bool> done{false};
  std::atomic<int> read_locked_ooms{0};       ///< OOMs thrown with every input batch read-locked
  std::atomic<int> shrinks_with_locks{0};     ///< Shrink calls that could lock every input batch
  std::atomic<int> shrinks_without_locks{0};  ///< Shrink calls that found a batch still locked

  std::mutex mutex;
  std::vector<std::size_t> floors;  ///< Retry reservation floor seen by each attempt

  std::vector<std::size_t> observed_floors()
  {
    std::lock_guard<std::mutex> lock(mutex);
    return floors;
  }
};

// Input whose shrink_for_retry() halves a "window" for its first `shrinks_allowed` calls.
class shrinking_operator_data : public sirius::op::pipelineable_operator_data {
 public:
  explicit shrinking_operator_data(std::shared_ptr<retry_probe> probe)
    : pipelineable_operator_data(std::vector<std::shared_ptr<cucascade::data_batch>>{}),
      _probe(std::move(probe))
  {
  }

  bool shrink_for_retry() override
  {
    _probe->shrink_calls.fetch_add(1, std::memory_order_relaxed);
    if (_probe->shrinks_done.load(std::memory_order_relaxed) >= _probe->shrinks_allowed) {
      return false;
    }
    _probe->shrinks_done.fetch_add(1, std::memory_order_relaxed);
    _probe->window.store(_probe->window.load(std::memory_order_relaxed) / 2,
                         std::memory_order_relaxed);
    return true;
  }

 private:
  std::shared_ptr<retry_probe> _probe;
};

// Input whose shrink_for_retry() needs exclusive access to its batches, as a spill or drop would.
class lock_probing_operator_data : public sirius::op::pipelineable_operator_data {
 public:
  lock_probing_operator_data(std::vector<std::shared_ptr<cucascade::data_batch>> batches,
                             std::shared_ptr<retry_probe> probe)
    : pipelineable_operator_data(std::move(batches)), _probe(std::move(probe))
  {
  }

  bool shrink_for_retry() override
  {
    _probe->shrink_calls.fetch_add(1, std::memory_order_relaxed);
    bool locked_all = true;
    for (auto const& batch : get_data_batches()) {
      // Non-blocking, and released at once: a batch that is still locked must not hang the test.
      locked_all = batch->try_to_mutable().has_value() && locked_all;
    }
    (locked_all ? _probe->shrinks_with_locks : _probe->shrinks_without_locks)
      .fetch_add(1, std::memory_order_relaxed);
    return false;
  }

 private:
  std::shared_ptr<retry_probe> _probe;
};

class scripted_retry_task : public oom_test_task_base {
 public:
  scripted_retry_task(uint64_t task_id,
                      std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state> local_state,
                      std::shared_ptr<oom_test_global_state> global_state,
                      std::shared_ptr<retry_probe> probe)
    : oom_test_task_base(task_id, std::move(local_state), std::move(global_state)),
      _probe(std::move(probe))
  {
  }

  void execute(::cuda::stream_ref stream) override
  {
    auto& local       = _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>();
    auto const floor  = local.get_retry_reservation_floor();
    std::size_t index = 0;
    {
      std::lock_guard<std::mutex> lock(_probe->mutex);
      index = _probe->floors.size();
      _probe->floors.push_back(floor);
    }
    auto const action =
      index < _probe->script.size() ? _probe->script[index] : _probe->after_script;
    switch (action) {
      case attempt_action::oom:
        // As gpu_pipeline_task does at its OOM throw sites: raise the floor before the executor
        // sees the exception. The reservation of a retried attempt is at least its floor.
        local.update_retry_reservation_floor_after_oom(
          std::max(kReservationSize, floor), 0, std::nullopt);
        _probe->oom_count.fetch_add(1, std::memory_order_relaxed);
        if (_probe->prepare_space != nullptr) { read_lock_input(local, stream); }
        throw sirius::pipeline::oom_reschedule_exception(
          std::move(local._input_data), 0, "scripted OOM");
      case attempt_action::launch_failure:
        throw sirius::pipeline::cuda_launch_reschedule_exception(
          std::move(local._input_data),
          0,
          static_cast<int>(cudaErrorLaunchFailure),
          "scripted CUDA launch failure");
      case attempt_action::complete: _probe->done.store(true, std::memory_order_relaxed); break;
    }
  }

  // A compute OOM carries an input that prepare_for_processing() has read-locked.
  void read_lock_input(sirius::pipeline::gpu_pipeline_task_local_state& local,
                       ::cuda::stream_ref stream)
  {
    local._input_data->prepare_for_processing(_probe->prepare_space, stream);
    stream.sync();
    auto const* input =
      dynamic_cast<const sirius::op::pipelineable_operator_data*>(local._input_data.get());
    if (input == nullptr || input->get_data_batches().empty()) { return; }
    for (auto const& batch : input->get_data_batches()) {
      if (batch->get_state() != cucascade::batch_state::read_only ||
          batch->try_to_mutable().has_value()) {
        return;
      }
    }
    _probe->read_locked_ooms.fetch_add(1, std::memory_order_relaxed);
  }

  std::unique_ptr<gpu_pipeline_task> create_rescheduled_task(
    uint64_t task_id,
    std::unique_ptr<sirius::pipeline::sirius_pipeline_task_local_state> local_state) override
  {
    auto typed_local = std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state>(
      static_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(local_state.release()));
    return std::make_unique<scripted_retry_task>(
      task_id,
      std::move(typed_local),
      std::dynamic_pointer_cast<oom_test_global_state>(_global_state),
      _probe);
  }

 private:
  std::shared_ptr<retry_probe> _probe;
};

// Schedules one scripted task over `input` and waits until the script completes.
// Returns false on timeout.
bool run_scripted_task(oom_test_fixture& f,
                       const std::shared_ptr<retry_probe>& probe,
                       std::unique_ptr<sirius::op::operator_data> input)
{
  auto global_state = std::make_shared<oom_test_global_state>();
  global_state->set_completion_handler(f.completion);
  f.executor->start();
  f.executor->schedule(std::make_unique<scripted_retry_task>(
    0,
    std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(input)),
    global_state,
    probe));

  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!probe->done.load(std::memory_order_relaxed) && !f.completion->has_error() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  f.executor->stop();
  return probe->done.load(std::memory_order_relaxed);
}

}  // namespace

TEST_CASE("GPU pipeline executor keeps the pre-OOM floor while the input shrinks for retry",
          "[gpu_pipeline_executor][oom][shrink_for_retry]")
{
  oom_test_fixture f;
  if (!f.setup(1, "oom-shrink")) { SKIP("No GPU available"); }

  auto probe    = std::make_shared<retry_probe>();
  probe->script = {
    attempt_action::oom, attempt_action::oom, attempt_action::oom, attempt_action::oom};
  probe->shrinks_allowed = 2;
  probe->window          = 8;

  REQUIRE(run_scripted_task(f, probe, std::make_unique<shrinking_operator_data>(probe)));
  REQUIRE_FALSE(f.completion->has_error());

  // Attempts 0 and 1 OOM while the input can still shrink: the retry keeps the floor from before
  // the OOM (0). Attempts 2 and 3 OOM with the input at its own floor, so the floor doubles.
  CHECK(probe->observed_floors() ==
        std::vector<std::size_t>{0, 0, 0, 2 * kReservationSize, 4 * kReservationSize});
  CHECK(probe->oom_count.load() == 4);
  CHECK(probe->shrink_calls.load() == 4);
  CHECK(probe->shrinks_done.load() == 2);
  CHECK(probe->window.load() == 2);
}

TEST_CASE("GPU pipeline executor raises the retry floor for inputs that do not shrink",
          "[gpu_pipeline_executor][oom][shrink_for_retry]")
{
  oom_test_fixture f;
  if (!f.setup(1, "oom-noshrink")) { SKIP("No GPU available"); }

  auto probe    = std::make_shared<retry_probe>();
  probe->script = {attempt_action::oom, attempt_action::oom};

  REQUIRE(run_scripted_task(f,
                            probe,
                            std::make_unique<sirius::op::pipelineable_operator_data>(
                              std::vector<std::shared_ptr<cucascade::data_batch>>{})));
  REQUIRE_FALSE(f.completion->has_error());

  // max(old floor, 2 x reservation, live + requested): 2 x 50 MB, then 2 x 100 MB.
  CHECK(probe->observed_floors() ==
        std::vector<std::size_t>{0, 2 * kReservationSize, 4 * kReservationSize});
}

TEST_CASE("GPU pipeline executor releases the input's read-only locks before shrink_for_retry",
          "[gpu_pipeline_executor][oom][shrink_for_retry]")
{
  oom_test_fixture f;
  if (!f.setup(1, "oom-shrink-lock")) { SKIP("No GPU available"); }

  rmm::cuda_stream stream;
  auto table = sirius::create_cudf_table_with_random_data(1024,
                                                          {cudf::data_type{cudf::type_id::INT64}},
                                                          {std::make_pair(0, 1000000)},
                                                          stream,
                                                          f.mem_space->get_default_allocator());
  stream.synchronize();
  auto batch = sirius::make_data_batch(
    std::move(table), *f.mem_space, stream, sirius::telemetry::batch_telemetry_info{});

  auto probe           = std::make_shared<retry_probe>();
  probe->script        = {attempt_action::oom, attempt_action::oom};
  probe->prepare_space = f.mem_space;

  REQUIRE(run_scripted_task(f,
                            probe,
                            std::make_unique<lock_probing_operator_data>(
                              std::vector<std::shared_ptr<cucascade::data_batch>>{batch}, probe)));
  REQUIRE_FALSE(f.completion->has_error());

  // Both OOMs were thrown with the batch read-locked by the input (no exclusive lock possible).
  REQUIRE(probe->oom_count.load() == 2);
  REQUIRE(probe->read_locked_ooms.load() == 2);
  // By the time the input is asked to shrink, it can lock the batch exclusively.
  CHECK(probe->shrink_calls.load() == 2);
  CHECK(probe->shrinks_with_locks.load() == 2);
  CHECK(probe->shrinks_without_locks.load() == 0);
  CHECK(batch->get_state() == cucascade::batch_state::idle);
}

TEST_CASE("GPU pipeline executor does not shrink the input on a non-OOM reschedule",
          "[gpu_pipeline_executor][oom][shrink_for_retry]")
{
  oom_test_fixture f;
  if (!f.setup(1, "launch-noshrink")) { SKIP("No GPU available"); }

  auto probe             = std::make_shared<retry_probe>();
  probe->script          = {attempt_action::launch_failure, attempt_action::launch_failure};
  probe->shrinks_allowed = std::numeric_limits<std::size_t>::max();
  probe->window          = 8;

  REQUIRE(run_scripted_task(f, probe, std::make_unique<shrinking_operator_data>(probe)));
  REQUIRE_FALSE(f.completion->has_error());

  CHECK(probe->observed_floors() == std::vector<std::size_t>{0, 0, 0});
  CHECK(probe->shrink_calls.load() == 0);
  CHECK(probe->window.load() == 8);
}

TEST_CASE("GPU pipeline executor retry cap bounds an input that always shrinks",
          "[gpu_pipeline_executor][oom][shrink_for_retry][max_retries]")
{
  oom_test_fixture f;
  if (!f.setup(1, "oom-shrink-cap")) { SKIP("No GPU available"); }

  constexpr int kRetryLimit             = 3;
  auto injections                       = std::make_shared<sirius::op::scan::test_injections>();
  injections->gpu_task_retry_limit      = kRetryLimit;
  injections->gpu_task_retry_backoff_ms = 0;
  f.completion->injections              = std::move(injections);

  auto probe             = std::make_shared<retry_probe>();
  probe->after_script    = attempt_action::oom;
  probe->shrinks_allowed = std::numeric_limits<std::size_t>::max();
  probe->window          = std::size_t{1} << 20;

  REQUIRE_FALSE(run_scripted_task(f, probe, std::make_unique<shrinking_operator_data>(probe)));
  REQUIRE(f.completion->has_error());
  try {
    f.completion->get_awaitable().get();
    FAIL("the retry limit must fail the query");
  } catch (std::exception const& e) {
    CHECK(std::string(e.what()).find("exceeded maximum retry limit") != std::string::npos);
  }
  sirius::test::retire_scheduler_query(
    f.lifecycle, *f.executor, sirius::make_query_id(0), false, f.creator.get());

  // The first attempt plus kRetryLimit retries all OOM; the input shrinks before each retry and
  // is not asked again once the cap fails the query. The floor never rises.
  CHECK(probe->oom_count.load() == kRetryLimit + 1);
  CHECK(probe->shrink_calls.load() == kRetryLimit);
  CHECK(probe->observed_floors() == std::vector<std::size_t>(kRetryLimit + 1, 0));
  CHECK(probe->window.load() == (std::size_t{1} << 20) >> kRetryLimit);
}

TEST_CASE("Retry reservation floor remembers its value from before the last OOM",
          "[gpu_pipeline_executor][oom][shrink_for_retry]")
{
  using local_state_t = sirius::pipeline::gpu_pipeline_task_local_state;

  // Fresh state: nothing to restore.
  local_state_t fresh(nullptr);
  local_state_t from_fresh(nullptr);
  from_fresh.inherit_retry_reservation_floor_before_last_oom(fresh);
  CHECK(from_fresh.get_retry_reservation_floor() == 0);

  // Attempt 0 OOMs and its input does not shrink: the retry inherits the raised floor.
  local_state_t attempt0(nullptr);
  // max(2 x 256, 64 live + 1 MiB default request, 1 MiB) = 1 MiB + 64.
  attempt0.update_retry_reservation_floor_after_oom(256, 64, std::nullopt);
  auto const first_floor = attempt0.get_retry_reservation_floor();
  REQUIRE(first_floor == kMiB + 64);
  local_state_t attempt1(nullptr);
  attempt1.inherit_retry_reservation_floor(attempt0);
  CHECK(attempt1.get_retry_reservation_floor() == first_floor);

  // Attempt 1 OOMs and its input shrinks: the retry gets the floor from before that raise.
  // max(2 x first_floor, 2 MiB live + 3 MiB requested) = 5 MiB.
  attempt1.update_retry_reservation_floor_after_oom(
    first_floor, 2 * kMiB, std::optional<std::size_t>{3 * kMiB});
  REQUIRE(attempt1.get_retry_reservation_floor() == 5 * kMiB);
  local_state_t attempt2(nullptr);
  attempt2.inherit_retry_reservation_floor_before_last_oom(attempt1);
  CHECK(attempt2.get_retry_reservation_floor() == first_floor);

  // Chained: attempt 2 OOMs and the input shrinks again — still the same pre-raise floor.
  attempt2.update_retry_reservation_floor_after_oom(first_floor, 0, std::nullopt);
  REQUIRE(attempt2.get_retry_reservation_floor() == 2 * first_floor);
  local_state_t attempt3(nullptr);
  attempt3.inherit_retry_reservation_floor_before_last_oom(attempt2);
  CHECK(attempt3.get_retry_reservation_floor() == first_floor);

  // Attempt 3 OOMs with the input at its floor: escalation resumes from the kept floor.
  attempt3.update_retry_reservation_floor_after_oom(first_floor, 0, std::nullopt);
  local_state_t attempt4(nullptr);
  attempt4.inherit_retry_reservation_floor(attempt3);
  CHECK(attempt4.get_retry_reservation_floor() == 2 * first_floor);

  // A state that inherited a floor and never OOMed has nothing lower to hand back.
  local_state_t attempt5(nullptr);
  attempt5.inherit_retry_reservation_floor_before_last_oom(attempt4);
  CHECK(attempt5.get_retry_reservation_floor() == 2 * first_floor);
}
