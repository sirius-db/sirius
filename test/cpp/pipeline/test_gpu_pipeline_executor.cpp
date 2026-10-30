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
#include "event/query_event_publisher.hpp"
#include "exec/channel.hpp"
#include "exec/config.hpp"
#include "log/logging.hpp"
#include "memory/memory_wait_board.hpp"
#include "pipeline/executor_event_recorder.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_request.hpp"
#include "scan/test_utils.hpp"
#include "utils/telemetry_utils.hpp"

#include <rmm/cuda_stream.hpp>

#include <absl/cleanup/cleanup.h>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
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

  /// Size of the reservation the last task executed with.
  std::atomic<std::size_t> granted_bytes{0};
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

// Only the error diagnostic throws; normal manager startup/shutdown remains usable.
class throwing_error_sink : public sirius::log::sink {
 public:
  explicit throwing_error_sink(
    std::shared_ptr<sirius::pipeline::completion_handler> completion = {})
    : completion_(std::move(completion))
  {
  }
  void set_level(sirius::log::level) override {}
  bool should_log(sirius::log::level level) const override
  {
    return level == sirius::log::level::error;
  }
  void log(sirius::log::level, const std::source_location&, std::string_view) override
  {
    ++attempts;
    if (completion_ && !completion_->has_error()) { reported_first = false; }
    throw std::runtime_error("diagnostic failure");
  }
  bool flush() override { return true; }

  std::atomic<int> attempts{0};
  std::atomic<bool> reported_first{true};

 private:
  std::shared_ptr<sirius::pipeline::completion_handler> completion_;
};

class failing_gpu_task : public sirius_pipeline_task {
 public:
  failing_gpu_task(std::shared_ptr<test_gpu_pipeline_task_global_state> global,
                   bool preparation_failure,
                   bool nonstandard_exception)
    : sirius_pipeline_task(
        1, std::make_unique<test_gpu_pipeline_task_local_state>(nullptr), std::move(global)),
      preparation_failure_(preparation_failure),
      nonstandard_exception_(nonstandard_exception)
  {
  }
  sirius::pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space* space) const override
  {
    if (preparation_failure_) { fail(); }
    return sirius_pipeline_task::get_estimated_reservation_size_info(space);
  }
  void execute(::cuda::stream_ref) override { fail(); }

 private:
  [[noreturn]] void fail() const
  {
    if (nonstandard_exception_) { throw 42; }
    throw std::runtime_error("original task failure");
  }
  bool preparation_failure_;
  bool nonstandard_exception_;
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

// Asks for a fixed reservation and records the size it was granted; allocates nothing.
class sized_reservation_task : public sirius_pipeline_task {
 public:
  sized_reservation_task(uint64_t task_id,
                         std::shared_ptr<test_gpu_pipeline_task_global_state> global_state,
                         std::size_t reservation_bytes)
    : sirius_pipeline_task(task_id,
                           std::make_unique<test_gpu_pipeline_task_local_state>(
                             std::make_unique<sirius::op::pipelineable_operator_data>(
                               std::vector<std::shared_ptr<cucascade::data_batch>>{})),
                           std::move(global_state)),
      reservation_bytes_(reservation_bytes)
  {
  }

  void execute(::cuda::stream_ref) override
  {
    auto& global     = _global_state->cast<test_gpu_pipeline_task_global_state>();
    auto& local      = _local_state->cast<test_gpu_pipeline_task_local_state>();
    auto reservation = local.release_reservation();
    global.granted_bytes.store(reservation ? reservation->size() : 0);
    global.executed_count.fetch_add(1, std::memory_order_relaxed);
  }

  sirius::pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space*) const override
  {
    sirius::pipeline::reservation_size_info info;
    info.reservation_size = reservation_bytes_;
    return info;
  }

 private:
  std::size_t reservation_bytes_;
};

template <typename Pred>
bool wait_until(Pred&& done, std::chrono::milliseconds limit = std::chrono::seconds(10))
{
  auto const deadline = std::chrono::steady_clock::now() + limit;
  while (!done()) {
    if (std::chrono::steady_clock::now() > deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

// GPU memory counted against the space but held by no reservation, so it holds no notifier:
// a blocking reservation that cannot be met then ends in a partial grant at once.
struct untracked_allocation {
  cucascade::memory::reservation_aware_resource_adaptor* mr{nullptr};
  rmm::cuda_stream stream;
  void* ptr{nullptr};
  std::size_t bytes{0};

  void allocate(cucascade::memory::memory_space& space, std::size_t size)
  {
    mr = space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    REQUIRE(mr != nullptr);
    ptr   = mr->allocate(::cuda::stream_ref{stream.value()}, size, alignof(std::max_align_t));
    bytes = size;
  }

  ~untracked_allocation()
  {
    if (ptr != nullptr) {
      mr->deallocate(::cuda::stream_ref{stream.value()}, ptr, bytes, alignof(std::max_align_t));
      stream.synchronize();
    }
  }
};

// One GPU executor with a 256 MiB space, a recorder on its events and, optionally, a wait board
// and a downgrade executor that is left unstarted until the test starts it.
struct stall_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager = [] {
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(256ULL * 1024 * 1024)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(1ULL * 1024 * 1024 * 1024)
      .use_gpu_id_as_host_id()
      .track_reservation_per_stream(false)
      .set_reservation_fraction_per_numa_region(0.75);
    return std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
  }();
  cucascade::memory::memory_space* space =
    manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  untracked_allocation pressure;
  sirius::exec::query_lifecycle_registry lifecycle;
  std::shared_ptr<sirius::event::query_event_publisher> publisher =
    std::make_shared<sirius::event::query_event_publisher>();
  sirius::test::executor_event_recorder recorder{*publisher};
  std::shared_ptr<sirius::memory::memory_wait_board> board;
  sirius::data::data_repository_manager_registry repositories;
  sirius::parallel::downgrade_executor downgrade{
    lifecycle,
    sirius::exec::downgrade_executor_config{
      .thread_pool    = {.num_threads = 1, .thread_name_prefix = "stall-downgrade"},
      .monitor_period = std::chrono::milliseconds{0}},
    repositories,
    cucascade::memory::memory_space_id{cucascade::memory::Tier::GPU, 0},
    nullptr,
    *manager};
  std::shared_ptr<sirius::pipeline::completion_handler> completion =
    std::make_shared<sirius::pipeline::completion_handler>();
  std::shared_ptr<test_gpu_pipeline_task_global_state> global =
    std::make_shared<test_gpu_pipeline_task_global_state>();
  sirius::exec::channel<std::unique_ptr<sirius::pipeline::task_request>> requests;
  std::optional<sirius::pipeline::gpu_pipeline_executor> executor;

  stall_fixture()
  {
    REQUIRE(space != nullptr);
    lifecycle.open_query(sirius::make_query_id(0));
    global->set_completion_handler(completion);
  }

  ~stall_fixture()
  {
    if (executor) { executor->stop(); }
  }

  void start(bool use_board, bool with_downgrade, int threads = 1)
  {
    executor.emplace(lifecycle,
                     sirius::exec::thread_pool_config{threads, "stall-test"},
                     space,
                     requests.make_publisher(),
                     with_downgrade ? &downgrade : nullptr,
                     sirius::test::make_test_telemetry_context());
    executor->set_query_event_publisher(*publisher);
    if (use_board) {
      board = std::make_shared<sirius::memory::memory_wait_board>(0);
      executor->set_memory_wait_board(board);
    }
    executor->start();
  }

  [[nodiscard]] int waiting() const { return board ? board->waiting(0) : 0; }
};

using recorder_log = sirius::test::executor_event_recorder::log;

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
  executor.discard_query_tasks(cancelled_query);
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

TEST_CASE("GPU errors precede terminal task destruction even when logging throws",
          "[gpu_pipeline_executor][query_lifecycle_gate]")
{
  using namespace std::chrono_literals;
  bool preparation_failure   = false;
  bool nonstandard_exception = false;
  SECTION("preparation error") { preparation_failure = true; }
  SECTION("unknown preparation error")
  {
    preparation_failure   = true;
    nonstandard_exception = true;
  }
  SECTION("worker error") {}
  SECTION("unknown worker error") { nonstandard_exception = true; }

  auto completion    = std::make_shared<sirius::pipeline::completion_handler>();
  auto done          = completion->get_awaitable();
  auto previous_sink = sirius::log::get_sink();
  auto sink          = std::make_shared<throwing_error_sink>(completion);
  sirius::log::set_sink(sink);
  absl::Cleanup restore_sink = [&] { sirius::log::set_sink(previous_sink); };

  const auto failed_query = sirius::make_query_id(7);
  const auto next_query   = sirius::make_query_id(0);
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(failed_query);
  lifecycle.open_query(next_query);
  auto manager    = initialize_memory_manager(1);
  auto* mem_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(mem_space);

  // A real exhausted terminal pipeline: destroying its sole task attempts success publication.
  sirius::op::sirius_physical_operator terminal(
    sirius::op::SiriusPhysicalOperatorType::RESULT_COLLECTOR, {}, 0);
  auto pipe = std::make_shared<sirius::pipeline::sirius_pipeline>(
    sirius::pipeline::pipeline_build_context{sirius::test::make_test_telemetry_context()});
  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*pipe, terminal);
  build_state.set_pipeline_sink(*pipe, &terminal, 1);
  pipe->set_query_id(failed_query);
  pipe->set_completion_handler(completion);
  auto global = std::make_shared<test_gpu_pipeline_task_global_state>();
  global->set_pipeline(pipe);
  global->set_completion_handler(completion);

  sirius::exec::channel<std::unique_ptr<sirius::pipeline::task_request>> requests;
  sirius::pipeline::gpu_pipeline_executor executor(lifecycle,
                                                   {1, "error-logging"},
                                                   mem_space,
                                                   requests.make_publisher(),
                                                   nullptr,
                                                   sirius::test::make_test_telemetry_context());
  executor.start();
  REQUIRE(executor.schedule(
    std::make_unique<failing_gpu_task>(global, preparation_failure, nonstandard_exception)));
  REQUIRE(done.wait_for(5s) == std::future_status::ready);
  if (nonstandard_exception) {
    try {
      done.get();
      FAIL("expected original nonstandard task exception");
    } catch (int error) {
      CHECK(error == 42);
    }
  } else {
    CHECK_THROWS_WITH(done.get(), "original task failure");
  }
  lifecycle.quiesce_and_wait_for_submissions(failed_query);
  lifecycle.wait_for_work(failed_query);
  CHECK(completion->has_error());
  CHECK(pipe->is_pipeline_finished());
  CHECK(lifecycle.activity(failed_query).work == 0);
#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_ERROR
  CHECK(sink->attempts == 1);
  CHECK(sink->reported_first);
#endif
  lifecycle.close(failed_query);

  // An unrelated query can use the same manager and worker after the error.
  auto next_state = std::make_shared<test_gpu_pipeline_task_global_state>();
  REQUIRE(executor.schedule(std::make_unique<sirius_pipeline_task>(
    2, std::make_unique<test_gpu_pipeline_task_local_state>(nullptr), next_state)));
  auto deadline = std::chrono::steady_clock::now() + 5s;
  while (next_state->executed_count.load() == 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  executor.stop();
  CHECK(next_state->executed_count.load() == 1);
  CHECK(next_state->error_count.load() == 0);
  lifecycle.quiesce_and_wait_for_submissions(next_query);
  CHECK(lifecycle.activity(next_query).work == 0);
  lifecycle.close(next_query);
}

TEST_CASE("bounded pool survives a throwing error sink and releases its slot",
          "[bounded_thread_pool]")
{
  bool nonstandard_exception = false;
  SECTION("standard exception") {}
  SECTION("unknown exception") { nonstandard_exception = true; }
  auto previous_sink = sirius::log::get_sink();
  auto sink          = std::make_shared<throwing_error_sink>();
  sirius::log::set_sink(sink);
  absl::Cleanup restore_sink = [&] { sirius::log::set_sink(previous_sink); };
  sirius::exec::bounded_thread_pool pool(1, "throwing-log");
  pool.dispatch(pool.reserve(), [=] {
    if (nonstandard_exception) { throw 42; }
    throw std::runtime_error("worker failure");
  });
  pool.wait_all();
#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_ERROR
  CHECK(sink->attempts == 1);
#endif
  std::atomic<bool> ran{false};
  pool.dispatch(pool.reserve(), [&] { ran = true; });
  pool.wait_all();
  CHECK(ran);
}

TEST_CASE("GPU executor publishes one task_completed per successful task",
          "[gpu_pipeline_executor][memory_wait_board]")
{
  bool const use_board = GENERATE(true, false);
  stall_fixture f;

  // One task carries a pipeline, so its event names the real query and the source operator.
  auto const piped_query = sirius::make_query_id(7);
  f.lifecycle.open_query(piped_query);
  sirius::op::sirius_physical_operator source(
    sirius::op::SiriusPhysicalOperatorType::FILTER, {}, 0);
  source.operator_id = 5;
  auto pipe          = std::make_shared<sirius::pipeline::sirius_pipeline>(
    sirius::pipeline::pipeline_build_context{sirius::test::make_test_telemetry_context()});
  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*pipe, source);
  build_state.set_pipeline_sink(*pipe, &source, 1);
  pipe->set_query_id(piped_query);
  auto piped_state = std::make_shared<test_gpu_pipeline_task_global_state>();
  piped_state->set_pipeline(pipe);

  f.start(use_board, /*with_downgrade=*/false, /*threads=*/2);
  constexpr int plain_tasks = 4;
  for (int i = 0; i < plain_tasks; ++i) {
    REQUIRE(f.executor->schedule(
      std::make_unique<sized_reservation_task>(i, f.global, 8ULL * 1024 * 1024)));
  }
  REQUIRE(f.executor->schedule(
    std::make_unique<sized_reservation_task>(plain_tasks, piped_state, 8ULL * 1024 * 1024)));

  REQUIRE(f.recorder.wait_for(
    [](recorder_log const& l) { return l.completions.size() == plain_tasks + 1; }));
  f.executor->stop();
  auto const log = f.recorder.snapshot();

  CHECK(log.completions.size() == plain_tasks + 1);
  CHECK(log.count(sirius::event::task_outcome::succeeded) == plain_tasks + 1);
  int piped = 0;
  for (auto const& c : log.completions) {
    CHECK(c.gpu == 0);
    if (c.query == piped_query) {
      ++piped;
      CHECK(c.operator_id == 5);
    } else {
      CHECK(c.query == sirius::make_query_id(0));
      CHECK(c.operator_id == sirius::op::sirius_physical_operator::invalid_operator_id);
    }
  }
  CHECK(piped == 1);
  CHECK(f.global->executed_count.load() == plain_tasks);
  CHECK(piped_state->executed_count.load() == 1);
  // Nothing stalled, so no reserve phase was announced or ended.
  CHECK(log.waits.empty());
  CHECK(log.downgrades.empty());
  CHECK(log.ends.empty());
  CHECK(f.waiting() == 0);
  if (use_board) { CHECK(f.board->oom_pending(0) == 0); }
}

TEST_CASE("A stalled reserve phase stays on the wait board from the partial grant into downgrade",
          "[gpu_pipeline_executor][memory_wait_board]")
{
  bool const use_board = GENERATE(true, false);
  stall_fixture f;
  auto const limit = f.space->get_max_memory();
  auto const need  = limit / 2;
  // About a quarter of the limit stays reservable, so the task's blocking reservation ends in a
  // partial grant (no reservation is alive to wait for) and the executor turns to downgrade.
  f.pressure.allocate(*f.space, limit - limit / 4);
  f.start(use_board, /*with_downgrade=*/true);
  REQUIRE(f.executor->schedule(std::make_unique<sized_reservation_task>(1, f.global, need)));

  // The downgrade executor is not running, so its request stays queued and the task stays parked
  // inside its reserve phase, past both stall announcements.
  REQUIRE(f.recorder.wait_for([](recorder_log const& l) { return !l.downgrades.empty(); }));
  auto const parked = f.recorder.snapshot();
  REQUIRE(parked.waits.size() == 1);
  REQUIRE(parked.downgrades.size() == 1);
  CHECK(parked.waits[0].bytes == need);
  CHECK(parked.downgrades[0].bytes < need);
  CHECK(parked.ends.empty());
  CHECK(parked.completions.empty());
  if (use_board) { CHECK(f.waiting() == 1); }

  SECTION("the downgrade returns and the task runs on its partial grant")
  {
    f.downgrade.start();
    REQUIRE(f.recorder.wait_for(
      [](recorder_log const& l) { return !l.ends.empty() && !l.completions.empty(); }));
    auto const done = f.recorder.snapshot();
    REQUIRE(done.ends.size() == 1);
    CHECK(done.ends[0].query == done.waits[0].query);
    CHECK(done.ends[0].operator_id == done.waits[0].operator_id);
    CHECK(done.ends[0].gpu == 0);
    CHECK(done.ends[0].bytes > 0);
    CHECK(done.ends[0].bytes < need);
    CHECK(f.global->granted_bytes.load() == done.ends[0].bytes);
    REQUIRE(done.completions.size() == 1);
    CHECK(done.completions[0].outcome == sirius::event::task_outcome::succeeded);
    CHECK(f.global->executed_count.load() == 1);
    CHECK_FALSE(f.completion->has_error());
  }

  SECTION("the space shuts down during the downgrade and the task gives up")
  {
    f.space->shutdown();
    f.downgrade.start();
    REQUIRE(f.recorder.wait_for([](recorder_log const& l) { return !l.ends.empty(); }));
    REQUIRE(wait_until([&] { return f.completion->has_error(); }));
    auto const done = f.recorder.snapshot();
    REQUIRE(done.ends.size() == 1);
    CHECK(done.ends[0].bytes == 0);
    CHECK(done.completions.empty());
    CHECK(f.global->executed_count.load() == 0);
  }

  f.executor->stop();
  auto const final_log = f.recorder.snapshot();
  CHECK(final_log.waits.size() == 1);
  CHECK(final_log.downgrades.size() == 1);
  CHECK(final_log.ends.size() == 1);
  CHECK(f.waiting() == 0);
}

TEST_CASE("A cancelled downgrade ends the stalled reserve phase",
          "[gpu_pipeline_executor][memory_wait_board]")
{
  bool const use_board = GENERATE(true, false);
  stall_fixture f;
  auto const limit = f.space->get_max_memory();
  f.pressure.allocate(*f.space, limit - limit / 4);
  // A stopped downgrade executor refuses the request, which the task sees as a cancellation.
  f.downgrade.start();
  f.downgrade.stop();
  f.start(use_board, /*with_downgrade=*/true);
  REQUIRE(f.executor->schedule(std::make_unique<sized_reservation_task>(1, f.global, limit / 2)));

  REQUIRE(f.recorder.wait_for([](recorder_log const& l) { return !l.ends.empty(); }));
  REQUIRE(wait_until([&] { return f.completion->has_error(); }));
  f.executor->stop();
  auto const log = f.recorder.snapshot();
  CHECK(log.waits.size() == 1);
  CHECK(log.downgrades.size() == 1);
  REQUIRE(log.ends.size() == 1);
  CHECK(log.ends[0].bytes == 0);
  CHECK(log.ends[0].query == log.waits[0].query);
  CHECK(log.completions.empty());
  CHECK(f.global->executed_count.load() == 0);
  CHECK(f.waiting() == 0);
}

TEST_CASE("A blocking reserve wait ends exactly once, with or without a grant",
          "[gpu_pipeline_executor][memory_wait_board]")
{
  bool const use_board = GENERATE(true, false);
  stall_fixture f;
  auto const limit = f.space->get_max_memory();
  auto const need  = limit / 2;
  // A live reservation makes the task's reservation wait rather than settle for a partial grant.
  auto holder = f.space->make_reservation_or_null(limit - limit / 4);
  REQUIRE(holder);
  f.start(use_board, /*with_downgrade=*/false);
  REQUIRE(f.executor->schedule(std::make_unique<sized_reservation_task>(1, f.global, need)));

  REQUIRE(f.recorder.wait_for([](recorder_log const& l) { return !l.waits.empty(); }));
  CHECK(f.recorder.snapshot().ends.empty());
  if (use_board) { CHECK(f.waiting() == 1); }

  SECTION("the holder releases its reservation and the task gets all it asked for")
  {
    holder.reset();
    REQUIRE(f.recorder.wait_for(
      [](recorder_log const& l) { return !l.ends.empty() && !l.completions.empty(); }));
    auto const log = f.recorder.snapshot();
    REQUIRE(log.ends.size() == 1);
    CHECK(log.ends[0].bytes == need);
    CHECK(f.global->granted_bytes.load() == need);
    CHECK(log.downgrades.empty());
    REQUIRE(log.completions.size() == 1);
    CHECK(log.completions[0].outcome == sirius::event::task_outcome::succeeded);
  }

  SECTION("the space shuts down and the task gives up")
  {
    f.space->shutdown();
    REQUIRE(f.recorder.wait_for([](recorder_log const& l) { return !l.ends.empty(); }));
    REQUIRE(wait_until([&] { return f.completion->has_error(); }));
    auto const log = f.recorder.snapshot();
    REQUIRE(log.ends.size() == 1);
    CHECK(log.ends[0].bytes == 0);
    CHECK(log.completions.empty());
    CHECK(f.global->executed_count.load() == 0);
  }

  f.executor->stop();
  CHECK(f.recorder.snapshot().ends.size() == 1);
  CHECK(f.waiting() == 0);
}
