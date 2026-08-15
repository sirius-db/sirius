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
 * @file test_task_executor_error_bracket.cpp
 * @brief The error-path quiesce bracket waits per-query, not for the whole pool.
 *
 * itask_executor::wait_and_drain_query joins the manager (closing the pop-to-attach window) and
 * then waits for in-flight work before dropping the failing query's queued tasks. A whole-pool
 * wait at that point means the ERROR path of one query waits for EVERY query's in-flight tasks —
 * including a co-tenant's parked memory wait, which lasts until memory frees somewhere. These
 * cases pin the per-query discipline: the bracket returns while a co-tenant's reservation wait is
 * still parked, the failing query's running task completes BEFORE the bracket returns (the
 * plan-safety invariant), its queued task is dropped, and a task whose query is unknowable (no
 * pipeline — an untagged slot) is still waited for conservatively.
 */

#include "catch.hpp"
#include "exec/channel.hpp"
#include "exec/config.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_request.hpp"
#include "query_id.hpp"
#include "utils/telemetry_utils.hpp"

#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

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

//! Execution log shared by every task in a test, independent of the tasks' global states.
struct bracket_recorder {
  void record(uint64_t task_id)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _executed.push_back(task_id);
  }

  [[nodiscard]] bool has_executed(uint64_t task_id)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto id : _executed) {
      if (id == task_id) { return true; }
    }
    return false;
  }

 private:
  std::mutex _mutex;
  std::vector<uint64_t> _executed;
};

//! Per-pipeline global state carrying the shared recorder.
class bracket_global_state : public sirius::pipeline::sirius_pipeline_task_global_state {
 public:
  bracket_global_state(duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> pipeline,
                       std::shared_ptr<bracket_recorder> recorder)
    : sirius_pipeline_task_global_state(std::move(pipeline),
                                        sirius::test::make_test_telemetry_context()),
      recorder(std::move(recorder))
  {
  }

  std::shared_ptr<bracket_recorder> recorder;
};

//! A task with a controllable reservation demand and execution time. execute() optionally
//! sleeps (to be provably in flight when the bracket starts), drops the reservation (waking any
//! parked waiter), and records its completion.
class bracket_task : public sirius::pipeline::gpu_pipeline_task {
 public:
  bracket_task(uint64_t task_id,
               std::size_t reservation_bytes,
               std::chrono::milliseconds execute_for,
               std::shared_ptr<bracket_global_state> global_state)
    : gpu_pipeline_task(task_id,
                        std::vector<std::shared_ptr<cucascade::shared_data_repository>>{},
                        std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(
                          std::make_unique<sirius::op::pipelineable_operator_data>(
                            std::vector<std::shared_ptr<cucascade::data_batch>>{})),
                        std::move(global_state)),
      _reservation_bytes(reservation_bytes),
      _execute_for(execute_for)
  {
  }

  void execute(rmm::cuda_stream_view /*stream*/) override
  {
    auto& global = _global_state->cast<bracket_global_state>();
    auto& local  = _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>();
    if (_execute_for.count() > 0) { std::this_thread::sleep_for(_execute_for); }
    auto reservation = local.release_reservation();
    reservation.reset();
    global.recorder->record(get_task_id());
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
  std::chrono::milliseconds _execute_for;
};

//! One small GPU memory space plus a 2-worker executor and no downgrade executor, so a
//! reservation that does not fit has to WAIT for a release. Pipelines carry real query ids so
//! the executor attributes each dispatched task's pool slot to its query.
struct bracket_fixture {
  bool valid = false;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* mem_space = nullptr;
  sirius::exec::channel<std::unique_ptr<sirius::pipeline::task_request>> request_channel;
  std::unique_ptr<sirius::pipeline::gpu_pipeline_executor> executor;
  std::shared_ptr<bracket_recorder> recorder = std::make_shared<bracket_recorder>();

  sirius::pipeline::pipeline_build_context build_ctx{nullptr, true};
  //! Outlive every pipeline/task built from this fixture: ~gpu_pipeline_task walks the
  //! pipeline's operators via mark_task_completed().
  std::vector<std::unique_ptr<sirius::op::sirius_physical_operator>> operators;
  std::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>> pipelines;

  bracket_fixture()
  {
    try {
      cucascade::memory::reservation_manager_configurator builder;
      builder.set_number_of_gpus(1)
        .set_gpu_usage_limit(256 * kMiB)
        .set_reservation_fraction_per_gpu(0.75)
        .set_per_numa_region_capacity(256 * kMiB)
        .use_gpu_id_as_host_id()
        .track_reservation_per_stream(false)
        .set_reservation_fraction_per_numa_region(0.75);
      manager =
        std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    } catch (const std::exception& e) {
      WARN("Skipping error-bracket test (no usable GPU): " << e.what());
      return;
    }
    mem_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
    if (!mem_space) {
      WARN("Skipping error-bracket test: no GPU memory space available.");
      return;
    }

    sirius::exec::thread_pool_config config;
    config.num_threads        = 2;
    config.thread_name_prefix = "gpu-bracket-test";

    executor = std::make_unique<sirius::pipeline::gpu_pipeline_executor>(
      config,
      mem_space,
      request_channel.make_publisher(),
      nullptr,
      sirius::test::make_test_telemetry_context());
    valid = true;
  }

  ~bracket_fixture()
  {
    if (executor) { executor->stop(); }
    request_channel.close();
  }

  duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> make_pipeline(sirius::query_id_t query_id)
  {
    auto pipeline  = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(build_ctx);
    auto& op       = *operators.emplace_back(std::make_unique<sirius::op::sirius_physical_operator>(
      sirius::op::SiriusPhysicalOperatorType::FILTER,
      duckdb::vector<sirius::logical_type>{},
      /*estimated_cardinality=*/0));
    op.operator_id = operators.size() - 1;

    sirius::pipeline::sirius_pipeline_build_state build_state;
    build_state.set_pipeline_source(*pipeline, op);
    build_state.set_pipeline_sink(*pipeline, &op, /*sink_pipeline_count=*/1);

    pipeline->set_query_id(query_id);
    pipelines.push_back(pipeline);
    return pipeline;
  }

  //! Schedule a task attributed to @p query_id (each task gets its own single-op pipeline).
  void schedule(uint64_t task_id,
                sirius::query_id_t query_id,
                std::size_t reservation_bytes,
                std::chrono::milliseconds execute_for = std::chrono::milliseconds(0))
  {
    auto global_state = std::make_shared<bracket_global_state>(make_pipeline(query_id), recorder);
    executor->schedule(std::make_unique<bracket_task>(
      task_id, reservation_bytes, execute_for, std::move(global_state)));
  }

  //! Schedule a task with NO pipeline: its query is unknowable, so it runs on an untagged slot.
  void schedule_untagged(uint64_t task_id,
                         std::size_t reservation_bytes,
                         std::chrono::milliseconds execute_for = std::chrono::milliseconds(0))
  {
    auto global_state = std::make_shared<bracket_global_state>(nullptr, recorder);
    executor->schedule(std::make_unique<bracket_task>(
      task_id, reservation_bytes, execute_for, std::move(global_state)));
  }

  [[nodiscard]] static bool wait_until(const std::function<bool()>& done,
                                       std::chrono::seconds deadline)
  {
    const auto give_up = steady_clock::now() + deadline;
    while (!done()) {
      if (steady_clock::now() > give_up) { return false; }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
  }
};

constexpr uint64_t kCoTenantHungry = 2000;
constexpr uint64_t kRunningA       = 2001;
constexpr uint64_t kQueuedA        = 2002;
constexpr uint64_t kUntaggedSlow   = 2003;

}  // namespace

TEST_CASE("error bracket: a co-tenant's parked memory wait no longer extends cleanup",
          "[task_executor][gpu_pipeline_executor][memory_wait][concurrency]")
{
  bracket_fixture f;
  if (!f.valid) { return; }
  f.executor->start();

  const auto query_a = sirius::make_query_id(11);  // the erroring query
  const auto query_b = sirius::make_query_id(22);  // the co-tenant

  // Occupy the space so only a small head room is grantable; the co-tenant's demand cannot be
  // met until the hold is released, so its worker parks in a memory wait ATTRIBUTED TO B.
  const std::size_t space_max = f.mem_space->get_max_memory();
  const std::size_t head_room = 8 * kMiB;
  REQUIRE(space_max > 4 * head_room);
  auto hold = f.mem_space->make_reservation_or_null(space_max - head_room);
  REQUIRE(hold);

  f.schedule(kCoTenantHungry, query_b, space_max / 2);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  const bool cotenant_parked = !f.recorder->has_executed(kCoTenantHungry);

  // Query A: one task provably IN FLIGHT when the bracket starts (long execute), and one task
  // stuck in the executor queue behind it (both workers are busy: B parked + A running, and the
  // manager blocks in reserve() at capacity).
  f.schedule(kRunningA, query_a, 1 * kMiB, std::chrono::milliseconds(400));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  f.schedule(kQueuedA, query_a, 1 * kMiB);

  // The error-path bracket for query A, from its own thread so a regression to a whole-pool
  // wait shows up as a timeout rather than a hung test binary.
  std::atomic<bool> bracket_returned{false};
  std::thread bracket([&] {
    f.executor->wait_and_drain_query(query_a);
    bracket_returned.store(true, std::memory_order_release);
  });

  // The bracket must return while the co-tenant is STILL parked on its memory wait. With a
  // whole-pool wait it cannot: B's parked worker holds an active slot until the hold releases.
  const bool returned = bracket_fixture::wait_until(
    [&] { return bracket_returned.load(std::memory_order_acquire); }, std::chrono::seconds(20));

  // Snapshot the invariants at bracket-return time, BEFORE unblocking anything.
  const bool running_a_completed  = f.recorder->has_executed(kRunningA);
  const bool cotenant_still_waits = !f.recorder->has_executed(kCoTenantHungry);

  // The queued A task was drained: give it every opportunity to (incorrectly) run.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  const bool queued_a_never_ran = !f.recorder->has_executed(kQueuedA);

  // Unblock the co-tenant and let everything settle before asserting. In the whole-pool
  // failure mode the bracket unblocks once the co-tenant finishes, so a late join still works.
  hold.reset();
  const bool cotenant_finished = bracket_fixture::wait_until(
    [&] { return f.recorder->has_executed(kCoTenantHungry); }, std::chrono::seconds(30));
  const bool joinable = bracket_fixture::wait_until(
    [&] { return bracket_returned.load(std::memory_order_acquire); }, std::chrono::seconds(30));
  if (joinable) {
    bracket.join();
  } else {
    bracket.detach();  // genuinely wedged; the REQUIREs below report the failure
  }

  REQUIRE(cotenant_parked);
  REQUIRE(returned);
  // Plan-safety invariant: when wait_and_drain_query returns, no thread is still executing a
  // task that references the failing query's plan — the in-flight A task ran to completion
  // first, and the queued one was dropped, never to touch the plan again.
  REQUIRE(running_a_completed);
  REQUIRE(cotenant_still_waits);
  REQUIRE(queued_a_never_ran);
  REQUIRE(cotenant_finished);
  REQUIRE_FALSE(f.recorder->has_executed(kQueuedA));

  f.executor->stop();
}

TEST_CASE("error bracket: an untagged in-flight task is still waited for conservatively",
          "[task_executor][gpu_pipeline_executor][concurrency]")
{
  bracket_fixture f;
  if (!f.valid) { return; }
  f.executor->start();

  const auto query_a = sirius::make_query_id(33);

  // A task with no pipeline: attribution is impossible, so its slot stays untagged. It could in
  // principle belong to the erroring query, so the bracket must NOT return while it runs.
  f.schedule_untagged(kUntaggedSlow, 1 * kMiB, std::chrono::milliseconds(500));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::atomic<bool> bracket_returned{false};
  std::thread bracket([&] {
    f.executor->wait_and_drain_query(query_a);
    bracket_returned.store(true, std::memory_order_release);
  });

  const bool returned = bracket_fixture::wait_until(
    [&] { return bracket_returned.load(std::memory_order_acquire); }, std::chrono::seconds(20));
  const bool untagged_completed = f.recorder->has_executed(kUntaggedSlow);
  if (returned) {
    bracket.join();
  } else {
    bracket.detach();  // genuinely wedged; the REQUIREs below report the failure
  }

  REQUIRE(returned);
  // The bracket waited the untagged task out rather than racing past it.
  REQUIRE(untagged_completed);

  f.executor->stop();
}
