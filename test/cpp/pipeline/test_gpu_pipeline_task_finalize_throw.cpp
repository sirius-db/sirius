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

// ~gpu_pipeline_task calls mark_task_completed(), which runs finalize_operator() on every
// operator — device work that throws. The destructor is implicitly noexcept, so an unguarded
// throw there is std::terminate.
//
// Each case below destroys a task whose pipeline throws out of finalize. Remove the guard and
// they do not fail: the test binary aborts, which is the signal.

#include "catch.hpp"
#include "exec/channel.hpp"
#include "exec/config.hpp"
#include "helper/logical_type.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_request.hpp"
#include "utils/telemetry_utils.hpp"

#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr char const* kFinalizeThrowMessage = "test: on_finalize_operator threw";
constexpr std::size_t kReservationBytes     = 4 * 1024 * 1024;

using sirius::op::SiriusPhysicalOperatorType;
using sirius::pipeline::gpu_pipeline_task;
using sirius::pipeline::gpu_pipeline_task_local_state;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;
using sirius::pipeline::sirius_pipeline_task_global_state;

// Reports its limit as exhausted, which is all update_pipeline_status() needs to declare the
// pipeline finished, then throws out of finalize.
class throwing_finalize_operator : public sirius::op::sirius_physical_operator {
 public:
  throwing_finalize_operator()
    : sirius_physical_operator(
        SiriusPhysicalOperatorType::FILTER, duckdb::vector<sirius::logical_type>{}, 0)
  {
  }

  std::string get_name() const override { return "throwing_finalize_operator"; }

  bool is_limit_exhausted() const override { return true; }

  std::atomic<int> finalize_calls{0};

 protected:
  void on_finalize_operator() override
  {
    finalize_calls.fetch_add(1, std::memory_order_relaxed);
    throw std::runtime_error(kFinalizeThrowMessage);
  }
};

std::unique_ptr<gpu_pipeline_task_local_state> make_local_state()
{
  return std::make_unique<gpu_pipeline_task_local_state>(
    std::make_unique<sirius::op::pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{}));
}

std::unique_ptr<gpu_pipeline_task> make_task(
  const std::shared_ptr<sirius_pipeline_task_global_state>& global_state)
{
  return std::make_unique<gpu_pipeline_task>(
    /*task_id=*/1,
    std::vector<cucascade::shared_data_repository*>{},
    make_local_state(),
    global_state);
}

std::string message_of(std::exception_ptr error)
{
  try {
    std::rethrow_exception(std::move(error));
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "<non-std exception>";
  }
}

}  // namespace

TEST_CASE("a finalize throw in the task destructor does not abort, and completion still stands",
          "[pipeline][gpu_pipeline_task][finalize_throw]")
{
  auto pipeline = duckdb::make_shared_ptr<sirius_pipeline>(
    sirius::pipeline::pipeline_build_context{sirius::test::make_test_telemetry_context()});
  pipeline->set_pipeline_id(7);

  throwing_finalize_operator op;
  sirius_pipeline_build_state build_state;
  build_state.add_pipeline_operator(*pipeline, op);

  auto global_state = std::make_shared<sirius_pipeline_task_global_state>(
    pipeline, sirius::test::make_test_telemetry_context());

  // Destroying the task runs mark_task_completed(); the process must survive it.
  make_task(global_state).reset();

  CHECK(op.finalize_calls.load() == 1);

  // update_pipeline_status() stores pipeline_finished before running the finalize loop, so the
  // throw lands after the flip and the executor epilogue still sees a finished pipeline.
  CHECK(pipeline->is_pipeline_finished());
  CHECK(pipeline->get_tasks_created() == 1);
  CHECK(pipeline->get_tasks_completed() == 1);

  // The error is parked for whoever destroyed the task, and is handed out exactly once.
  auto parked = pipeline->take_task_completion_error();
  REQUIRE(parked != nullptr);
  CHECK(message_of(parked) == kFinalizeThrowMessage);
  CHECK(pipeline->take_task_completion_error() == nullptr);
}

TEST_CASE("a throw raised before the pipeline_finished flip is still parked for the caller",
          "[pipeline][gpu_pipeline_task][finalize_throw]")
{
  // No operators and no sink makes update_pipeline_status() throw "First node of pipeline is
  // nullptr" — the throw source that fires before pipeline_finished is set. Nothing can finish
  // the pipeline afterwards, so the parked error is the only way out of a hang.
  auto pipeline = duckdb::make_shared_ptr<sirius_pipeline>(
    sirius::pipeline::pipeline_build_context{sirius::test::make_test_telemetry_context()});
  pipeline->set_pipeline_id(9);

  auto global_state = std::make_shared<sirius_pipeline_task_global_state>(
    pipeline, sirius::test::make_test_telemetry_context());

  make_task(global_state).reset();

  CHECK_FALSE(pipeline->is_pipeline_finished());

  auto parked = pipeline->take_task_completion_error();
  REQUIRE(parked != nullptr);
  CHECK(message_of(parked).find("First node of pipeline is nullptr") != std::string::npos);
}

TEST_CASE("only the first task-completion error is kept",
          "[pipeline][gpu_pipeline_task][finalize_throw]")
{
  auto pipeline = duckdb::make_shared_ptr<sirius_pipeline>(
    sirius::pipeline::pipeline_build_context{sirius::test::make_test_telemetry_context()});

  pipeline->record_task_completion_error(std::make_exception_ptr(std::runtime_error("first")));
  pipeline->record_task_completion_error(std::make_exception_ptr(std::runtime_error("second")));
  pipeline->record_task_completion_error(nullptr);

  auto parked = pipeline->take_task_completion_error();
  REQUIRE(parked != nullptr);
  CHECK(message_of(parked) == "first");
  CHECK(pipeline->take_task_completion_error() == nullptr);
}

namespace {

// Executor-driven task. Reservation handling mirrors test_gpu_pipeline_executor.cpp; the work
// itself is deliberately nothing, since the point of the task is its destructor.
class finalize_throwing_task : public gpu_pipeline_task {
 public:
  finalize_throwing_task(uint64_t task_id,
                         std::shared_ptr<sirius_pipeline_task_global_state> global_state,
                         std::atomic<int>& executed)
    : gpu_pipeline_task(task_id,
                        std::vector<cucascade::shared_data_repository*>{},
                        make_local_state(),
                        std::move(global_state)),
      _executed(executed)
  {
  }

  void execute(rmm::cuda_stream_view stream) override
  {
    auto& local      = _local_state->cast<gpu_pipeline_task_local_state>();
    auto reservation = local.release_reservation();
    if (reservation) {
      auto* allocator =
        reservation
          ->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
      if (allocator && allocator->attach_reservation_to_tracker(stream, std::move(reservation))) {
        allocator->reset_stream_reservation(stream);
      }
    }
    _executed.fetch_add(1, std::memory_order_relaxed);
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
  std::atomic<int>& _executed;
};

}  // namespace

TEST_CASE("the executor fails the query when the task destructor's completion throws",
          "[pipeline][gpu_pipeline_task][finalize_throw]")
{
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  try {
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(256 * 1024 * 1024)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(1024ULL * 1024 * 1024)
      .use_gpu_id_as_host_id()
      .track_reservation_per_stream(false)
      .set_reservation_fraction_per_numa_region(0.75);
    manager = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
  } catch (const std::exception& e) {
    WARN("Skipping test due to insufficient GPUs: " << e.what());
    return;
  }

  auto* mem_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  if (!mem_space) {
    WARN("Skipping test because no GPU memory space is available.");
    return;
  }

  auto pipeline = duckdb::make_shared_ptr<sirius_pipeline>(
    sirius::pipeline::pipeline_build_context{sirius::test::make_test_telemetry_context()});
  pipeline->set_pipeline_id(11);

  throwing_finalize_operator op;
  sirius::op::sirius_physical_operator result_collector(
    SiriusPhysicalOperatorType::RESULT_COLLECTOR, duckdb::vector<sirius::logical_type>{}, 1);
  sirius_pipeline_build_state build_state;
  build_state.add_pipeline_operator(*pipeline, op);
  build_state.set_pipeline_sink(*pipeline, &result_collector, 0);
  REQUIRE(pipeline->is_query_terminal());

  sirius::exec::channel<std::unique_ptr<sirius::pipeline::task_request>> request_channel;
  sirius::exec::thread_pool_config config;
  config.num_threads        = 1;
  config.thread_name_prefix = "finalize-throw-test";

  sirius::pipeline::gpu_pipeline_executor executor(config,
                                                   mem_space,
                                                   request_channel.make_publisher(),
                                                   nullptr,
                                                   sirius::test::make_test_telemetry_context());
  sirius::pipeline::completion_handler handler;
  executor.set_completion_handler(&handler);
  auto awaitable = handler.get_awaitable();

  auto global_state = std::make_shared<sirius_pipeline_task_global_state>(
    pipeline, sirius::test::make_test_telemetry_context());

  std::atomic<int> executed{0};
  executor.start();
  executor.schedule(std::make_unique<finalize_throwing_task>(1, global_state, executed));

  // The query must terminate — not abort, and not hang on a swallowed throw that signals nobody.
  const auto status = awaitable.wait_for(std::chrono::seconds(30));
  executor.stop();
  request_channel.close();

  REQUIRE(status == std::future_status::ready);
  CHECK(executed.load() == 1);
  CHECK(handler.has_error());
  CHECK_THROWS_WITH(awaitable.get(), Catch::Contains(kFinalizeThrowMessage));
}
