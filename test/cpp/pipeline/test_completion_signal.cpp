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

/**
 * Regression tests for #1486: completion driven outside the GPU epilogue must still signal.
 * Bounded waits turn a lost signal into a test failure instead of a hang.
 */

#include "creator/task_creator.hpp"
#include "helper/logical_type.hpp"
#include "op/input_free_operator_data.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "operator/operator_test_utils.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/task_scheduler.hpp"
#include "utils/telemetry_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>

#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <vector>

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::pipeline::completion_handler;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;
using namespace std::chrono_literals;

namespace {

/// Poll until @p done() or fail after @p timeout.
template <typename Pred>
void wait_or_fail(Pred done, std::chrono::seconds timeout, const char* what)
{
  const auto start = std::chrono::steady_clock::now();
  while (!done()) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) { FAIL(what); }
  }
}

/// Build a zero-task pipeline that can finish without an epilogue fallback.
std::shared_ptr<sirius_pipeline> make_finishable_pipeline(sirius_physical_operator& sink_op)
{
  auto pipeline =
    std::make_shared<sirius_pipeline>(sirius::pipeline::pipeline_build_context{nullptr});
  sirius_pipeline_build_state build_state;
  build_state.set_pipeline_sink(*pipeline, &sink_op, 0);
  return pipeline;
}

sirius_physical_operator make_sink(SiriusPhysicalOperatorType type)
{
  return sirius_physical_operator{type, duckdb::vector<sirius::logical_type>{}, 0};
}

}  // namespace

TEST_CASE("Terminal pipeline finishing off the epilogue still signals completion",
          "[pipeline][completion][issue-1486]")
{
  auto sink     = make_sink(SiriusPhysicalOperatorType::RESULT_COLLECTOR);
  auto pipeline = make_finishable_pipeline(sink);
  REQUIRE(pipeline->is_query_terminal());

  auto handler = std::make_shared<completion_handler>();
  pipeline->set_completion_handler(handler);
  auto awaitable = handler->get_awaitable();

  // Model task-creator and end-of-stream completion outside the GPU epilogue.
  std::thread driver([&] { pipeline->update_pipeline_status(false); });
  driver.join();

  REQUIRE(pipeline->is_pipeline_finished());
  wait_or_fail([&] { return awaitable.wait_for(0s) == std::future_status::ready; },
               std::chrono::seconds(10),
               "terminal pipeline finished without signalling completion (#1486)");
  REQUIRE(handler->is_completed());
  REQUIRE_NOTHROW(awaitable.get());
}

TEST_CASE("A streaming-sink pipeline is query-terminal and signals the same way",
          "[pipeline][completion][issue-1486]")
{
  auto sink     = make_sink(SiriusPhysicalOperatorType::STREAMING_SINK);
  auto pipeline = make_finishable_pipeline(sink);
  REQUIRE(pipeline->is_query_terminal());

  auto handler = std::make_shared<completion_handler>();
  pipeline->set_completion_handler(handler);
  auto awaitable = handler->get_awaitable();

  std::thread driver([&] { pipeline->update_pipeline_status(false); });
  driver.join();

  wait_or_fail([&] { return awaitable.wait_for(0s) == std::future_status::ready; },
               std::chrono::seconds(10),
               "streaming-sink pipeline finished without signalling completion (#1486)");
  REQUIRE(handler->is_completed());
}

TEST_CASE("Re-entering the finish transition re-signals harmlessly",
          "[pipeline][completion][issue-1486]")
{
  auto sink     = make_sink(SiriusPhysicalOperatorType::RESULT_COLLECTOR);
  auto pipeline = make_finishable_pipeline(sink);

  auto handler = std::make_shared<completion_handler>();
  pipeline->set_completion_handler(handler);
  auto awaitable = handler->get_awaitable();

  // Model duplicate transition and epilogue signals, including concurrent re-entry.
  std::vector<std::thread> drivers;
  drivers.reserve(4);
  for (int i = 0; i < 4; ++i) {
    drivers.emplace_back([&] { pipeline->update_pipeline_status(false); });
  }
  for (auto& driver : drivers) {
    driver.join();
  }
  REQUIRE_NOTHROW(handler->mark_completed());
  REQUIRE_NOTHROW(pipeline->update_pipeline_status(false));

  REQUIRE(awaitable.wait_for(0s) == std::future_status::ready);
  REQUIRE_NOTHROW(awaitable.get());
  REQUIRE_FALSE(handler->has_error());
}

TEST_CASE("A non-terminal pipeline finishing does not signal the query",
          "[pipeline][completion][issue-1486]")
{
  // A hash-join build pipeline is intermediate and must not complete the query.
  auto sink     = make_sink(SiriusPhysicalOperatorType::HASH_JOIN);
  auto pipeline = make_finishable_pipeline(sink);
  REQUIRE_FALSE(pipeline->is_query_terminal());

  auto handler = std::make_shared<completion_handler>();
  pipeline->set_completion_handler(handler);
  auto awaitable = handler->get_awaitable();

  pipeline->update_pipeline_status(false);

  REQUIRE(pipeline->is_pipeline_finished());
  REQUIRE_FALSE(handler->is_completed());
  REQUIRE(awaitable.wait_for(50ms) == std::future_status::timeout);
}

TEST_CASE("A pipeline left over from a finished query cannot signal the next one",
          "[pipeline][completion][issue-1486]")
{
  // Dropping the scheduler-owned handler models retirement at the next query.
  auto sink     = make_sink(SiriusPhysicalOperatorType::RESULT_COLLECTOR);
  auto pipeline = make_finishable_pipeline(sink);

  auto handler = std::make_shared<completion_handler>();
  pipeline->set_completion_handler(handler);
  handler.reset();

  REQUIRE_NOTHROW(pipeline->update_pipeline_status(false));
  REQUIRE(pipeline->is_pipeline_finished());
}

namespace {

class waiting_source final : public sirius_physical_operator {
 public:
  explicit waiting_source(bool return_input)
    : sirius_physical_operator(SiriusPhysicalOperatorType::PROJECTION, {}, 0),
      return_input_(return_input),
      release_(release.get_future())
  {
  }

  std::optional<sirius::op::task_creation_hint> get_next_task_hint() override
  {
    return sirius::op::task_creation_hint{sirius::op::TaskCreationHint::READY, this};
  }

  bool all_ports_empty() override { return exhausted.load(); }

  std::unique_ptr<sirius::op::operator_data> get_next_task_input_data() override
  {
    if (claims++ == 0) return std::make_unique<sirius::op::input_free_operator_data>(0);
    // The last input has been claimed, but its task has not been registered yet.
    exhausted = true;
    entered.set_value();
    release_.wait();
    if (return_input_) return std::make_unique<sirius::op::input_free_operator_data>(0);
    return nullptr;
  }

  std::promise<void> entered;
  std::promise<void> release;
  std::atomic<bool> exhausted{false};
  unsigned claims = 0;

 private:
  bool return_input_;
  std::future<void> release_;
};

// Supply the plan state without a SQL frontend; scheduling and task construction stay real.
class registered_creator final : public sirius::creator::task_creator {
 public:
  using task_creator::task_creator;

  void register_pipeline(std::shared_ptr<sirius_pipeline> const& pipeline,
                         std::shared_ptr<completion_handler> const& completion)
  {
    auto state                = get_query_task_global_state(pipeline->get_query_id());
    state->completion_handler = completion;
    auto global               = std::make_shared<sirius::pipeline::gpu_pipeline_task_global_state>(
      pipeline, sirius::test::make_test_telemetry_context());
    global->set_completion_handler(completion);
    state->global_states.emplace(pipeline->get_source()->get_operator_id(), std::move(global));
    pipeline->set_task_creator(this);
    pipeline->set_completion_handler(completion);
  }
};

}  // namespace

TEST_CASE("Task creation waiting on input allows completion and rechecks closure",
          "[pipeline][completion][task_creation_wait]")
{
  auto const return_input = GENERATE(true, false);
  CAPTURE(return_input);
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto memory = sirius::test::operator_utils::initialize_memory_manager();
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius::pipeline::task_scheduler scheduler(
    lifecycle, {.num_threads = 1}, *memory, sirius::test::make_test_telemetry_context());
  waiting_source source(return_input);
  auto sink     = make_sink(SiriusPhysicalOperatorType::RESULT_COLLECTOR);
  auto pipeline = make_finishable_pipeline(sink);
  sirius_pipeline_build_state{}.set_pipeline_source(*pipeline, source);
  pipeline->set_pipeline_id(0);
  auto const query_id = sirius::make_query_id(1);
  pipeline->set_query_id(query_id);
  source.set_pipeline(pipeline);
  source.operator_id = 0;
  auto handler       = std::make_shared<completion_handler>();
  auto finished      = handler->get_awaitable();
  registered_creator creator(lifecycle, {.thread_pool = {.num_threads = 1}}, *memory);
  lifecycle.open_query(query_id);
  creator.set_client_context(query_id, *con.context);
  creator.set_task_scheduler(scheduler);
  creator.register_pipeline(pipeline, handler);
  auto entered = source.entered.get_future();
  creator.start_thread_pool();
  creator.schedule(&source, query_id);
  auto const claimed = entered.wait_for(5s) == std::future_status::ready;
  std::future<void> completed;
  bool progressed = false;
  bool premature  = false;
  auto first      = scheduler.get_pipeline_task_queue()->try_pop();
  if (claimed && first) {
    // Task destruction reports completion, just as it does after GPU execution.
    // Keep the scheduler stopped so the test controls retirement without running kernels.
    completed =
      std::async(std::launch::async, [task = std::move(*first)]() mutable { task.reset(); });
    progressed = completed.wait_for(2s) == std::future_status::ready;
    premature =
      pipeline->is_pipeline_finished() || finished.wait_for(0s) == std::future_status::ready;
  }
  // An independent escape releases the source even with the old deadlocking status lock.
  // Join both sides before assertions, so a regression fails instead of hanging the suite.
  source.release.set_value();
  if (completed.valid()) completed.get();
  if (first) first->reset();
  creator.stop();
  auto completed_count = pipeline->get_tasks_completed();
  auto closed          = pipeline->is_pipeline_finished();
  auto queued          = scheduler.get_pipeline_task_queue()->size();
  scheduler.get_pipeline_task_queue()->drain();
  REQUIRE(claimed);
  REQUIRE(progressed);
  CHECK_FALSE(premature);
  REQUIRE_FALSE(handler->has_error());
  CHECK(source.claims == 2);
  CHECK(queued == (return_input ? 1 : 0));
  CHECK(pipeline->get_tasks_created() == (return_input ? 2 : 1));
  CHECK(completed_count == 1);
  CHECK(closed == !return_input);
  REQUIRE(finished.wait_for(0s) == std::future_status::ready);
  REQUIRE_NOTHROW(finished.get());
}
