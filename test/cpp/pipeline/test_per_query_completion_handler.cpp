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
 * @file test_per_query_completion_handler.cpp
 * @brief A query's completion handler travels on its tasks' global state.
 *
 * It used to live on task_scheduler and be pushed into every executor, so a second query
 * overwrote the first's: query A's completion signalled B's promise, and A's failure poisoned B.
 * These cases pin the isolation that replaces it, plus the lifetime property that lets the
 * owning sirius_engine be destroyed while tasks are still unwinding.
 */

#include "catch.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "utils/telemetry_utils.hpp"

#include <chrono>
#include <future>
#include <memory>

namespace {

using sirius::pipeline::completion_handler;
using sirius::pipeline::sirius_pipeline_task_global_state;

//! A global state carrying its query's handler, as task_creator::prepare_for_query builds it.
std::shared_ptr<sirius_pipeline_task_global_state> make_global_state(
  const sirius::pipeline::pipeline_build_context& ctx, std::shared_ptr<completion_handler> handler)
{
  auto pipeline = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(ctx);
  auto gs       = std::make_shared<sirius_pipeline_task_global_state>(
    pipeline, sirius::test::make_test_telemetry_context());
  gs->set_completion_handler(std::move(handler));
  return gs;
}

bool is_ready(std::future<void>& f)
{
  return f.wait_for(std::chrono::seconds{0}) == std::future_status::ready;
}

}  // namespace

TEST_CASE("completing one query leaves another query's future unset",
          "[completion_handler][per_query]")
{
  sirius::pipeline::pipeline_build_context ctx{nullptr, true};
  auto handler_a = std::make_shared<completion_handler>();
  auto handler_b = std::make_shared<completion_handler>();
  auto future_a  = handler_a->get_awaitable();
  auto future_b  = handler_b->get_awaitable();

  auto gs_a = make_global_state(ctx, handler_a);
  auto gs_b = make_global_state(ctx, handler_b);

  // Query A finishes. With a single scheduler-wide handler this would have satisfied whichever
  // query happened to be installed last.
  gs_a->get_completion_handler()->mark_completed();

  REQUIRE(is_ready(future_a));
  CHECK_FALSE(is_ready(future_b));
  future_a.get();  // no throw
}

TEST_CASE("one query's failure does not poison another's handler",
          "[completion_handler][per_query]")
{
  sirius::pipeline::pipeline_build_context ctx{nullptr, true};
  auto handler_a = std::make_shared<completion_handler>();
  auto handler_b = std::make_shared<completion_handler>();
  auto future_b  = handler_b->get_awaitable();

  auto gs_a = make_global_state(ctx, handler_a);
  auto gs_b = make_global_state(ctx, handler_b);

  gs_a->get_completion_handler()->report_error("query A failed");

  CHECK(handler_a->has_error());
  // B stays clean. The executor's reschedule path consults has_error() to decide whether to
  // abandon a task, so a shared handler made A's failure silently stop B's rescheduling.
  CHECK_FALSE(handler_b->has_error());
  CHECK_FALSE(is_ready(future_b));
}

TEST_CASE("the handler outlives the engine-side reference", "[completion_handler][per_query]")
{
  sirius::pipeline::pipeline_build_context ctx{nullptr, true};
  auto handler = std::make_shared<completion_handler>();
  auto future  = handler->get_awaitable();
  auto gs      = make_global_state(ctx, handler);

  // sirius_engine is destroyed in cleanup_internal, BEFORE run_mandatory_cleanup drains the
  // queues — so dropping the owning reference must not invalidate the handler a still-unwinding
  // task holds through its global state. A raw pointer here would dangle.
  std::weak_ptr<completion_handler> observer = handler;
  handler.reset();
  REQUIRE_FALSE(observer.expired());

  gs->get_completion_handler()->report_error("late failure after the engine went away");
  CHECK(is_ready(future));
  CHECK_THROWS(future.get());

  // Released only when the last task-side reference goes.
  gs.reset();
  CHECK(observer.expired());
}

TEST_CASE("every pipeline of one query shares its handler", "[completion_handler][per_query]")
{
  sirius::pipeline::pipeline_build_context ctx{nullptr, true};
  auto handler = std::make_shared<completion_handler>();

  // prepare_for_query stamps the same handler onto each pipeline's global state, so whichever
  // pipeline's task reports, it reaches the one promise the engine is waiting on.
  auto gs_first  = make_global_state(ctx, handler);
  auto gs_second = make_global_state(ctx, handler);

  CHECK(gs_first->get_completion_handler().get() == gs_second->get_completion_handler().get());
  CHECK(gs_first->get_completion_handler().get() == handler.get());
}

TEST_CASE("a global state built without a query carries no handler",
          "[completion_handler][per_query]")
{
  sirius::pipeline::pipeline_build_context ctx{nullptr, true};
  auto pipeline = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(ctx);
  sirius_pipeline_task_global_state gs(pipeline, sirius::test::make_test_telemetry_context());

  // Reporting sites are all null-guarded, so a state built outside a query (tests) is inert
  // rather than a crash.
  CHECK(gs.get_completion_handler() == nullptr);
}
