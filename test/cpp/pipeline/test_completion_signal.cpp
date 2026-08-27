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

#include "helper/logical_type.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/sirius_pipeline.hpp"

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
duckdb::shared_ptr<sirius_pipeline> make_finishable_pipeline(sirius_physical_operator& sink_op)
{
  auto pipeline =
    duckdb::make_shared_ptr<sirius_pipeline>(sirius::pipeline::pipeline_build_context{nullptr});
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
