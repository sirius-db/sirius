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

#include "catch.hpp"
#include "pipeline/completion_handler.hpp"

#include <duckdb/common/exception.hpp>

#include <barrier>
#include <chrono>
#include <thread>

using sirius::pipeline::completion_handler;
using namespace std::chrono_literals;

TEST_CASE("GPU success waits for preparation closure and physical quiescence",
          "[scan_preparation][completion]")
{
  completion_handler handler;
  auto future = handler.get_awaitable();
  handler.begin_preparation();
  handler.mark_completed();
  CHECK(future.wait_for(0ms) == std::future_status::timeout);
  handler.close_preparation_inputs();
  CHECK(future.wait_for(0ms) == std::future_status::timeout);
  handler.preparation_quiescent();
  REQUIRE(future.wait_for(0ms) == std::future_status::ready);
  CHECK_NOTHROW(future.get());
}

TEST_CASE("Preparation failure wakes the query without a coordinator event",
          "[scan_preparation][completion]")
{
  completion_handler handler;
  auto future = handler.get_awaitable();
  handler.begin_preparation();
  handler.mark_completed();
  handler.report_error(std::make_exception_ptr(std::runtime_error("preparation failed")));
  REQUIRE(future.wait_for(0ms) == std::future_status::ready);
  CHECK_THROWS_WITH(future.get(), "preparation failed");
  CHECK(handler.has_error());
}

TEST_CASE("Queries without preparation still complete immediately",
          "[scan_preparation][completion]")
{
  completion_handler handler;
  auto future = handler.get_awaitable();
  handler.mark_completed();
  REQUIRE(future.wait_for(0ms) == std::future_status::ready);
  CHECK_NOTHROW(future.get());
}

TEST_CASE("Preparation quiescence alone is not input closure", "[scan_preparation][completion]")
{
  completion_handler handler;
  auto future = handler.get_awaitable();
  handler.begin_preparation();
  handler.preparation_quiescent();
  handler.mark_completed();
  CHECK(future.wait_for(0ms) == std::future_status::timeout);
  handler.close_preparation_inputs();
  REQUIRE(future.wait_for(0ms) == std::future_status::ready);
  CHECK_NOTHROW(future.get());
}
TEST_CASE("Preparation cannot register after GPU work or terminal success",
          "[scan_preparation][completion]")
{
  completion_handler handler;
  handler.mark_completed();
  CHECK_THROWS_AS(handler.begin_preparation(), std::logic_error);
}

TEST_CASE("Preparation can finish before GPU without declaring query success",
          "[scan_preparation][completion]")
{
  completion_handler handler;
  auto future = handler.get_awaitable();
  handler.begin_preparation();
  CHECK_THROWS_AS(handler.begin_preparation(), std::logic_error);
  handler.close_preparation_inputs();
  handler.preparation_quiescent();
  CHECK_FALSE(handler.is_completed());
  CHECK(future.wait_for(0ms) == std::future_status::timeout);
  handler.mark_completed();
  REQUIRE(future.wait_for(0ms) == std::future_status::ready);
  CHECK_NOTHROW(future.get());
}

TEST_CASE("GPU completion racing preparation failure cannot bypass an open gate",
          "[scan_preparation][completion]")
{
  for (int iteration = 0; iteration < 50; ++iteration) {
    completion_handler handler;
    auto future = handler.get_awaitable();
    handler.begin_preparation();
    std::barrier start(3);
    std::jthread gpu([&] {
      start.arrive_and_wait();
      handler.mark_completed();
    });
    std::jthread preparation([&] {
      start.arrive_and_wait();
      handler.report_error(sirius::scan_manager::preparation_failure{
        sirius::transparent::late_failure_cause::reader_io, {}, "delayed read failed", {}});
    });
    start.arrive_and_wait();
    gpu.join();
    preparation.join();
    REQUIRE(future.wait_for(0ms) == std::future_status::ready);
    CHECK_THROWS_WITH(future.get(), "delayed read failed");
    CHECK(handler.failure().cause == sirius::transparent::late_failure_cause::reader_io);
  }
}

TEST_CASE("Cancellation wakes an armed query before preparation input closure",
          "[scan_preparation][completion]")
{
  completion_handler handler;
  auto future = handler.get_awaitable();
  handler.begin_preparation();
  handler.report_error(std::make_exception_ptr(duckdb::InterruptException()));
  REQUIRE(future.wait_for(0ms) == std::future_status::ready);
  CHECK_THROWS_AS(future.get(), duckdb::InterruptException);
  CHECK(handler.has_error());
  handler.close_preparation_inputs();
  handler.preparation_quiescent();
  handler.mark_completed();
  CHECK(handler.has_error());
}
