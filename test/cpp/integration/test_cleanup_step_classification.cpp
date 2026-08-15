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

// D5, STEP-LEVEL: run_mandatory_cleanup classifies failures per STEP. A failed drain of a
// SHARED downgrade executor or a repository-registry invariant break is evidence the SHARED
// runtime is wedged — every co-tenant's cleanup hits the same subsystem next — so it takes
// the process-wide latch (runtime_health::UNAVAILABLE). A per-query step failure (e.g. the
// query's own task_creator reset) stays contained: the query errors, its state is dropped
// best-effort, and healthy co-tenants keep running.
//
// Driven on a BARE SiriusContext (no initialize, no GPU): every subsystem pointer is null and
// the executor list is empty, so the cleanup walks all its steps as no-ops — the injected
// fault is the only thing that can fail, which makes the dispatch deterministic.

#include "query_id.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <sirius_context.hpp>

#include <stdexcept>
#include <string>
#include <string_view>

namespace {

duckdb::SiriusContext::runtime_health health(const duckdb::SiriusContext& ctx)
{
  return ctx.get_runtime_health();
}

}  // namespace

TEST_CASE("cleanup step classification: shared-step failures latch the runtime",
          "[concurrency][cleanup_classification]")
{
  const auto query_id = sirius::make_query_id(42);

  SECTION("a shared downgrade-executor drain failure takes the shared verdict")
  {
    duckdb::SiriusContext ctx;
    ctx.inject_cleanup_step_fault_for_testing([](std::string_view step) {
      if (step == "downgrade_drain") { throw std::runtime_error("injected drain wedge"); }
    });

    ctx.run_mandatory_cleanup_backstop_for_testing(query_id);

    CHECK(health(ctx) == duckdb::SiriusContext::runtime_health::UNAVAILABLE);
    // Shared verdict, not a per-query containment.
    CHECK(ctx.per_query_cleanup_failures() == 0);
  }

  SECTION("a repository-registry erase failure takes the shared verdict")
  {
    duckdb::SiriusContext ctx;
    ctx.inject_cleanup_step_fault_for_testing([](std::string_view step) {
      if (step == "repository_erase") { throw std::runtime_error("injected registry break"); }
    });

    ctx.run_mandatory_cleanup_backstop_for_testing(query_id);

    CHECK(health(ctx) == duckdb::SiriusContext::runtime_health::UNAVAILABLE);
    CHECK(ctx.per_query_cleanup_failures() == 0);
  }

  SECTION("a per-query step failure is contained to the query")
  {
    duckdb::SiriusContext ctx;
    ctx.inject_cleanup_step_fault_for_testing([](std::string_view step) {
      if (step == "task_creator_reset") { throw std::runtime_error("injected creator failure"); }
    });

    ctx.run_mandatory_cleanup_backstop_for_testing(query_id);

    // Per-query verdict: counted, contained, and the shared runtime stays healthy.
    CHECK(health(ctx) == duckdb::SiriusContext::runtime_health::OK);
    CHECK(ctx.per_query_cleanup_failures() == 1);
  }

  SECTION("a clean cleanup classifies nothing")
  {
    duckdb::SiriusContext ctx;
    ctx.run_mandatory_cleanup_backstop_for_testing(query_id);
    CHECK(health(ctx) == duckdb::SiriusContext::runtime_health::OK);
    CHECK(ctx.per_query_cleanup_failures() == 0);
  }
}
