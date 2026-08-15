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

// Register H8: window ids come from a 32-bit counter and wrap after 2^32
// queries. Two things must never come out of the wrap:
//
//   - raw value 0: make_query_id(0) is the "no query" sentinel (unattributed
//     downgrade requests, pre-window defaults), so a window minted with id 0
//     would alias every sentinel comparison in the engine;
//   - an id that is still LIVE: begin_execution_window's create_for_query
//     throws "already registered" on a duplicate, killing an innocent query
//     ~4 billion windows into a long-running deployment.
//
// SiriusContext::allocate_window_id() guards both. The unit scenarios drive it
// directly on a bare context (no GPU); the end-to-end scenario forces the
// counter to the wrap on a live engine and proves real queries keep executing.

#include "query_id.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <sirius_context.hpp>
#include <utils/concurrent_test_utils.hpp>

#include <cstdint>
#include <string>

namespace {

constexpr std::uint32_t kMaxRaw = 0xFFFF'FFFFU;

}  // namespace

TEST_CASE("query-id wrap: the 0 sentinel is never minted",
          "[concurrency][query_id_wrap][isolated_context]")
{
  duckdb::SiriusContext ctx;  // bare: the allocator touches only the counter + registries

  // Counter at the last raw value: the next fetch_add computes 0xFFFFFFFF+1 == 0,
  // which the allocator must skip in favour of 1.
  ctx.set_next_window_id_for_test(kMaxRaw);
  const auto id = ctx.allocate_window_id();
  REQUIRE(sirius::value_of(id) == 1);
  // Exactly one extra increment: the skipped sentinel consumed raw 0, nothing else.
  REQUIRE(ctx.next_window_id_for_test() == 1);
}

TEST_CASE("query-id wrap: ids live in the lifecycle registry are skipped",
          "[concurrency][query_id_wrap][isolated_context]")
{
  duckdb::SiriusContext ctx;
  auto& lifecycle = ctx.get_query_lifecycle_registry();

  SECTION("a single live id is skipped")
  {
    lifecycle.open_query(sirius::make_query_id(5));
    ctx.set_next_window_id_for_test(4);
    REQUIRE(sirius::value_of(ctx.allocate_window_id()) == 6);
    lifecycle.close(sirius::make_query_id(5));
  }

  SECTION("a quiescing id is still live and skipped")
  {
    lifecycle.open_query(sirius::make_query_id(7));
    lifecycle.quiesce(sirius::make_query_id(7));
    ctx.set_next_window_id_for_test(6);
    REQUIRE(sirius::value_of(ctx.allocate_window_id()) == 8);
    lifecycle.close(sirius::make_query_id(7));
  }

  SECTION("a run of consecutive live ids is skipped in one allocation")
  {
    for (std::uint32_t raw = 10; raw < 18; ++raw) {
      lifecycle.open_query(sirius::make_query_id(raw));
    }
    ctx.set_next_window_id_for_test(9);
    REQUIRE(sirius::value_of(ctx.allocate_window_id()) == 18);
    for (std::uint32_t raw = 10; raw < 18; ++raw) {
      lifecycle.close(sirius::make_query_id(raw));
    }
  }

  SECTION("sentinel and live id combined at the wrap")
  {
    // Wrap lands on 0 (sentinel), then 1 (live), then 2 (free).
    lifecycle.open_query(sirius::make_query_id(1));
    ctx.set_next_window_id_for_test(kMaxRaw);
    REQUIRE(sirius::value_of(ctx.allocate_window_id()) == 2);
    lifecycle.close(sirius::make_query_id(1));
  }

  // A closed id is reusable again — the registry holds no tombstones.
  ctx.set_next_window_id_for_test(4);
  REQUIRE(sirius::value_of(ctx.allocate_window_id()) == 5);
}

TEST_CASE("query-id wrap: the engine keeps executing across the 2^32 wrap",
          "[concurrency][query_id_wrap][isolated_context]")
{
  using namespace sirius::test::concurrent;

  env_options opt;
  opt.rows                   = 500'000;  // light: this scenario proves liveness, not pressure
  opt.max_concurrent_queries = 2;
  adversarial_env env(opt);

  auto& ctx = *env.sirius_ctx;

  // Pin a fake live query on the first post-wrap raw values so the wrap has to
  // clear BOTH hazards: the 0 sentinel and a collision with a live query.
  auto& lifecycle = ctx.get_query_lifecycle_registry();
  lifecycle.open_query(sirius::make_query_id(1));
  // The repository registry is the duplicate check that actually throws in
  // production; occupy raw 2 there so the second skip path is exercised
  // end-to-end as well.
  auto& repositories = ctx.get_data_repository_registry();
  (void)repositories.create_for_query(sirius::make_query_id(2));

  ctx.set_next_window_id_for_test(kMaxRaw - 1);

  // Two queries straddle the wrap: the first mints 0xFFFFFFFF, the second
  // wraps and must skip 0 (sentinel), 1 (lifecycle-live) and 2 (repository-
  // live). Then a few more prove the engine is still healthy past the wrap.
  duckdb::Connection con(*env.db);
  for (int i = 0; i < 4; ++i) {
    const auto& sql = env.shapes[static_cast<std::size_t>(i) % env.shapes.size()];
    auto result     = con.Query(sql);
    REQUIRE_FALSE(result->HasError());
    REQUIRE(materialize(*result) == env.reference[static_cast<std::size_t>(i) % env.shapes.size()]);
  }

  // The counter wrapped (it sits far below its pre-wrap value) and settled
  // past the occupied ids — so id 0 was never minted and neither live id was
  // reused, or the queries above would have thrown "already registered".
  const auto counter_after = ctx.next_window_id_for_test();
  REQUIRE(counter_after >= 3);
  REQUIRE(counter_after < 1000);

  // The fake live entries were untouched by the wrap traffic.
  REQUIRE(lifecycle.state(sirius::make_query_id(1)).has_value());
  REQUIRE(repositories.get(sirius::make_query_id(2)) != nullptr);

  (void)repositories.erase(sirius::make_query_id(2));
  lifecycle.close(sirius::make_query_id(1));
}
