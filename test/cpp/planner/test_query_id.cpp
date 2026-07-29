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

#include "planner/query.hpp"
#include "query_id.hpp"
#include "utils/telemetry_utils.hpp"

#include <catch.hpp>

#include <cstdint>
#include <format>
#include <limits>

using sirius::make_query_id;
using sirius::query_id_t;
using sirius::query_priority_bits;
using sirius::value_of;

TEST_CASE("query_id: round-trips through its underlying value", "[query_id]")
{
  CHECK(value_of(make_query_id(0)) == 0U);
  CHECK(value_of(make_query_id(42)) == 42U);
  CHECK(value_of(make_query_id(std::numeric_limits<std::uint32_t>::max())) ==
        std::numeric_limits<std::uint32_t>::max());
}

TEST_CASE("query_id: orders by underlying value", "[query_id]")
{
  // std::map in the repository registry relies on this ordering for its ascending get_all().
  CHECK(make_query_id(1) < make_query_id(2));
  CHECK(make_query_id(2) > make_query_id(1));
  CHECK(make_query_id(7) == make_query_id(7));
}

TEST_CASE("query_id: formats as its numeric value", "[query_id]")
{
  // Logging is std::format-based and C++20 has no built-in enum formatting, so query_id_t
  // carries its own formatter; without it every log site would need a cast.
  CHECK(std::format("{}", make_query_id(1234)) == "1234");
}

TEST_CASE("query_id: priority bits place the id above the pipeline rank", "[query_id]")
{
  // The low 32 bits are left free for the within-query pipeline rank.
  CHECK(query_priority_bits(make_query_id(0)) == 0);
  CHECK(query_priority_bits(make_query_id(1)) == (std::int64_t{1} << 32));
  CHECK(query_priority_bits(make_query_id(3)) == (std::int64_t{3} << 32));
}

TEST_CASE("query_id: an earlier query always sorts before a later one", "[query_id]")
{
  // The priority queue pops the lowest value first, so every task of query N (even its last
  // pipeline) must sort ahead of query N+1's first pipeline.
  constexpr std::int64_t max_rank = 0xFFFF'FFFF;
  const auto earlier_last         = query_priority_bits(make_query_id(5)) | max_rank;
  const auto later_first          = query_priority_bits(make_query_id(6));
  CHECK(earlier_last < later_first);
}

TEST_CASE("query_id: priority bits stay non-negative with bit 31 set", "[query_id]")
{
  // Regression: queue_priority is SIGNED. Before masking, an id >= 2^31 shifted into the sign
  // bit, making the packed priority negative and inverting the "earlier query first" ordering.
  const auto high_id = make_query_id(0x8000'0000U);
  CHECK(query_priority_bits(high_id) >= 0);

  const auto higher_id = make_query_id(0xFFFF'FFFFU);
  CHECK(query_priority_bits(higher_id) >= 0);

  // The mask is what makes this hold: bit 31 is dropped rather than shifted into the sign bit.
  CHECK(query_priority_bits(high_id) == query_priority_bits(make_query_id(0)));
}

TEST_CASE("planner::query reports the id it was constructed with", "[query_id]")
{
  // The query no longer mints its own id from a private counter; it must carry the execution
  // window's id so repositories, scheduling and window logs all agree.
  auto tctx           = sirius::test::make_test_telemetry_context();
  const auto query_id = make_query_id(9876);
  sirius::telemetry::query_telemetry_info tinfo{
    tctx->engine_id(), tctx->worker_id(), query_id};

  sirius::planner::query q(
    duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>>{},
    tctx->context(),
    query_id,
    tinfo);

  CHECK(q.query_id() == query_id);
}

TEST_CASE("planner::query ids are not drawn from a shared counter", "[query_id]")
{
  // Two queries built with the same id keep it: nothing auto-increments behind the caller's
  // back, which is what allowed a second, independent query-id counter to exist.
  auto tctx           = sirius::test::make_test_telemetry_context();
  const auto query_id = make_query_id(11);
  sirius::telemetry::query_telemetry_info tinfo{
    tctx->engine_id(), tctx->worker_id(), query_id};

  sirius::planner::query first(
    duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>>{},
    tctx->context(),
    query_id,
    tinfo);
  sirius::planner::query second(
    duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>>{},
    tctx->context(),
    query_id,
    tinfo);

  CHECK(first.query_id() == query_id);
  CHECK(second.query_id() == query_id);
}
