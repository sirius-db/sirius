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

// [late_mat][defer] — which bundles are worth deferring. No GPU.
//
// The cases are the measured ones, because the thresholds only mean anything
// against the shapes that produced them: a wide bundle that won, a narrow
// dimension ride that cost time, and the arbitration that a first-wins policy
// got wrong by letting a small ride occupy the slot the big one needed.

#include <catch.hpp>
#include <late_mat/defer_policy.hpp>

#include <string>
#include <vector>

using sirius::late_mat::choose_deferrals;
using sirius::late_mat::defer_candidate;
using sirius::late_mat::defer_column;
using sirius::late_mat::defer_policy;
using sirius::late_mat::defer_refusal;

namespace {

defer_candidate bundle(std::string slot, std::vector<std::int64_t> widths, int boundaries)
{
  defer_candidate c;
  c.slot            = std::move(slot);
  c.boundaries      = boundaries;
  std::uint32_t pos = 0;
  for (auto const w : widths) {
    c.columns.push_back(defer_column{pos++, w});
  }
  return c;
}

}  // namespace

TEST_CASE("a wide bundle over a long ride installs", "[late_mat][defer]")
{
  // The shape that won: a customer-class bundle of ~154 B over 6 crossings.
  auto const out = choose_deferrals({bundle("agg", {25, 40, 89}, 6)});
  REQUIRE(out.size() == 1);
  REQUIRE(out[0].installed());
  REQUIRE(out[0].net_value_bytes == 25 + 40 + 89 - 8);
}

TEST_CASE("a narrow dimension ride is refused", "[late_mat][defer]")
{
  // The shape that cost +61 ms: an 11-25 B name column. It saves almost
  // nothing per row and still pays to materialize.
  auto const out = choose_deferrals({bundle("agg", {25}, 8)});
  REQUIRE_FALSE(out[0].installed());
  REQUIRE(out[0].refusal == defer_refusal::too_little_value);
}

TEST_CASE("a wide bundle over a short ride is refused", "[late_mat][defer]")
{
  // Width alone is not the case for deferring — the ride has to be long
  // enough that not carrying the values repays materializing them.
  auto const out = choose_deferrals({bundle("agg", {200}, 2)});
  REQUIRE_FALSE(out[0].installed());
  REQUIRE(out[0].refusal == defer_refusal::too_short_a_ride);
}

TEST_CASE("the value floor is net of the rowid the ride carries instead", "[late_mat][defer]")
{
  defer_policy const policy;  // 32 B floor, 8 B rowid
  // 39 B of values is 31 B net — just under.
  REQUIRE(choose_deferrals({bundle("agg", {39}, 6)}, policy)[0].refusal ==
          defer_refusal::too_little_value);
  // 41 B is 33 B net — just over. The rowid is the difference, not a rounding.
  REQUIRE(choose_deferrals({bundle("agg", {41}, 6)}, policy)[0].installed());
}

TEST_CASE("a wider bundle evicts a narrower one holding the same slot", "[late_mat][defer]")
{
  // The bug first-wins produced: a small ride arrives first and locks out the
  // bundle the query actually needed.
  auto const out = choose_deferrals({
    bundle("agg", {45}, 6),      // arrives first, would have won on order
    bundle("agg", {60, 95}, 6),  // wider, and the one that matters
  });
  REQUIRE(out[0].refusal == defer_refusal::evicted);
  REQUIRE(out[1].installed());
}

TEST_CASE("bundles for different slots do not compete", "[late_mat][defer]")
{
  auto const out = choose_deferrals({bundle("join", {60, 95}, 6), bundle("agg", {45, 40}, 5)});
  REQUIRE(out[0].installed());
  REQUIRE(out[1].installed());
}

TEST_CASE("a candidate that could never install does not evict one that could", "[late_mat][defer]")
{
  // A huge bundle over a ride too short to qualify must not take the slot from
  // a qualifying one — eviction is between candidates that both passed.
  auto const out = choose_deferrals({bundle("agg", {500}, 1), bundle("agg", {60, 95}, 6)});
  REQUIRE(out[0].refusal == defer_refusal::too_short_a_ride);
  REQUIRE(out[1].installed());
}

TEST_CASE("every candidate gets an outcome, refusals included", "[late_mat][defer]")
{
  // A deferral that silently did not happen looks exactly like one that did
  // nothing, which is the whole question when a measurement disappoints.
  auto const out = choose_deferrals({
    bundle("a", {}, 9),
    bundle("b", {12}, 9),
    bundle("c", {200}, 1),
    bundle("d", {60, 95}, 6),
  });
  REQUIRE(out.size() == 4);
  REQUIRE(out[0].refusal == defer_refusal::no_columns);
  REQUIRE(out[1].refusal == defer_refusal::too_little_value);
  REQUIRE(out[2].refusal == defer_refusal::too_short_a_ride);
  REQUIRE(out[3].installed());
  for (auto const& o : out) {
    REQUIRE(std::string(sirius::late_mat::describe(o.refusal)).size() > 0);
  }
}
