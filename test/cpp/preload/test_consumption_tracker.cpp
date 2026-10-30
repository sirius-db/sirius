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
#include "preload/consumption_tracker.hpp"
#include "query_id.hpp"

#include <cstdint>
#include <tuple>

using sirius::make_query_id;
using sirius::preload::consumption_tracker;
using sirius::preload::input_readiness;
using state = consumption_tracker::batch_state;

namespace {

/// Enables the process-wide tracker for one test case and clears it afterwards.
struct tracker_scope {
  consumption_tracker& t = consumption_tracker::instance();
  tracker_scope() { t.enable(); }
  ~tracker_scope() { t.disable(); }
};

/// Prepares @p batch_id the way prepare_for_processing does.
void prepare(consumption_tracker& t,
             std::uint64_t batch_id,
             bool upgraded,
             std::uint64_t wait_ns    = 0,
             std::uint64_t prepare_ns = 0)
{
  auto const before = t.begin_input(batch_id);
  REQUIRE(before.has_value());
  t.end_input(batch_id, *before, upgraded, wait_ns, prepare_ns);
}

}  // namespace

TEST_CASE("consumption_tracker ignores everything while disabled", "[preload]")
{
  auto& t = consumption_tracker::instance();
  REQUIRE_FALSE(t.enabled());
  t.note_downgraded(make_query_id(1), 7);
  CHECK_FALSE(t.begin_input(7).has_value());
  auto const c = t.take_query(make_query_id(1));
  CHECK(c.ready + c.waited + c.upgraded == 0);
}

TEST_CASE("consumption_tracker classifies how a task found each downgraded input", "[preload]")
{
  tracker_scope s;
  auto& t      = s.t;
  auto const q = make_query_id(3);

  // Promoted before the task asked: ready.
  t.note_downgraded(q, 1);
  t.note_promotion_started(1);
  t.note_promotion_finished(1, true);
  CHECK(t.begin_input(1) == state::promoted);
  prepare(t, 1, false);

  // A promotion was copying it: waited, with the lock wait.
  t.note_downgraded(q, 2);
  t.note_promotion_started(2);
  CHECK(t.begin_input(2) == state::promoting);
  prepare(t, 2, false, 5'000);
  t.note_promotion_finished(2, true);  // after the task counted it: no effect

  // A failed promotion leaves it on HOST, and the task upgrades it itself.
  t.note_downgraded(q, 3);
  t.note_promotion_started(3);
  t.note_promotion_finished(3, false);
  CHECK(t.begin_input(3) == state::downgraded);
  prepare(t, 3, true, 0, 7'000);

  auto const c = t.take_query(q);
  CHECK(c.ready == 1);
  CHECK(c.waited == 1);
  CHECK(c.waited_ns == 5'000);
  CHECK(c.upgraded == 1);
  CHECK(c.upgraded_ns == 7'000);
}

TEST_CASE("consumption_tracker counts an input once per downgrade", "[preload]")
{
  tracker_scope s;
  auto& t      = s.t;
  auto const q = make_query_id(4);

  t.note_downgraded(q, 9);
  auto const before = t.begin_input(9);
  REQUIRE(before.has_value());
  t.end_input(9, *before, true, 0, 1);
  t.end_input(9, *before, true, 0, 1);  // a second reader of the same batch
  CHECK_FALSE(t.begin_input(9).has_value());

  // Downgraded again: tracked again.
  t.note_downgraded(q, 9);
  prepare(t, 9, true);

  CHECK(t.take_query(q).upgraded == 2);
}

TEST_CASE("consumption_tracker forgets only the query taken", "[preload]")
{
  tracker_scope s;
  auto& t      = s.t;
  auto const a = make_query_id(5);
  auto const b = make_query_id(6);

  t.note_downgraded(a, 10);
  t.note_downgraded(b, 11);
  prepare(t, 10, false);

  CHECK(t.take_query(a).waited == 1);
  CHECK_FALSE(t.begin_input(10).has_value());
  CHECK(t.begin_input(11) == state::downgraded);
  CHECK(t.take_query(a).waited == 0);  // taken once
  std::ignore = t.take_query(b);
  CHECK_FALSE(t.begin_input(11).has_value());
}

TEST_CASE("consumption_tracker classify", "[preload]")
{
  CHECK(consumption_tracker::classify(state::promoted, false) == input_readiness::ready);
  CHECK(consumption_tracker::classify(state::promoting, false) == input_readiness::waited);
  CHECK(consumption_tracker::classify(state::downgraded, false) == input_readiness::waited);
  CHECK(consumption_tracker::classify(state::promoted, true) == input_readiness::upgraded);
  CHECK(consumption_tracker::classify(state::downgraded, true) == input_readiness::upgraded);
}
