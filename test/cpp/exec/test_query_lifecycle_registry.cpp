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

#include "catch.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "query_id.hpp"

#include <atomic>
#include <thread>
#include <vector>

using sirius::make_query_id;
using sirius::exec::query_lifecycle_registry;
using sirius::exec::query_lifecycle_state;

TEST_CASE("an unknown query accepts work", "[query_lifecycle_gate][concurrency]")
{
  // Deliberate direction: a missed open_query() must not silently stop a query from scheduling
  // anything, which would present as a hang. Components with no registry bound behave as before.
  query_lifecycle_registry registry;
  REQUIRE(registry.accepts_work(make_query_id(7)));
  REQUIRE_FALSE(registry.state(make_query_id(7)).has_value());
  REQUIRE(registry.size() == 0);
}

TEST_CASE("open -> quiescing -> closed", "[query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry registry;
  const auto q = make_query_id(1);

  registry.open_query(q);
  REQUIRE(registry.accepts_work(q));
  REQUIRE(registry.state(q) == query_lifecycle_state::open);
  REQUIRE(registry.size() == 1);

  registry.quiesce(q);
  REQUIRE_FALSE(registry.accepts_work(q));
  REQUIRE(registry.state(q) == query_lifecycle_state::quiescing);

  registry.close(q);
  REQUIRE_FALSE(registry.state(q).has_value());
  REQUIRE(registry.size() == 0);
}

TEST_CASE("quiescing one query leaves every other query accepting work",
          "[query_lifecycle_gate][concurrency]")
{
  // The whole point of the gate: teardown of one query must not refuse another query's work, the
  // way interrupting a shared queue does.
  query_lifecycle_registry registry;
  const auto a = make_query_id(1);
  const auto b = make_query_id(2);
  const auto c = make_query_id(3);

  registry.open_query(a);
  registry.open_query(b);
  registry.open_query(c);

  registry.quiesce(b);

  REQUIRE(registry.accepts_work(a));
  REQUIRE_FALSE(registry.accepts_work(b));
  REQUIRE(registry.accepts_work(c));

  registry.close(b);
  REQUIRE(registry.accepts_work(a));
  REQUIRE(registry.accepts_work(c));
  REQUIRE(registry.size() == 2);
}

TEST_CASE("quiesce and close are idempotent and safe on unknown queries",
          "[query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry registry;
  const auto q       = make_query_id(1);
  const auto unknown = make_query_id(99);

  // Cleanup can run twice on a failed query (finish() then the destructor backstop), and the
  // best-effort teardown path quiesces a query that may never have opened.
  REQUIRE_NOTHROW(registry.quiesce(unknown));
  REQUIRE_NOTHROW(registry.close(unknown));
  REQUIRE(registry.accepts_work(unknown));

  registry.open_query(q);
  registry.quiesce(q);
  registry.quiesce(q);
  REQUIRE(registry.state(q) == query_lifecycle_state::quiescing);
  registry.close(q);
  REQUIRE_NOTHROW(registry.close(q));
  REQUIRE(registry.size() == 0);
}

TEST_CASE("a quiesce is visible to every reader once it returns",
          "[query_lifecycle_gate][concurrency]")
{
  // Producers call accepts_work() from pool workers while the cleanup thread quiesces. Readers
  // may legitimately observe either state before the quiesce lands, but once it has returned no
  // reader may still see the query as accepting work.
  query_lifecycle_registry registry;
  const auto gated  = make_query_id(1);
  const auto other  = make_query_id(2);
  constexpr int kNr = 4;

  registry.open_query(gated);
  registry.open_query(other);

  std::atomic<bool> stop{false};
  std::atomic<bool> quiesced{false};
  std::atomic<int> accepted_after_quiesce{0};
  std::atomic<int> other_refused{0};

  std::vector<std::thread> readers;
  readers.reserve(kNr);
  for (int i = 0; i < kNr; ++i) {
    readers.emplace_back([&] {
      while (!stop.load(std::memory_order_acquire)) {
        const bool seen_quiesced = quiesced.load(std::memory_order_acquire);
        if (registry.accepts_work(gated) && seen_quiesced) {
          accepted_after_quiesce.fetch_add(1, std::memory_order_relaxed);
        }
        // The unrelated query must never be refused, no matter what the gated one is doing.
        if (!registry.accepts_work(other)) {
          other_refused.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  registry.quiesce(gated);
  quiesced.store(true, std::memory_order_release);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  stop.store(true, std::memory_order_release);
  for (auto& t : readers) {
    t.join();
  }

  REQUIRE(accepted_after_quiesce.load() == 0);
  REQUIRE(other_refused.load() == 0);
}
