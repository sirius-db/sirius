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
#include "exec/multi_index_priority_queue.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

using namespace sirius::exec;
using sirius::op::SiriusPhysicalOperatorType;

namespace {

struct payload {
  int id;
  index_keys keys;
};

// A level == one priority == one pipeline. Same-priority tasks therefore share a
// pipeline (and must also share an operator, which the scheduler guarantees);
// different priorities are different pipelines. Device may vary within a level.
index_keys keys_of(queue_priority priority,
                   SiriusPhysicalOperatorType op = SiriusPhysicalOperatorType::FILTER,
                   query_key query               = 0,
                   device_key device             = no_preferred_device)
{
  return index_keys{priority, op, query, device};
}

std::unique_ptr<payload> task(int id, const index_keys& keys)
{
  return std::make_unique<payload>(payload{id, keys});
}

multi_index_priority_queue<payload>::key_extractor by_keys()
{
  return [](const payload& p) { return p.keys; };
}

template <typename Queue>
std::vector<int> drain_front(Queue& q)
{
  std::vector<int> ids;
  while (auto t = q.try_pop()) {
    ids.push_back((*t)->id);
  }
  return ids;
}

}  // namespace

// =============================================================================
// Global priority ordering
// =============================================================================

TEST_CASE("multi_index pops lowest priority first", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(10, keys_of(1)));
  q.push(task(11, keys_of(5)));
  q.push(task(12, keys_of(3)));
  q.push(task(13, keys_of(5)));  // same priority 5 -> same pipeline -> FIFO after id 11
  q.push(task(14, keys_of(0)));

  REQUIRE(drain_front(q) == std::vector<int>{14, 10, 12, 11, 13});
}

TEST_CASE("multi_index equal priorities preserve FIFO order", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  for (int i = 0; i < 6; ++i) {
    q.push(task(i, keys_of(7)));  // one pipeline, six FIFO tasks
  }
  REQUIRE(drain_front(q) == std::vector<int>{0, 1, 2, 3, 4, 5});
}

TEST_CASE("multi_index pop and pop_back walk from both ends", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(10, keys_of(1)));
  q.push(task(11, keys_of(5)));
  q.push(task(12, keys_of(3)));

  REQUIRE(q.pop()->id == 10);       // priority 1
  REQUIRE(q.pop_back()->id == 11);  // priority 5
  REQUIRE(q.size() == 1);
  REQUIRE(q.pop()->id == 12);
  REQUIRE(q.empty());
}

TEST_CASE("multi_index supports negative priorities", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(10, keys_of(-5)));
  q.push(task(11, keys_of(0)));
  q.push(task(12, keys_of(-1)));
  REQUIRE(drain_front(q) == std::vector<int>{10, 12, 11});
}

// =============================================================================
// try_pop / try_pop_back on empty
// =============================================================================

TEST_CASE("multi_index try_pop variants return nullopt when empty", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  REQUIRE_FALSE(q.try_pop().has_value());
  REQUIRE_FALSE(q.try_pop_back().has_value());
  REQUIRE_FALSE(q.try_pop_from(operator_index{SiriusPhysicalOperatorType::FILTER}).has_value());
  REQUIRE_FALSE(q.try_pop_from(query_index{0}).has_value());
  REQUIRE_FALSE(q.try_pop_from(gpu_index{0}).has_value());
  REQUIRE(q.empty());
  REQUIRE(q.size() == 0);
}

// =============================================================================
// Per-bucket introspection getters
// =============================================================================

TEST_CASE("multi_index reports per-bucket task counts", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // Three pipelines (priorities 1, 2, 3); operator is constant within each.
  q.push(task(1, keys_of(1, SiriusPhysicalOperatorType::FILTER, /*query=*/100)));
  q.push(task(2, keys_of(2, SiriusPhysicalOperatorType::FILTER, /*query=*/200)));
  q.push(task(3, keys_of(3, SiriusPhysicalOperatorType::PROJECTION, /*query=*/100)));

  REQUIRE(q.operator_bucket_count() == 2);  // FILTER (2 levels), PROJECTION (1)
  REQUIRE(q.query_bucket_count() == 2);     // 100 (2 levels), 200 (1)
  REQUIRE(q.device_bucket_count() == 1);    // all no-preference (-1)

  const auto op_sizes = q.operator_bucket_sizes();
  REQUIRE(op_sizes.at(SiriusPhysicalOperatorType::FILTER) == 2);
  REQUIRE(op_sizes.at(SiriusPhysicalOperatorType::PROJECTION) == 1);

  const auto query_sizes = q.query_bucket_sizes();
  REQUIRE(query_sizes.at(100) == 2);
  REQUIRE(query_sizes.at(200) == 1);

  // Draining the PROJECTION pipeline (priority 3) prunes it from every dimension.
  REQUIRE(q.try_pop_from(operator_index{SiriusPhysicalOperatorType::PROJECTION}).has_value());
  REQUIRE(q.operator_bucket_count() == 1);
  REQUIRE(q.query_bucket_sizes().at(100) == 1);  // only the priority-1 pipeline left in query 100
  REQUIRE(q.operator_bucket_sizes().count(SiriusPhysicalOperatorType::PROJECTION) == 0);
}

// =============================================================================
// Operator index (operator spans multiple pipelines / levels)
// =============================================================================

TEST_CASE("multi_index try_pop_from operator selects lowest-priority match",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // FILTER pipelines at priorities 4, 1, 9; HASH_JOIN pipelines at 0, 2.
  q.push(task(10, keys_of(4, SiriusPhysicalOperatorType::FILTER)));
  q.push(task(20, keys_of(0, SiriusPhysicalOperatorType::HASH_JOIN)));
  q.push(task(11, keys_of(1, SiriusPhysicalOperatorType::FILTER)));
  q.push(task(21, keys_of(2, SiriusPhysicalOperatorType::HASH_JOIN)));
  q.push(task(12, keys_of(9, SiriusPhysicalOperatorType::FILTER)));

  const operator_index filter{SiriusPhysicalOperatorType::FILTER};
  REQUIRE(q.size(filter) == 3);

  REQUIRE(q.try_pop_from(filter).value()->id == 11);       // lowest-priority FILTER (p=1)
  REQUIRE(q.try_pop_back_from(filter).value()->id == 12);  // highest-priority FILTER (p=9)

  REQUIRE(q.size(filter) == 1);  // id 10 (p=4) remains
  REQUIRE(q.size(operator_index{SiriusPhysicalOperatorType::HASH_JOIN}) == 2);
  REQUIRE(q.size() == 3);
}

TEST_CASE("multi_index try_pop_from returns nullopt for an absent operator",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(10, keys_of(1, SiriusPhysicalOperatorType::FILTER)));
  REQUIRE_FALSE(q.try_pop_from(operator_index{SiriusPhysicalOperatorType::HASH_JOIN}).has_value());
  REQUIRE(q.size() == 1);
}

// =============================================================================
// Query-index popping (a query spans several priority levels)
// =============================================================================

TEST_CASE("multi_index query index spans its levels", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // Query 7 spans two levels: priority 5 (ids 1, 4 -- FIFO) and priority 9 (id 3).
  // Query 8 is one level at priority 2 (id 2).
  q.push(task(1, keys_of(5, SiriusPhysicalOperatorType::FILTER, /*query=*/7)));
  q.push(task(4, keys_of(5, SiriusPhysicalOperatorType::FILTER, /*query=*/7)));
  q.push(task(3, keys_of(9, SiriusPhysicalOperatorType::FILTER, /*query=*/7)));
  q.push(task(2, keys_of(2, SiriusPhysicalOperatorType::PROJECTION, /*query=*/8)));

  REQUIRE(q.size(query_index{7}) == 3);
  REQUIRE(q.size(query_index{8}) == 1);

  // Query 7 scans its levels in priority order: priority 5 first (FIFO id 1, then
  // id 4), and the back is the highest level, priority 9 (id 3).
  REQUIRE(q.try_pop_from(query_index{7}).value()->id == 1);
  REQUIRE(q.try_pop_from(query_index{7}).value()->id == 4);
  REQUIRE(q.try_pop_back_from(query_index{7}).value()->id == 3);

  REQUIRE(q.size() == 1);
  REQUIRE(q.pop()->id == 2);
}

// =============================================================================
// Per-query drain
// =============================================================================

TEST_CASE("multi_index drain(query_index) leaves the queue open for later pushes",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(1, keys_of(5, SiriusPhysicalOperatorType::FILTER, /*query=*/7)));

  q.drain(query_index{7});

  // Unlike interrupt(), a per-query drain must not close the queue: other queries keep
  // producing through it, and the drained query id may even be reused by a later push.
  REQUIRE(q.is_open());
  q.push(task(2, keys_of(5, SiriusPhysicalOperatorType::FILTER, /*query=*/8)));
  REQUIRE(q.size() == 1);
  REQUIRE(q.pop()->id == 2);
}

// =============================================================================
// Cross-index consistency
// =============================================================================

TEST_CASE("multi_index removal keeps every index consistent", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // One pipeline (priority 3, query 7, FILTER); two tasks preferring different GPUs.
  q.push(task(1, keys_of(3, SiriusPhysicalOperatorType::FILTER, /*query=*/7, /*device=*/0)));
  q.push(task(2, keys_of(3, SiriusPhysicalOperatorType::FILTER, /*query=*/7, /*device=*/1)));

  // Pop id 1 via the device index; it must vanish from level/operator/query too.
  REQUIRE(q.try_pop_from(gpu_index{0}).value()->id == 1);
  REQUIRE(q.size() == 1);
  REQUIRE(q.size(operator_index{SiriusPhysicalOperatorType::FILTER}) == 1);
  REQUIRE(q.size(query_index{7}) == 1);
  REQUIRE(q.size(gpu_index{0}) == 0);
  REQUIRE(q.size(gpu_index{1}) == 1);

  // Draining the last task empties and prunes every index.
  REQUIRE(q.try_pop_back().value()->id == 2);
  REQUIRE(q.empty());
  REQUIRE(q.operator_bucket_count() == 0);
  REQUIRE(q.query_bucket_count() == 0);
  REQUIRE(q.device_bucket_count() == 0);
}

TEST_CASE("multi_index refills a level after it drains to empty", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  const operator_index filter{SiriusPhysicalOperatorType::FILTER};

  q.push(task(1, keys_of(1, SiriusPhysicalOperatorType::FILTER)));
  REQUIRE(q.try_pop_from(filter).has_value());
  REQUIRE(q.size(filter) == 0);
  REQUIRE(q.operator_bucket_count() == 0);  // the level was pruned

  // Re-pushing the same priority recreates the pruned level cleanly.
  q.push(task(2, keys_of(1, SiriusPhysicalOperatorType::FILTER)));
  REQUIRE(q.size(filter) == 1);
  REQUIRE(q.try_pop_from(filter).value()->id == 2);
}

// =============================================================================
// Device (gpu) index -- preferred-device routing, -1 == no preference
// =============================================================================

TEST_CASE("multi_index gpu_index pops per preferred device across levels",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // Device 0 is preferred by tasks in two pipelines (priorities 1 and 5); device 1
  // by one pipeline (priority 2).
  q.push(task(2, keys_of(1, SiriusPhysicalOperatorType::FILTER, 0, /*device=*/0)));
  q.push(task(1, keys_of(5, SiriusPhysicalOperatorType::FILTER, 0, /*device=*/0)));
  q.push(task(3, keys_of(2, SiriusPhysicalOperatorType::FILTER, 0, /*device=*/1)));

  REQUIRE(q.size(gpu_index{0}) == 2);
  REQUIRE(q.size(gpu_index{1}) == 1);
  REQUIRE(q.device_bucket_count() == 2);

  REQUIRE(q.try_pop_from(gpu_index{0}).value()->id == 2);       // lowest priority (1)
  REQUIRE(q.try_pop_back_from(gpu_index{0}).value()->id == 1);  // highest priority (5)
  REQUIRE(q.size(gpu_index{0}) == 0);

  REQUIRE(q.try_pop_from(gpu_index{1}).value()->id == 3);
  REQUIRE(q.device_bucket_count() == 0);
  REQUIRE_FALSE(q.try_pop_from(gpu_index{0}).has_value());
}

TEST_CASE("multi_index gpu_index{-1} routes tasks with no preferred device",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(10, keys_of(4)));  // device defaults to no_preferred_device (-1)
  q.push(task(11, keys_of(2)));  // device -1
  q.push(task(12, keys_of(1, SiriusPhysicalOperatorType::FILTER, 0, /*device=*/3)));

  const gpu_index no_pref{no_preferred_device};
  REQUIRE(q.size(no_pref) == 2);
  REQUIRE(q.size(gpu_index{3}) == 1);

  REQUIRE(q.try_pop_from(no_pref).value()->id == 11);  // p=2 before p=4
  REQUIRE(q.try_pop_from(no_pref).value()->id == 10);
  REQUIRE_FALSE(q.try_pop_from(no_pref).has_value());

  // The device-preferring task never leaked into the no-preference bucket.
  REQUIRE(q.size() == 1);
  REQUIRE(q.try_pop_from(gpu_index{3}).value()->id == 12);
  REQUIRE(q.device_bucket_sizes().empty());
}

TEST_CASE("multi_index device index stays consistent across other-index pops",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(1, keys_of(1, SiriusPhysicalOperatorType::HASH_JOIN, /*query=*/9, /*device=*/2)));

  // Popping via the query index must also remove the node from the device index.
  REQUIRE(q.try_pop_from(query_index{9}).value()->id == 1);
  REQUIRE(q.size(gpu_index{2}) == 0);
  REQUIRE(q.device_bucket_count() == 0);
}

// =============================================================================
// try_pop_if -- predicate-filtered pops in pop order
// =============================================================================

TEST_CASE("multi_index try_pop_if pops the lowest-priority match globally",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(1, keys_of(1)));  // odd id, priority 1
  q.push(task(2, keys_of(2)));  // even id, priority 2
  q.push(task(4, keys_of(3)));  // even id, priority 3

  // Scans in pop order (priority 1, 2, 3); first even id is 2 (at priority 2),
  // even though id 1 sits ahead globally.
  auto match = q.try_pop_if([](const payload& p) { return p.id % 2 == 0; });
  REQUIRE(match.has_value());
  REQUIRE((*match)->id == 2);
  REQUIRE(q.size() == 2);

  REQUIRE_FALSE(q.try_pop_if([](const payload&) { return false; }).has_value());
  REQUIRE(q.size() == 2);
}

TEST_CASE("multi_index try_pop_if honors FIFO within a level", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(1, keys_of(5)));  // one pipeline (priority 5), three FIFO tasks
  q.push(task(2, keys_of(5)));
  q.push(task(3, keys_of(5)));

  auto match = q.try_pop_if([](const payload& p) { return p.id >= 2; });
  REQUIRE(match.has_value());
  REQUIRE((*match)->id == 2);  // FIFO: id 2 before id 3
  REQUIRE(q.size() == 2);
}

TEST_CASE("multi_index try_pop_if scoped to an operator bucket", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(10, keys_of(1, SiriusPhysicalOperatorType::FILTER)));
  q.push(task(20, keys_of(2, SiriusPhysicalOperatorType::HASH_JOIN)));
  q.push(task(12, keys_of(3, SiriusPhysicalOperatorType::FILTER)));

  // Only FILTER levels are scanned (ascending priority); the HASH_JOIN task is
  // never considered even though it matches the predicate.
  auto match = q.try_pop_if(operator_index{SiriusPhysicalOperatorType::FILTER},
                            [](const payload& p) { return p.id != 10; });
  REQUIRE(match.has_value());
  REQUIRE((*match)->id == 12);
  REQUIRE(q.size(operator_index{SiriusPhysicalOperatorType::HASH_JOIN}) == 1);

  // No FILTER task matches -> nullopt, nothing removed.
  REQUIRE_FALSE(q.try_pop_if(operator_index{SiriusPhysicalOperatorType::FILTER},
                             [](const payload& p) { return p.id == 999; })
                  .has_value());
  REQUIRE(q.size() == 2);
}

TEST_CASE("multi_index try_pop_if scoped to a device bucket", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // Device 0 preferred at priorities 1 (id 2) and 5 (id 1); device 1 at priority 2.
  q.push(task(2, keys_of(1, SiriusPhysicalOperatorType::FILTER, 0, /*device=*/0)));
  q.push(task(1, keys_of(5, SiriusPhysicalOperatorType::FILTER, 0, /*device=*/0)));
  q.push(task(9, keys_of(2, SiriusPhysicalOperatorType::FILTER, 0, /*device=*/1)));

  // Scans device-0 buckets ascending (priority 1 then 5); id 2 fails, id 1 matches.
  auto match = q.try_pop_if(gpu_index{0}, [](const payload& p) { return p.id == 1; });
  REQUIRE(match.has_value());
  REQUIRE((*match)->id == 1);
  REQUIRE(q.size(gpu_index{0}) == 1);  // id 2 remains
  REQUIRE(q.size(gpu_index{1}) == 1);  // device 1 untouched

  REQUIRE_FALSE(q.try_pop_if(gpu_index{7}, [](const payload&) { return true; }).has_value());
}

// =============================================================================
// mutable_pop_if -- mutable predicate, directional scan (used by the downgrade path)
// =============================================================================

TEST_CASE("multi_index mutable_pop_if scans in the requested direction",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(1, keys_of(1)));
  q.push(task(2, keys_of(2)));
  q.push(task(3, keys_of(3)));

  // front_to_back: lowest priority first -> first id >= 2 is id 2 (priority 2).
  auto front = q.mutable_pop_if([](payload& p) { return p.id >= 2; }, /*front_to_back=*/true);
  REQUIRE(front.has_value());
  REQUIRE((*front)->id == 2);

  // back_to_front over {1, 3}: highest priority first -> first match is id 3 (priority 3).
  auto back = q.mutable_pop_if([](payload& p) { return p.id >= 1; }, /*front_to_back=*/false);
  REQUIRE(back.has_value());
  REQUIRE((*back)->id == 3);

  REQUIRE(q.size() == 1);
  REQUIRE_FALSE(q.mutable_pop_if([](payload&) { return false; }, true).has_value());
}

TEST_CASE("multi_index mutable_pop_if honors direction within a level",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  for (int i = 1; i <= 3; ++i) {
    q.push(task(i, keys_of(5)));  // one level (priority 5), FIFO: 1, 2, 3
  }
  REQUIRE(q.mutable_pop_if([](payload&) { return true; }, /*front_to_back=*/true).value()->id == 1);
  REQUIRE(q.mutable_pop_if([](payload&) { return true; }, /*front_to_back=*/false).value()->id ==
          3);
  REQUIRE(q.pop()->id == 2);
}

// =============================================================================
// Thread-safety: blocking pop, interrupt, drain
// =============================================================================

TEST_CASE("multi_index blocking pop wakes when a task is pushed", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());

  int popped_id = -1;
  std::thread consumer([&] {
    auto t = q.pop();  // blocks until the producer pushes
    if (t) { popped_id = t->id; }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  q.push(task(42, keys_of(1)));
  consumer.join();

  REQUIRE(popped_id == 42);
  REQUIRE(q.empty());
}

TEST_CASE("multi_index interrupt wakes a blocked pop with nullptr", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());

  bool got_null = false;
  std::thread consumer([&] {
    auto t   = q.pop();  // blocks on the empty queue
    got_null = (t == nullptr);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  REQUIRE(q.is_open());
  q.interrupt();
  consumer.join();

  REQUIRE(got_null);
  REQUIRE_FALSE(q.is_open());

  // While interrupted, push drops the task instead of enqueuing it (the shutdown
  // contract the downgrade/RAII return path relies on).
  q.push(task(99, keys_of(1)));
  REQUIRE(q.empty());
  REQUIRE(q.pop() == nullptr);  // still interrupted and empty

  // reactivate() restores normal push/pop.
  q.reactivate();
  REQUIRE(q.is_open());
  q.push(task(7, keys_of(1)));
  REQUIRE(q.pop()->id == 7);
}

TEST_CASE("multi_index drain drops all tasks and clears every index",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(1, keys_of(1, SiriusPhysicalOperatorType::FILTER, 100, 0)));
  q.push(task(2, keys_of(2, SiriusPhysicalOperatorType::PROJECTION, 200, 1)));
  REQUIRE(q.size() == 2);

  q.drain();

  REQUIRE(q.empty());
  REQUIRE(q.size() == 0);
  REQUIRE(q.operator_bucket_count() == 0);
  REQUIRE(q.query_bucket_count() == 0);
  REQUIRE(q.device_bucket_count() == 0);
  REQUIRE_FALSE(q.try_pop().has_value());

  // Still usable after a drain.
  q.push(task(3, keys_of(1)));
  REQUIRE(q.pop()->id == 3);
}

TEST_CASE("multi_index drain(query_index) drops only that query's tasks",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // Query 7: pipelines at priorities 5 (id 1) and 9 (id 3). Query 8: priority 2 (id 2).
  q.push(task(1, keys_of(5, SiriusPhysicalOperatorType::FILTER, /*query=*/7, /*device=*/2)));
  q.push(task(2, keys_of(2, SiriusPhysicalOperatorType::PROJECTION, /*query=*/8)));
  q.push(task(3, keys_of(9, SiriusPhysicalOperatorType::FILTER, /*query=*/7)));

  q.drain(query_index{7});

  // Only query 8 (id 2) survives -- in every index.
  REQUIRE(q.size() == 1);
  REQUIRE(q.query_bucket_count() == 1);
  REQUIRE(q.size(query_index{7}) == 0);
  REQUIRE(q.size(query_index{8}) == 1);
  REQUIRE(q.size(operator_index{SiriusPhysicalOperatorType::FILTER}) == 0);
  REQUIRE(q.size(operator_index{SiriusPhysicalOperatorType::PROJECTION}) == 1);
  REQUIRE(q.size(gpu_index{2}) == 0);  // id 1's device bucket cleared
  REQUIRE(q.size(gpu_index{no_preferred_device}) == 1);
  REQUIRE(q.pop()->id == 2);

  // Draining an absent query is a no-op.
  q.drain(query_index{999});
  REQUIRE(q.empty());
}

TEST_CASE("multi_index survives concurrent producers and consumers", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  constexpr int kProducers   = 4;
  constexpr int kPerProducer = 250;
  constexpr int kTotal       = kProducers * kPerProducer;

  std::atomic<int> consumed{0};
  std::atomic<bool> stop{false};

  std::vector<std::thread> consumers;
  for (int c = 0; c < 3; ++c) {
    consumers.emplace_back([&] {
      while (!stop.load() || !q.empty()) {
        if (auto t = q.try_pop()) { consumed.fetch_add(1); }
      }
    });
  }

  std::vector<std::thread> producers;
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < kPerProducer; ++i) {
        q.push(task(p * kPerProducer + i, keys_of(i % 7)));  // 7 pipelines, shared FIFO levels
      }
    });
  }

  for (auto& t : producers) {
    t.join();
  }
  stop.store(true);
  for (auto& t : consumers) {
    t.join();
  }

  REQUIRE(consumed.load() == kTotal);
  REQUIRE(q.empty());
}

// =============================================================================
// Cross-query fairness (F1): pop() and try_pop_from(gpu_index) round-robin
// across query bands so one query cannot starve another (the packing puts the
// query id in the priority's high bits, so without rotation EVERY task of the
// earliest query outranks every task of a later one).
// =============================================================================

namespace {

/// Priorities exactly as production packs them (query_priority_bits): the query
/// id in the high 32 bits, the within-query pipeline rank in the low 32.
queue_priority banded(query_key query, queue_priority rank)
{
  return (static_cast<queue_priority>(query) << 32) | rank;
}

index_keys banded_keys(query_key query,
                       queue_priority rank,
                       device_key device = no_preferred_device)
{
  return keys_of(banded(query, rank), SiriusPhysicalOperatorType::FILTER, query, device);
}

template <typename Queue>
std::vector<int> drain_pop(Queue& q)
{
  std::vector<int> ids;
  while (!q.empty()) {
    ids.push_back(q.pop()->id);
  }
  return ids;
}

}  // namespace

TEST_CASE("multi_index pop round-robins across query bands", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // Query 1 and query 2, three pipeline ranks each. Without rotation the packed
  // priorities make every query-1 task pop before any query-2 task.
  q.push(task(10, banded_keys(1, 0)));
  q.push(task(11, banded_keys(1, 1)));
  q.push(task(12, banded_keys(1, 2)));
  q.push(task(20, banded_keys(2, 0)));
  q.push(task(21, banded_keys(2, 1)));
  q.push(task(22, banded_keys(2, 2)));

  // First pop has no rotation history and takes the globally-first task (query
  // 1); after that the served query alternates while both have work.
  REQUIRE(drain_pop(q) == std::vector<int>{10, 20, 11, 21, 12, 22});
}

TEST_CASE("multi_index pop rotation preserves within-query order and FIFO",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // Query 1: rank 0 holds two FIFO tasks (10 then 11), rank 1 holds 12.
  // Query 2: rank 0 holds 20 and 21 (FIFO).
  q.push(task(10, banded_keys(1, 0)));
  q.push(task(11, banded_keys(1, 0)));
  q.push(task(12, banded_keys(1, 1)));
  q.push(task(20, banded_keys(2, 0)));
  q.push(task(21, banded_keys(2, 0)));

  // Rotation only decides WHICH query is served; within a query the order is
  // exactly the pre-fairness one (lowest rank first, FIFO within a rank).
  REQUIRE(drain_pop(q) == std::vector<int>{10, 20, 11, 21, 12});
}

TEST_CASE("multi_index pop rotation skips exhausted queries", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(10, banded_keys(1, 0)));
  q.push(task(20, banded_keys(2, 0)));
  q.push(task(21, banded_keys(2, 1)));
  q.push(task(22, banded_keys(2, 2)));

  // Query 1 runs dry after one pop; the rotation must keep serving query 2
  // rather than stalling on the empty band.
  REQUIRE(drain_pop(q) == std::vector<int>{10, 20, 21, 22});
}

TEST_CASE("multi_index single-query pop order is unchanged by fairness",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // One query, several ranks, FIFO pairs — the exact shape the whole existing
  // suite runs (max_concurrent_queries=1). Must be bit-identical to the
  // pre-fairness queue: strictly ascending priority, FIFO within a level.
  q.push(task(1, banded_keys(7, 3)));
  q.push(task(2, banded_keys(7, 0)));
  q.push(task(3, banded_keys(7, 0)));
  q.push(task(4, banded_keys(7, 2)));
  q.push(task(5, banded_keys(7, 1)));

  REQUIRE(drain_pop(q) == std::vector<int>{2, 3, 5, 4, 1});
}

TEST_CASE("multi_index try_pop_from(gpu_index) rotates across queries per device",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // Both queries stage two tasks for device 0.
  q.push(task(10, banded_keys(1, 0, /*device=*/0)));
  q.push(task(11, banded_keys(1, 1, /*device=*/0)));
  q.push(task(20, banded_keys(2, 0, /*device=*/0)));
  q.push(task(21, banded_keys(2, 1, /*device=*/0)));

  REQUIRE(q.try_pop_from(gpu_index{0}).value()->id == 10);  // no history: global first
  REQUIRE(q.try_pop_from(gpu_index{0}).value()->id == 20);  // rotate to query 2
  REQUIRE(q.try_pop_from(gpu_index{0}).value()->id == 11);  // back to query 1
  REQUIRE(q.try_pop_from(gpu_index{0}).value()->id == 21);
  REQUIRE(q.empty());
}

TEST_CASE("multi_index device rotation covers the no-preference bucket",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(10, banded_keys(1, 0)));  // no preferred device
  q.push(task(11, banded_keys(1, 1)));
  q.push(task(20, banded_keys(2, 0)));
  q.push(task(21, banded_keys(2, 1)));

  const gpu_index no_pref{no_preferred_device};
  REQUIRE(q.try_pop_from(no_pref).value()->id == 10);
  REQUIRE(q.try_pop_from(no_pref).value()->id == 20);
  REQUIRE(q.try_pop_from(no_pref).value()->id == 11);
  REQUIRE(q.try_pop_from(no_pref).value()->id == 21);
}

TEST_CASE("multi_index device rotation skips queries with nothing for that device",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // Query 1 prefers device 0; query 2 prefers device 1 only.
  q.push(task(10, banded_keys(1, 0, /*device=*/0)));
  q.push(task(11, banded_keys(1, 1, /*device=*/0)));
  q.push(task(20, banded_keys(2, 0, /*device=*/1)));

  // Device-0 pops must serve query 1 twice in priority order — never a spurious
  // nullopt because the rotation looked at query 2 first.
  REQUIRE(q.try_pop_from(gpu_index{0}).value()->id == 10);
  REQUIRE(q.try_pop_from(gpu_index{0}).value()->id == 11);
  REQUIRE_FALSE(q.try_pop_from(gpu_index{0}).has_value());
  REQUIRE(q.size(gpu_index{1}) == 1);  // query 2 untouched
  REQUIRE(q.try_pop_from(gpu_index{1}).value()->id == 20);
}

TEST_CASE("multi_index try_pop_back_from(gpu_index) keeps its globally-last semantics",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(10, banded_keys(1, 0, /*device=*/0)));
  q.push(task(20, banded_keys(2, 0, /*device=*/0)));
  q.push(task(21, banded_keys(2, 1, /*device=*/0)));

  // Back pops are downgrade-victim selection, not dispatch: no rotation. The
  // globally-highest device-0 priority is query 2's rank 1.
  REQUIRE(q.try_pop_back_from(gpu_index{0}).value()->id == 21);
  REQUIRE(q.try_pop_back_from(gpu_index{0}).value()->id == 20);
  REQUIRE(q.try_pop_back_from(gpu_index{0}).value()->id == 10);
}

TEST_CASE("multi_index rotation survives draining the cursor query", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(task(10, banded_keys(1, 0)));
  q.push(task(11, banded_keys(1, 1)));
  q.push(task(20, banded_keys(2, 0)));
  q.push(task(30, banded_keys(3, 0)));

  REQUIRE(q.pop()->id == 10);  // cursor now at query 1
  q.drain(query_index{1});     // drop the cursor query entirely

  // Rotation continues from the next live query id and wraps cleanly.
  REQUIRE(q.pop()->id == 20);
  REQUIRE(q.pop()->id == 30);
  REQUIRE(q.empty());

  // A cursor above every live id wraps to the lowest one.
  q.push(task(40, banded_keys(2, 0)));
  REQUIRE(q.pop()->id == 40);
}

TEST_CASE("multi_index rotation is safe against concurrent drain(query_index)",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  constexpr int kQueries            = 4;
  constexpr int kPerQuery           = 300;
  constexpr int kTotal              = kQueries * kPerQuery;
  constexpr query_key kDrainedQuery = 2;

  std::atomic<int> consumed{0};

  std::vector<std::thread> consumers;
  for (int c = 0; c < 2; ++c) {
    consumers.emplace_back([&] {
      // pop() blocks until interrupted at the end; rotation runs concurrently
      // with the drainer below.
      while (auto t = q.pop()) {
        consumed.fetch_add(1);
      }
    });
  }

  std::atomic<bool> producers_done{false};
  std::thread drainer([&] {
    while (!producers_done.load()) {
      q.drain(query_index{kDrainedQuery});
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    q.drain(query_index{kDrainedQuery});
  });

  std::vector<std::thread> producers;
  for (int p = 0; p < kQueries; ++p) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < kPerQuery; ++i) {
        const auto query = static_cast<query_key>(p + 1);
        q.push(task(p * kPerQuery + i, banded_keys(query, i % 5)));
      }
    });
  }

  for (auto& t : producers) {
    t.join();
  }
  producers_done.store(true);
  drainer.join();

  // Everything not consumed belongs to the drained query; sweep the leftovers
  // and unblock the consumers.
  q.drain(query_index{kDrainedQuery});
  while (!q.empty()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  q.interrupt();
  for (auto& t : consumers) {
    t.join();
  }

  // Consumed plus drained must account for every push; the drained query's
  // tasks may land on either side depending on timing.
  REQUIRE(consumed.load() <= kTotal);
  REQUIRE(consumed.load() >= kTotal - kPerQuery);
  REQUIRE(q.empty());
  REQUIRE(q.query_bucket_count() == 0);
  REQUIRE(q.device_bucket_count() == 0);
}
