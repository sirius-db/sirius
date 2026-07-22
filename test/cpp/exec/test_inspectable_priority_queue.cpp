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
#include "exec/inspectable_priority_queue.hpp"

#include <string>
#include <vector>

using namespace sirius::exec;

namespace {

struct prio_payload {
  int id;
  queue_priority priority;
  prio_payload(int i, queue_priority p) : id(i), priority(p) {}
};

// Extractor used by most tests: the item's own priority field.
inspectable_priority_queue<prio_payload>::priority_fn by_field()
{
  return [](const prio_payload& p) { return p.priority; };
}

// Drains the queue and returns the popped ids in pop order (highest priority first).
std::vector<int> drain_ids(inspectable_priority_queue<prio_payload>& q)
{
  std::vector<int> out;
  while (auto item = q.try_pop()) {
    out.push_back(item->id);
  }
  return out;
}

}  // namespace

// =============================================================================
// Default extractor reduces to FIFO
// =============================================================================

TEST_CASE("priority_queue default extractor pops in FIFO order", "[inspectable_priority_queue]")
{
  inspectable_priority_queue<int> queue;
  for (int i = 0; i < 5; ++i) {
    REQUIRE(queue.push(std::make_unique<int>(i)));
  }
  for (int i = 0; i < 5; ++i) {
    auto item = queue.try_pop();
    REQUIRE(item != nullptr);
    REQUIRE(*item == i);
  }
  REQUIRE(queue.try_pop() == nullptr);
}

// =============================================================================
// Priority ordering
// =============================================================================

TEST_CASE("priority_queue pops lowest priority first", "[inspectable_priority_queue]")
{
  inspectable_priority_queue<prio_payload> queue(by_field());
  // Pushed in arbitrary order; priorities: 10=>1, 11=>5, 12=>3, 13=>5, 14=>0
  queue.push(std::make_unique<prio_payload>(10, 1));
  queue.push(std::make_unique<prio_payload>(11, 5));
  queue.push(std::make_unique<prio_payload>(12, 3));
  queue.push(std::make_unique<prio_payload>(13, 5));
  queue.push(std::make_unique<prio_payload>(14, 0));

  // Lowest priority value first; equal priorities (11,13 both =5) keep insertion order.
  REQUIRE(drain_ids(queue) == std::vector<int>{14, 10, 12, 11, 13});
}

TEST_CASE("priority_queue equal priorities preserve FIFO order", "[inspectable_priority_queue]")
{
  inspectable_priority_queue<prio_payload> queue(by_field());
  for (int i = 0; i < 6; ++i) {
    queue.push(std::make_unique<prio_payload>(i, 7));
  }
  REQUIRE(drain_ids(queue) == std::vector<int>{0, 1, 2, 3, 4, 5});
}

TEST_CASE("priority_queue pop_back returns highest priority", "[inspectable_priority_queue]")
{
  inspectable_priority_queue<prio_payload> queue(by_field());
  queue.push(std::make_unique<prio_payload>(10, 1));
  queue.push(std::make_unique<prio_payload>(11, 5));
  queue.push(std::make_unique<prio_payload>(12, 3));

  auto highest = queue.pop_back();
  REQUIRE(highest != nullptr);
  REQUIRE(highest->id == 11);  // priority 5 is highest (last to run)

  auto lowest = queue.pop_front();
  REQUIRE(lowest != nullptr);
  REQUIRE(lowest->id == 10);  // priority 1 is lowest (first to run)
}

TEST_CASE("priority_queue supports negative priorities", "[inspectable_priority_queue]")
{
  inspectable_priority_queue<prio_payload> queue(by_field());
  queue.push(std::make_unique<prio_payload>(10, -5));
  queue.push(std::make_unique<prio_payload>(11, 0));
  queue.push(std::make_unique<prio_payload>(12, -1));
  REQUIRE(drain_ids(queue) == std::vector<int>{10, 12, 11});
}

TEST_CASE("priority_queue blocking pop returns lowest priority", "[inspectable_priority_queue]")
{
  inspectable_priority_queue<prio_payload> queue(by_field());
  queue.push(std::make_unique<prio_payload>(10, 1));
  queue.push(std::make_unique<prio_payload>(11, 9));
  queue.push(std::make_unique<prio_payload>(12, 4));

  auto a = queue.pop();
  REQUIRE(a != nullptr);
  REQUIRE(a->id == 10);
  auto b = queue.pop();
  REQUIRE(b != nullptr);
  REQUIRE(b->id == 12);
}

// =============================================================================
// pop_if / mutable_pop_if honor priority-sorted iteration
// =============================================================================

TEST_CASE("priority_queue pop_if front_to_back scans lowest priority first",
          "[inspectable_priority_queue]")
{
  inspectable_priority_queue<prio_payload> queue(by_field());
  queue.push(std::make_unique<prio_payload>(10, 1));
  queue.push(std::make_unique<prio_payload>(11, 5));
  queue.push(std::make_unique<prio_payload>(12, 5));

  // Queue order is ascending [10(1), 11(5), 12(5)]; front_to_back reaches 11 (first priority-5
  // entry, by insertion order among equals) before 12.
  auto match = queue.pop_if([](const prio_payload& p) { return p.priority == 5; }, true);
  REQUIRE(match != nullptr);
  REQUIRE(match->id == 11);
  REQUIRE(queue.size() == 2);
}

TEST_CASE("priority_queue pop_if back_to_front scans highest priority first",
          "[inspectable_priority_queue]")
{
  inspectable_priority_queue<prio_payload> queue(by_field());
  queue.push(std::make_unique<prio_payload>(10, 5));
  queue.push(std::make_unique<prio_payload>(11, 5));
  queue.push(std::make_unique<prio_payload>(12, 1));

  // Queue order is ascending [12(1), 10(5), 11(5)]; back_to_front reaches the last priority-5
  // entry (id 11) first.
  auto match = queue.pop_if([](const prio_payload& p) { return p.priority == 5; }, false);
  REQUIRE(match != nullptr);
  REQUIRE(match->id == 11);
}

TEST_CASE("priority_queue mutable_pop_if removes matching element", "[inspectable_priority_queue]")
{
  inspectable_priority_queue<prio_payload> queue(by_field());
  for (int i = 0; i < 5; ++i) {
    queue.push(std::make_unique<prio_payload>(i, i));
  }
  auto match = queue.mutable_pop_if([](prio_payload& p) { return p.id == 3; }, true);
  REQUIRE(match != nullptr);
  REQUIRE(match->id == 3);
  REQUIRE(queue.size() == 4);
}

// =============================================================================
// Lifecycle: interrupt / reactivate / drain / size / emplace
// =============================================================================

TEST_CASE("priority_queue push fails after interrupt", "[inspectable_priority_queue]")
{
  inspectable_priority_queue<prio_payload> queue(by_field());
  queue.interrupt();
  REQUIRE_FALSE(queue.is_open());
  REQUIRE_FALSE(queue.push(std::make_unique<prio_payload>(1, 1)));
  queue.reactivate();
  REQUIRE(queue.is_open());
  REQUIRE(queue.push(std::make_unique<prio_payload>(2, 1)));
}

TEST_CASE("priority_queue emplace inserts by priority", "[inspectable_priority_queue]")
{
  inspectable_priority_queue<prio_payload> queue(by_field());
  REQUIRE(queue.emplace(10, 1));
  REQUIRE(queue.emplace(11, 9));
  REQUIRE(queue.emplace(12, 4));
  REQUIRE(queue.size() == 3);
  REQUIRE(drain_ids(queue) == std::vector<int>{10, 12, 11});
}

TEST_CASE("priority_queue drain removes all items", "[inspectable_priority_queue]")
{
  inspectable_priority_queue<prio_payload> queue(by_field());
  for (int i = 0; i < 4; ++i) {
    queue.push(std::make_unique<prio_payload>(i, i));
  }
  REQUIRE(queue.size() == 4);
  queue.drain();
  REQUIRE(queue.is_empty());
}
