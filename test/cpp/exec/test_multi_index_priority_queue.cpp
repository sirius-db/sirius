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

// A task carries its own id (for identification) and the keys the extractor will
// surface to the queue.
struct payload {
  int id;
  index_keys keys;
};

index_keys make_keys(queue_priority priority,
                     SiriusPhysicalOperatorType op = SiriusPhysicalOperatorType::FILTER,
                     pipeline_key pipe             = 0,
                     query_key query               = 0)
{
  return index_keys{priority, op, pipe, query};
}

std::unique_ptr<payload> make_task(int id, const index_keys& keys)
{
  return std::make_unique<payload>(payload{id, keys});
}

// The extractor every test uses: read the keys straight off the task.
multi_index_priority_queue<payload>::key_extractor by_keys()
{
  return [](const payload& p) { return p.keys; };
}

template <typename Queue>
std::vector<int> drain_front(Queue& q)
{
  std::vector<int> ids;
  while (auto task = q.try_pop()) {
    ids.push_back((*task)->id);
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
  q.push(make_task(10, make_keys(1)));
  q.push(make_task(11, make_keys(5)));
  q.push(make_task(12, make_keys(3)));
  q.push(make_task(13, make_keys(5)));
  q.push(make_task(14, make_keys(0)));

  // Ascending priority; the two priority-5 tasks (11, 13) keep insertion order.
  REQUIRE(drain_front(q) == std::vector<int>{14, 10, 12, 11, 13});
}

TEST_CASE("multi_index equal priorities preserve FIFO order", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  for (int i = 0; i < 6; ++i) {
    q.push(make_task(i, make_keys(7)));
  }
  REQUIRE(drain_front(q) == std::vector<int>{0, 1, 2, 3, 4, 5});
}

TEST_CASE("multi_index pop and pop_back walk from both ends", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(make_task(10, make_keys(1)));
  q.push(make_task(11, make_keys(5)));
  q.push(make_task(12, make_keys(3)));

  auto lowest = q.pop();  // priority 1 -> id 10
  REQUIRE(lowest->id == 10);
  auto highest = q.pop_back();  // priority 5 -> id 11
  REQUIRE(highest->id == 11);
  REQUIRE(q.size() == 1);
  REQUIRE(q.pop()->id == 12);
  REQUIRE(q.empty());
}

TEST_CASE("multi_index supports negative priorities", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(make_task(10, make_keys(-5)));
  q.push(make_task(11, make_keys(0)));
  q.push(make_task(12, make_keys(-1)));
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
  REQUIRE_FALSE(q.try_pop_from(pipeline_index{0}).has_value());
  REQUIRE_FALSE(q.try_pop_from(query_index{0}).has_value());
  REQUIRE(q.empty());
  REQUIRE(q.size() == 0);
}

// =============================================================================
// Per-bucket introspection getters
// =============================================================================

TEST_CASE("multi_index reports per-bucket task counts", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // (priority, operator, pipeline, query)
  q.push(make_task(1, make_keys(1, SiriusPhysicalOperatorType::FILTER, 10, 100)));
  q.push(make_task(2, make_keys(2, SiriusPhysicalOperatorType::FILTER, 10, 200)));
  q.push(make_task(3, make_keys(3, SiriusPhysicalOperatorType::PROJECTION, 20, 100)));

  // Distinct-bucket counts per dimension.
  REQUIRE(q.operator_bucket_count() == 2);  // FILTER, PROJECTION
  REQUIRE(q.pipeline_bucket_count() == 2);  // 10, 20
  REQUIRE(q.query_bucket_count() == 2);     // 100, 200

  // Full per-bucket size maps.
  const auto op_sizes = q.operator_bucket_sizes();
  REQUIRE(op_sizes.size() == 2);
  REQUIRE(op_sizes.at(SiriusPhysicalOperatorType::FILTER) == 2);
  REQUIRE(op_sizes.at(SiriusPhysicalOperatorType::PROJECTION) == 1);

  const auto pipe_sizes = q.pipeline_bucket_sizes();
  REQUIRE(pipe_sizes.at(10) == 2);
  REQUIRE(pipe_sizes.at(20) == 1);

  const auto query_sizes = q.query_bucket_sizes();
  REQUIRE(query_sizes.at(100) == 2);
  REQUIRE(query_sizes.at(200) == 1);

  // Draining a bucket to empty prunes it from every getter.
  REQUIRE(q.try_pop_from(operator_index{SiriusPhysicalOperatorType::PROJECTION}).has_value());
  REQUIRE(q.operator_bucket_count() == 1);
  REQUIRE(q.pipeline_bucket_count() == 1);  // pipeline 20 is now empty and gone
  const auto op_sizes_after = q.operator_bucket_sizes();
  REQUIRE(op_sizes_after.size() == 1);
  REQUIRE(op_sizes_after.count(SiriusPhysicalOperatorType::PROJECTION) == 0);
  REQUIRE(q.pipeline_bucket_sizes().count(20) == 0);
}

// =============================================================================
// Operator-index popping
// =============================================================================

TEST_CASE("multi_index try_pop_from operator selects lowest-priority match",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // Interleave operators; FILTER tasks are 10(p=4), 11(p=1), 12(p=9).
  q.push(make_task(10, make_keys(4, SiriusPhysicalOperatorType::FILTER)));
  q.push(make_task(20, make_keys(0, SiriusPhysicalOperatorType::HASH_JOIN)));
  q.push(make_task(11, make_keys(1, SiriusPhysicalOperatorType::FILTER)));
  q.push(make_task(21, make_keys(2, SiriusPhysicalOperatorType::HASH_JOIN)));
  q.push(make_task(12, make_keys(9, SiriusPhysicalOperatorType::FILTER)));

  const operator_index filter{SiriusPhysicalOperatorType::FILTER};
  REQUIRE(q.size(filter) == 3);

  // Front of the FILTER bucket is the lowest-priority FILTER task (id 11, p=1),
  // even though a lower-priority HASH_JOIN task (id 20, p=0) sits ahead globally.
  auto front = q.try_pop_from(filter);
  REQUIRE(front.has_value());
  REQUIRE((*front)->id == 11);

  // Back of the FILTER bucket is the highest-priority FILTER task (id 12, p=9).
  auto back = q.try_pop_back_from(filter);
  REQUIRE(back.has_value());
  REQUIRE((*back)->id == 12);

  // One FILTER task (id 10) remains, HASH_JOIN untouched.
  REQUIRE(q.size(filter) == 1);
  REQUIRE(q.size(operator_index{SiriusPhysicalOperatorType::HASH_JOIN}) == 2);
  REQUIRE(q.size() == 3);
}

TEST_CASE("multi_index try_pop_from returns nullopt for an absent operator",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(make_task(10, make_keys(1, SiriusPhysicalOperatorType::FILTER)));
  REQUIRE_FALSE(q.try_pop_from(operator_index{SiriusPhysicalOperatorType::HASH_JOIN}).has_value());
  REQUIRE(q.size() == 1);
}

// =============================================================================
// Pipeline- and query-index popping
// =============================================================================

TEST_CASE("multi_index pipeline and query indexes pop independently",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // (priority, operator, pipeline, query)
  q.push(make_task(1, make_keys(5, SiriusPhysicalOperatorType::FILTER, 100, 7)));
  q.push(make_task(2, make_keys(2, SiriusPhysicalOperatorType::PROJECTION, 100, 8)));
  q.push(make_task(3, make_keys(9, SiriusPhysicalOperatorType::FILTER, 200, 7)));

  REQUIRE(q.size(pipeline_index{100}) == 2);
  REQUIRE(q.size(query_index{7}) == 2);

  // Pipeline 100: lowest priority is id 2 (p=2).
  auto by_pipeline = q.try_pop_from(pipeline_index{100});
  REQUIRE(by_pipeline.has_value());
  REQUIRE((*by_pipeline)->id == 2);

  // Query 7: remaining members are ids 1 (p=5) and 3 (p=9); back is id 3.
  auto by_query = q.try_pop_back_from(query_index{7});
  REQUIRE(by_query.has_value());
  REQUIRE((*by_query)->id == 3);

  REQUIRE(q.size() == 1);
  REQUIRE(q.pop()->id == 1);
}

TEST_CASE("multi_index try_pop_back_from walks a pipeline bucket from the back",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // Pipeline 5 holds three tasks with priorities 2, 8, 4 (ids 1, 2, 3).
  q.push(make_task(1, make_keys(2, SiriusPhysicalOperatorType::FILTER, 5, 0)));
  q.push(make_task(2, make_keys(8, SiriusPhysicalOperatorType::FILTER, 5, 0)));
  q.push(make_task(3, make_keys(4, SiriusPhysicalOperatorType::FILTER, 5, 0)));
  // A decoy in another pipeline must never be returned.
  q.push(make_task(9, make_keys(1, SiriusPhysicalOperatorType::FILTER, 6, 0)));

  const pipeline_index pipe{5};
  REQUIRE(q.size(pipe) == 3);
  REQUIRE(q.try_pop_back_from(pipe).value()->id == 2);   // highest priority (8)
  REQUIRE(q.try_pop_back_from(pipe).value()->id == 3);   // then 4
  REQUIRE(q.try_pop_back_from(pipe).value()->id == 1);   // then 2
  REQUIRE_FALSE(q.try_pop_back_from(pipe).has_value());  // bucket drained
  REQUIRE(q.size() == 1);                                // decoy id 9 remains
  REQUIRE(q.pop()->id == 9);
}

// =============================================================================
// Cross-index consistency: a pop through one index updates the others
// =============================================================================

TEST_CASE("multi_index removal keeps every index consistent", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(make_task(1, make_keys(3, SiriusPhysicalOperatorType::FILTER, 100, 7)));
  q.push(make_task(2, make_keys(1, SiriusPhysicalOperatorType::FILTER, 100, 7)));

  // Pop id 2 via the operator index; it must vanish from the global order, the
  // pipeline bucket, and the query bucket too.
  auto popped = q.try_pop_from(operator_index{SiriusPhysicalOperatorType::FILTER});
  REQUIRE(popped.has_value());
  REQUIRE((*popped)->id == 2);

  REQUIRE(q.size() == 1);
  REQUIRE(q.size(operator_index{SiriusPhysicalOperatorType::FILTER}) == 1);
  REQUIRE(q.size(pipeline_index{100}) == 1);
  REQUIRE(q.size(query_index{7}) == 1);

  // Draining the last task empties every index and prunes the buckets.
  auto last = q.try_pop_back();
  REQUIRE(last.has_value());
  REQUIRE((*last)->id == 1);
  REQUIRE(q.empty());
  REQUIRE(q.size(operator_index{SiriusPhysicalOperatorType::FILTER}) == 0);
  REQUIRE(q.size(pipeline_index{100}) == 0);
  REQUIRE(q.size(query_index{7}) == 0);
  REQUIRE_FALSE(q.try_pop_from(pipeline_index{100}).has_value());
}

TEST_CASE("multi_index refills a bucket after it drains to empty", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  const operator_index filter{SiriusPhysicalOperatorType::FILTER};

  q.push(make_task(1, make_keys(1, SiriusPhysicalOperatorType::FILTER)));
  REQUIRE(q.try_pop_from(filter).has_value());
  REQUIRE(q.size(filter) == 0);

  // Pushing the same operator again must recreate the pruned bucket cleanly.
  q.push(make_task(2, make_keys(1, SiriusPhysicalOperatorType::FILTER)));
  REQUIRE(q.size(filter) == 1);
  auto again = q.try_pop_from(filter);
  REQUIRE(again.has_value());
  REQUIRE((*again)->id == 2);
}

// =============================================================================
// Storage backend is pluggable: std::list must behave identically
// =============================================================================

TEST_CASE("multi_index works with the std::list storage backend", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload, detail::list_storage> q(
    [](const payload& p) { return p.keys; });
  // (priority, operator, pipeline, query)
  q.push(make_task(10, make_keys(4, SiriusPhysicalOperatorType::FILTER, 1, 100)));
  q.push(make_task(11, make_keys(1, SiriusPhysicalOperatorType::FILTER, 2, 100)));
  q.push(make_task(12, make_keys(9, SiriusPhysicalOperatorType::PROJECTION, 1, 200)));

  // The list backend must drive every index just like the colony backend.
  REQUIRE(q.size(operator_index{SiriusPhysicalOperatorType::FILTER}) == 2);
  REQUIRE(q.size(pipeline_index{1}) == 2);
  REQUIRE(q.size(query_index{100}) == 2);

  REQUIRE(q.pop()->id == 11);  // lowest global priority

  // Popping id 11 must have left every secondary index consistent.
  auto filtered = q.try_pop_from(operator_index{SiriusPhysicalOperatorType::FILTER});
  REQUIRE(filtered.has_value());
  REQUIRE((*filtered)->id == 10);
  REQUIRE(q.size(query_index{100}) == 0);  // both query-100 tasks gone

  // Drain the pipeline-1 bucket to empty, then refill it on the list backend.
  const pipeline_index pipe1{1};
  REQUIRE(q.try_pop_back_from(pipe1).value()->id == 12);
  REQUIRE(q.size(pipe1) == 0);
  REQUIRE(q.empty());
  q.push(make_task(13, make_keys(3, SiriusPhysicalOperatorType::PROJECTION, 1, 300)));
  REQUIRE(q.size(pipe1) == 1);
  REQUIRE(q.pop_back()->id == 13);
  REQUIRE(q.empty());
}

// =============================================================================
// Thread-safety: blocking pop, interrupt, drain
// =============================================================================

TEST_CASE("multi_index blocking pop wakes when a task is pushed", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());

  int popped_id = -1;
  std::thread consumer([&] {
    auto task = q.pop();  // blocks until the producer pushes
    if (task) { popped_id = task->id; }
  });

  // Give the consumer a moment to actually block, then produce.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  q.push(make_task(42, make_keys(1)));
  consumer.join();

  REQUIRE(popped_id == 42);
  REQUIRE(q.empty());
}

TEST_CASE("multi_index interrupt wakes a blocked pop with nullptr", "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());

  bool got_null = false;
  std::thread consumer([&] {
    auto task = q.pop();  // blocks on the empty queue
    got_null  = (task == nullptr);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  REQUIRE(q.is_open());
  q.interrupt();  // must wake the consumer, which returns nullptr
  consumer.join();

  REQUIRE(got_null);
  REQUIRE_FALSE(q.is_open());

  // After interrupt, blocking pop keeps draining existing items but does not wait.
  q.push(make_task(7, make_keys(1)));
  REQUIRE(q.pop()->id == 7);    // item present -> returned
  REQUIRE(q.pop() == nullptr);  // empty + interrupted -> nullptr, no hang

  // reactivate() restores normal blocking behavior.
  q.reactivate();
  REQUIRE(q.is_open());
}

TEST_CASE("multi_index drain drops all tasks and clears every index",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  q.push(make_task(1, make_keys(1, SiriusPhysicalOperatorType::FILTER, 10, 100)));
  q.push(make_task(2, make_keys(2, SiriusPhysicalOperatorType::PROJECTION, 20, 200)));
  REQUIRE(q.size() == 2);

  q.drain();

  REQUIRE(q.empty());
  REQUIRE(q.size() == 0);
  REQUIRE(q.operator_bucket_count() == 0);
  REQUIRE(q.pipeline_bucket_count() == 0);
  REQUIRE(q.query_bucket_count() == 0);
  REQUIRE_FALSE(q.try_pop().has_value());

  // The queue is still usable after a drain.
  q.push(make_task(3, make_keys(1)));
  REQUIRE(q.pop()->id == 3);
}

TEST_CASE("multi_index drain(query_index) drops only that query's tasks",
          "[multi_index_priority_queue]")
{
  multi_index_priority_queue<payload> q(by_keys());
  // query 7: ids 1, 3 (different operators/pipelines); query 8: id 2.
  q.push(make_task(1, make_keys(5, SiriusPhysicalOperatorType::FILTER, 100, 7)));
  q.push(make_task(2, make_keys(2, SiriusPhysicalOperatorType::PROJECTION, 100, 8)));
  q.push(make_task(3, make_keys(9, SiriusPhysicalOperatorType::HASH_JOIN, 200, 7)));

  q.drain(query_index{7});

  // Only query 8 (id 2) survives, and it survives in every index.
  REQUIRE(q.size() == 1);
  REQUIRE(q.query_bucket_count() == 1);
  REQUIRE(q.size(query_index{7}) == 0);
  REQUIRE(q.size(query_index{8}) == 1);
  REQUIRE(q.size(operator_index{SiriusPhysicalOperatorType::FILTER}) == 0);
  REQUIRE(q.size(operator_index{SiriusPhysicalOperatorType::HASH_JOIN}) == 0);
  REQUIRE(q.size(operator_index{SiriusPhysicalOperatorType::PROJECTION}) == 1);
  REQUIRE(q.size(pipeline_index{100}) == 1);
  REQUIRE(q.size(pipeline_index{200}) == 0);
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
        if (auto task = q.try_pop()) { consumed.fetch_add(1); }
      }
    });
  }

  std::vector<std::thread> producers;
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < kPerProducer; ++i) {
        q.push(make_task(p * kPerProducer + i, make_keys(i % 7)));
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
