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
#include "exec/query_lifecycle_registry.hpp"
#include "query_id.hpp"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using sirius::make_query_id;
using sirius::exec::query_lifecycle_registry;
using sirius::exec::query_lifecycle_state;

TEST_CASE("an unknown query refuses work", "[query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry registry;
  REQUIRE_FALSE(registry.accepts_work(make_query_id(7)));
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
  REQUIRE_FALSE(registry.accepts_work(q));
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
  REQUIRE_FALSE(registry.accepts_work(unknown));

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

namespace {
using namespace std::chrono_literals;
using sirius::exec::query_submission_status;
using work_lease       = query_lifecycle_registry::work_lease;
using submission_guard = query_lifecycle_registry::submission_guard;

struct leased_request {
  work_lease lease;
  sirius::query_id_t query;
};

using request_queue = sirius::exec::multi_index_priority_queue<leased_request>;

sirius::exec::index_keys request_keys(const leased_request& request)
{
  return {sirius::query_priority_bits(request.query),
          sirius::op::SiriusPhysicalOperatorType::INVALID,
          sirius::value_of(request.query)};
}

static_assert(!std::is_copy_constructible_v<work_lease>);
static_assert(!std::is_copy_assignable_v<work_lease>);
static_assert(std::is_nothrow_move_constructible_v<work_lease>);
static_assert(std::is_nothrow_move_assignable_v<work_lease>);
static_assert(!std::is_copy_constructible_v<submission_guard>);
static_assert(!std::is_copy_assignable_v<submission_guard>);
static_assert(std::is_nothrow_move_constructible_v<submission_guard>);
static_assert(std::is_nothrow_move_assignable_v<submission_guard>);
}  // namespace

TEST_CASE("submission refusals preserve the acquisition reason",
          "[query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry registry;
  const auto q = make_query_id(1);
  auto missing = registry.try_begin_submission(q);
  REQUIRE_FALSE(missing);
  REQUIRE(missing.status() == query_submission_status::unknown);
  REQUIRE_FALSE(registry.try_acquire_work(q));

  registry.open_query(q);
  registry.quiesce(q);
  auto refused = registry.try_begin_submission(q);
  REQUIRE_FALSE(refused);
  REQUIRE(refused.status() == query_submission_status::quiescing);
  REQUIRE_FALSE(registry.try_acquire_work(q));
  registry.close(q);
  // Diagnosing a refusal must not re-read the now-erased registry and call it an unknown query.
  REQUIRE(refused.status() == query_submission_status::quiescing);
  REQUIRE(registry.try_begin_submission(q).status() == query_submission_status::unknown);
}

TEST_CASE("duplicate registration cannot reopen a query or replace its accounting",
          "[query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry registry;
  const auto q = make_query_id(1);
  registry.open_query(q);
  auto submission = registry.try_begin_submission(q);
  REQUIRE_THROWS_AS(registry.open_query(q), std::logic_error);
  REQUIRE(registry.activity(q).submissions == 1);
  registry.quiesce(q);
  REQUIRE_THROWS_AS(registry.open_query(q), std::logic_error);
  REQUIRE_FALSE(registry.accepts_work(q));
  REQUIRE(registry.activity(q).work == 1);
  REQUIRE_THROWS_AS(registry.close(q), std::logic_error);
  REQUIRE(registry.size() == 1);
  submission.finish();
  REQUIRE_NOTHROW(registry.close(q));
}

TEST_CASE("submission and work moves preserve continuous accounting",
          "[query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry registry;
  const auto q = make_query_id(1);
  registry.open_query(q);
  auto first  = registry.try_begin_submission(q);
  auto second = registry.try_begin_submission(q);
  REQUIRE(registry.activity(q).submissions == 2);
  REQUIRE(registry.activity(q).work == 2);

  second = std::move(first);  // Releases second's old registration, transfers first's.
  REQUIRE_FALSE(first);
  REQUIRE(second);
  REQUIRE(registry.activity(q).submissions == 1);
  REQUIRE(registry.activity(q).work == 1);
  auto moved = std::move(second);
  REQUIRE_FALSE(second);
  auto lease = moved.take_work_lease();
  REQUIRE_THROWS_AS(moved.take_work_lease(), std::logic_error);
  moved.finish();
  moved.finish();
  REQUIRE(registry.activity(q).submissions == 0);
  REQUIRE(registry.activity(q).work == 1);

  auto independent = registry.try_acquire_work(q);
  REQUIRE(registry.activity(q).work == 2);
  independent = std::move(lease);
  REQUIRE_FALSE(lease);
  REQUIRE(registry.activity(q).work == 1);
  auto final_lease = std::move(independent);
  registry.quiesce_and_wait_for_submissions(q);
  REQUIRE_THROWS_AS(registry.close(q), std::logic_error);
  final_lease.reset();
  final_lease.reset();
  REQUIRE(registry.activity(q).work == 0);
  REQUIRE_NOTHROW(registry.wait_for_work(q));
  registry.close(q);
}

TEST_CASE("quiesce settles a paused publisher before draining its queue",
          "[query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry registry;
  const auto a = make_query_id(1);
  const auto b = make_query_id(2);
  registry.open_query(a);
  registry.open_query(b);
  request_queue queue(request_keys);
  auto paused_publisher = registry.try_begin_submission(a);
  auto other_publisher  = registry.try_begin_submission(b);
  REQUIRE(paused_publisher);
  REQUIRE(other_publisher);

  // Force the exact check-to-push window: A is already admitted but has not inserted anything.
  registry.quiesce(a);
  std::promise<void> cleanup_started;
  auto started = cleanup_started.get_future();
  auto cleanup = std::async(std::launch::async, [&] {
    cleanup_started.set_value();
    registry.quiesce_and_wait_for_submissions(a);
    queue.drain(sirius::exec::query_index{sirius::value_of(a)});
  });
  started.wait();
  CHECK(cleanup.wait_for(20ms) == std::future_status::timeout);
  CHECK_FALSE(registry.try_begin_submission(a));
  CHECK(registry.accepts_work(b));

  // Previously admitted publication is allowed to finish after quiesce, before the drain.
  auto request =
    std::make_unique<leased_request>(leased_request{paused_publisher.take_work_lease(), a});
  CHECK(queue.push(std::move(request)));
  paused_publisher.finish();
  CHECK(cleanup.wait_for(5s) == std::future_status::ready);
  cleanup.get();
  CHECK(queue.empty());
  CHECK(registry.activity(a).work == 0);
  CHECK(registry.activity(b).submissions == 1);  // Cleanup never waited for B.
  registry.close(a);
}

TEST_CASE("a lease covers the gap after queue pop and before worker attribution",
          "[query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry registry;
  const auto q = make_query_id(1);
  registry.open_query(q);
  request_queue queue(request_keys);
  auto submission = registry.try_begin_submission(q);
  auto request = std::make_unique<leased_request>(leased_request{submission.take_work_lease(), q});
  REQUIRE(queue.push(std::move(request)));
  submission.finish();

  auto in_hand = queue.pop();
  REQUIRE(queue.empty());
  REQUIRE(registry.activity(q).work == 1);
  registry.quiesce_and_wait_for_submissions(q);
  queue.drain();
  std::promise<void> waiter_started;
  auto started = waiter_started.get_future();
  auto retired = std::async(std::launch::async, [&] {
    waiter_started.set_value();
    registry.wait_for_work(q);
  });
  started.wait();
  CHECK(retired.wait_for(20ms) == std::future_status::timeout);
  // Move to another thread without a release/reacquire interval, then dispose there.
  std::thread worker([work = std::move(in_hand)]() mutable { work.reset(); });
  worker.join();
  CHECK(retired.wait_for(5s) == std::future_status::ready);
  retired.get();
  CHECK(registry.activity(q).work == 0);
  registry.close(q);
}

TEST_CASE("exceptions and rejected pushes release submission and work guards",
          "[query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry registry;
  const auto q = make_query_id(1);
  registry.open_query(q);
  request_queue queue(request_keys);
  SECTION("construction exception")
  {
    auto throws = [&] {
      auto submission = registry.try_begin_submission(q);
      auto lease      = submission.take_work_lease();
      throw std::runtime_error("construction failed");
    };
    REQUIRE_THROWS_AS(throws(), std::runtime_error);
  }
  SECTION("queue rejects insertion")
  {
    queue.interrupt();
    auto submission = registry.try_begin_submission(q);
    auto request =
      std::make_unique<leased_request>(leased_request{submission.take_work_lease(), q});
    REQUIRE_FALSE(queue.push(std::move(request)));
    REQUIRE(registry.activity(q).work == 0);
    REQUIRE(registry.activity(q).submissions == 1);
  }
  SECTION("queue key extraction throws")
  {
    request_queue throwing_queue([](const leased_request&) -> sirius::exec::index_keys {
      throw std::runtime_error("key extraction failed");
    });
    auto throws = [&] {
      auto submission = registry.try_begin_submission(q);
      auto request =
        std::make_unique<leased_request>(leased_request{submission.take_work_lease(), q});
      throwing_queue.push(std::move(request));
    };
    REQUIRE_THROWS_AS(throws(), std::runtime_error);
  }
  REQUIRE(registry.activity(q).submissions == 0);
  REQUIRE(registry.activity(q).work == 0);
  registry.quiesce_and_wait_for_submissions(q);
  registry.close(q);
}

TEST_CASE("lifecycle waits require a closed gate and clear preserves live accounting",
          "[query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry registry;
  const auto q = make_query_id(1);
  registry.open_query(q);
  REQUIRE_THROWS_AS(registry.wait_for_submissions(q), std::logic_error);
  REQUIRE_THROWS_AS(registry.wait_for_work(q), std::logic_error);
  auto lease = registry.try_acquire_work(q);
  REQUIRE_THROWS_AS(registry.clear(), std::logic_error);
  REQUIRE(registry.size() == 1);
  REQUIRE_FALSE(registry.accepts_work(q));
  lease.reset();
  REQUIRE_NOTHROW(registry.clear());
  REQUIRE(registry.size() == 0);
  REQUIRE_NOTHROW(registry.wait_for_submissions(q));
  REQUIRE_NOTHROW(registry.wait_for_work(q));
}

TEST_CASE("lifecycle handles can be released after the registry is destroyed",
          "[query_lifecycle_gate][concurrency]")
{
  work_lease lease;
  submission_guard submission;
  {
    query_lifecycle_registry registry;
    const auto q = make_query_id(1);
    registry.open_query(q);
    submission = registry.try_begin_submission(q);
    lease      = submission.take_work_lease();
  }
  // This protects accounting storage only; production resource owners must still drain users.
  REQUIRE_NOTHROW(submission.finish());
  REQUIRE_NOTHROW(lease.reset());
}

TEST_CASE("lifecycle waiters wake for each order of publisher and work retirement",
          "[query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry registry;
  const auto q = make_query_id(1);
  registry.open_query(q);
  auto submission = registry.try_begin_submission(q);
  work_lease lease;
  const auto order = GENERATE(0, 1, 2);
  // 0: both claims stay in the guard; 1: publisher retires first; 2: transferred work retires
  // first.
  if (order != 0) { lease = submission.take_work_lease(); }
  registry.quiesce(q);

  std::promise<void> publisher_wait_started;
  std::promise<void> work_wait_started;
  auto publisher_started = publisher_wait_started.get_future();
  auto work_started      = work_wait_started.get_future();
  auto publishers_done   = std::async(std::launch::async, [&] {
    publisher_wait_started.set_value();
    registry.wait_for_submissions(q);
  });
  auto work_done         = std::async(std::launch::async, [&] {
    work_wait_started.set_value();
    registry.wait_for_work(q);
  });
  publisher_started.wait();
  work_started.wait();
  CHECK(publishers_done.wait_for(20ms) == std::future_status::timeout);
  CHECK(work_done.wait_for(20ms) == std::future_status::timeout);

  if (order == 1) {
    submission.finish();
    CHECK(publishers_done.wait_for(5s) == std::future_status::ready);
    CHECK(work_done.wait_for(20ms) == std::future_status::timeout);
    lease.reset();
  } else {
    if (order == 2) {
      lease.reset();
      CHECK(work_done.wait_for(20ms) == std::future_status::timeout);
    }
    submission.finish();
  }

  CHECK(publishers_done.wait_for(5s) == std::future_status::ready);
  CHECK(work_done.wait_for(5s) == std::future_status::ready);
  publishers_done.get();
  work_done.get();
  CHECK(registry.activity(q).submissions == 0);
  CHECK(registry.activity(q).work == 0);
  registry.close(q);
}

TEST_CASE("query resources survive their engine and cannot retire while borrowed",
          "[query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry registry;
  auto q = make_query_id(31);
  registry.open_query(q);
  bool destroyed = false;
  auto owner     = std::shared_ptr<int>(new int(42), [&](int* value) {
    // Destruction can call back into the registry: neither mutex may be held here.
    CHECK(registry.size() <= 1);
    destroyed = true;
    delete value;
  });
  registry.retain_resources(q, owner);
  owner.reset();
  auto borrow = registry.try_acquire_work(q);
  registry.quiesce_and_wait_for_submissions(q);
  CHECK_FALSE(destroyed);
  REQUIRE_THROWS(registry.release_resources(q));
  borrow.reset();
  SECTION("explicit resource retirement")
  {
    registry.release_resources(q);
    registry.close(q);
  }
  SECTION("close") { registry.close(q); }
  SECTION("shutdown") { registry.clear(); }
  CHECK(destroyed);
}
