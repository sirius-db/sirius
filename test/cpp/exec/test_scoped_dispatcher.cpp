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

#include "exec/scoped_dispatcher.hpp"
#include "query_id.hpp"

#include <catch.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

TEST_CASE("dispatcher settles a refused pool submission", "[scoped_dispatcher]")
{
  sirius::exec::static_thread_pool pool(1);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  pool.stop();
  SECTION("enqueue")
  {
    CHECK_THROWS_WITH(dispatcher.enqueue([] {}), "thread pool is stopped");
  }
  SECTION("schedule")
  {
    CHECK_THROWS_WITH(dispatcher.schedule([] {}), "thread pool is stopped");
  }
  dispatcher.wait_for_all();
  CHECK_FALSE(dispatcher.schedule([] {}));
  bool executed = false;
  CHECK_NOTHROW(dispatcher.enqueue([&] { executed = true; }));
  dispatcher.wait_for_all();
  CHECK_FALSE(executed);
}

TEST_CASE("failed dispatcher submission discards pending work without executing it inline",
          "[scoped_dispatcher]")
{
  sirius::exec::static_thread_pool pool(1);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  unsigned executed    = 0;
  unsigned destroyed   = 0;
  auto pending_capture = std::shared_ptr<int>(new int{}, [&](int* value) {
    delete value;
    // Capture destruction must happen outside the dispatcher mutex. This reentrant enqueue
    // must be refused because the failed submission has stopped this dispatcher.
    dispatcher.enqueue([&] { ++executed; });
    ++destroyed;
  });
  auto failed_capture  = std::shared_ptr<int>(
    new int{}, [&, pending_capture = std::move(pending_capture)](int* value) mutable {
      delete value;
      // Destruction of the refused callable during unwinding runs here outside the pool
      // mutex, with the last dispatcher slot still reserved,
      // just before submit() enters its catch handler. Queue the competing producer's work
      // at that exact boundary, without timing assumptions or production failure hooks.
      pool.stop();  // Reentering the pool must also be safe during capture destruction.
      dispatcher.enqueue([&, pending_capture = std::move(pending_capture)] { ++executed; });
    });

  pool.stop();
  auto task = [failed_capture = std::move(failed_capture)] {};
  SECTION("enqueue")
  {
    CHECK_THROWS_WITH(dispatcher.enqueue(std::move(task)), "thread pool is stopped");
  }
  SECTION("schedule")
  {
    CHECK_THROWS_WITH(dispatcher.schedule(std::move(task)), "thread pool is stopped");
  }
  dispatcher.wait_for_all();
  CHECK(executed == 0);
  CHECK(destroyed == 1);
  CHECK_FALSE(dispatcher.schedule([] {}));
}

TEST_CASE("dispatcher stop callbacks can reenter enqueue", "[scoped_dispatcher]")
{
  sirius::exec::static_thread_pool pool(1);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  std::promise<void> entered, release;
  auto released     = release.get_future();
  bool callback_ran = false;
  bool task_ran     = false;
  dispatcher.enqueue([&](std::stop_token token) {
    std::stop_callback callback(token, [&] {
      dispatcher.enqueue([&] { task_ran = true; });
      callback_ran = true;
    });
    entered.set_value();
    released.wait();
  });
  entered.get_future().wait();
  dispatcher.request_stop();
  release.set_value();
  dispatcher.wait_for_all();
  CHECK(callback_ran);
  CHECK_FALSE(task_ran);
}

TEST_CASE("dispatcher drains long pending chains across handoffs", "[scoped_dispatcher]")
{
  sirius::exec::static_thread_pool pool(1);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  std::promise<void> entered, release;
  auto released = release.get_future();
  dispatcher.enqueue([&] {
    entered.set_value();
    released.wait();
  });
  entered.get_future().wait();
  std::atomic<unsigned> completed{0};
  for (int i = 0; i < 10000; ++i)
    dispatcher.enqueue([&] { ++completed; });
  release.set_value();
  dispatcher.wait_for_all();
  CHECK(completed == 10000);
}

TEST_CASE("dispatcher handoffs honor query priority and FIFO ties", "[scoped_dispatcher]")
{
  sirius::exec::static_thread_pool pool(1);
  auto const first_priority = sirius::query_priority_bits(sirius::make_query_id(2));
  auto second_priority      = first_priority;
  std::vector<int> expected;
  SECTION("older query overtakes a newer query's pending chain")
  {
    second_priority = sirius::query_priority_bits(sirius::make_query_id(1));
    expected        = {1, 4, 5, 2, 3};
  }
  SECTION("older query's continuations precede a newer query already queued")
  {
    second_priority = sirius::query_priority_bits(sirius::make_query_id(3));
    expected        = {1, 2, 3, 4, 5};
  }
  SECTION("equal priorities yield to already queued work") { expected = {1, 4, 2, 5, 3}; }

  sirius::exec::scoped_dispatcher first(pool, 1, first_priority);
  sirius::exec::scoped_dispatcher second(pool, 1, second_priority);
  std::vector<int> order;
  std::promise<void> entered, release;
  auto released = release.get_future();
  first.enqueue([&] {
    order.push_back(1);
    entered.set_value();
    released.wait();
  });
  entered.get_future().wait();
  first.enqueue([&] { order.push_back(2); });
  first.enqueue([&] { order.push_back(3); });
  // Exercise the blocking submission API as well; its continuation must retain priority.
  bool const accepted = second.schedule([&] { order.push_back(4); });
  second.enqueue([&] { order.push_back(5); });
  release.set_value();
  first.wait_for_all();
  second.wait_for_all();
  CHECK(accepted);
  CHECK(order == expected);
}

TEST_CASE("cancellation keeps a queued continuation outstanding", "[scoped_dispatcher]")
{
  sirius::exec::static_thread_pool pool(1);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  std::promise<void> entered, release, blocker_entered, release_blocker;
  auto released        = release.get_future();
  auto blocker_release = release_blocker.get_future();
  unsigned executed = 0, destroyed = 0;
  dispatcher.enqueue([&] {
    entered.set_value();
    released.wait();
  });
  entered.get_future().wait();
  auto capture = std::shared_ptr<int>(new int{}, [&](int* value) {
    delete value;
    // Cancellation must discard captures outside the dispatcher mutex.
    dispatcher.enqueue([&] { ++executed; });
    ++destroyed;
  });
  dispatcher.enqueue([&, capture = std::move(capture)] { ++executed; });
  // This is already queued when the dispatcher hands off its reserved slot.
  pool.schedule([&] {
    blocker_entered.set_value();
    blocker_release.wait();
  });
  release.set_value();
  blocker_entered.get_future().wait();
  dispatcher.request_stop();
  auto drained      = std::async(std::launch::async, [&] { dispatcher.wait_for_all(); });
  auto const status = drained.wait_for(std::chrono::milliseconds(50));
  release_blocker.set_value();
  drained.get();
  CHECK(status == std::future_status::timeout);
  CHECK(executed == 0);
  CHECK(destroyed == 1);
}

TEST_CASE("refused dispatcher handoff drains pending work on the current worker",
          "[scoped_dispatcher]")
{
  sirius::exec::static_thread_pool pool(1);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  std::promise<void> entered, release;
  auto released = release.get_future();
  std::thread::id worker;
  bool same_worker   = true;
  unsigned completed = 0, destroyed = 0;
  dispatcher.enqueue([&] {
    worker = std::this_thread::get_id();
    entered.set_value();
    released.wait();
    // Refuse the next handoff without adding a failure injection API to production code.
    pool.stop();
  });
  entered.get_future().wait();
  for (int i = 0; i < 3; ++i) {
    auto capture = std::shared_ptr<int>(new int{}, [&](int* value) {
      delete value;
      ++destroyed;
    });
    dispatcher.enqueue([&, capture = std::move(capture)] {
      same_worker = same_worker && std::this_thread::get_id() == worker;
      ++completed;
    });
  }
  release.set_value();
  dispatcher.wait_for_all();
  CHECK(completed == 3);
  CHECK(destroyed == 3);
  CHECK(same_worker);
  CHECK(worker != std::this_thread::get_id());
}
