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

#include <catch.hpp>

#include <atomic>
#include <future>
#include <memory>
#include <stdexcept>

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
      // The pool rejects the submission before moving its callable. Destruction of that
      // callable during unwinding runs here with the last dispatcher slot still reserved,
      // just before submit() enters its catch handler. Queue the competing producer's work
      // at that exact boundary, without timing assumptions or production failure hooks.
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

TEST_CASE("dispatcher drains long pending chains without resubmission", "[scoped_dispatcher]")
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
