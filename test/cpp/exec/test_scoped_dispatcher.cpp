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

TEST_CASE("dispatcher settles a refused pool submission", "[scoped_dispatcher]")
{
  sirius::exec::static_thread_pool pool(1);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  pool.stop();
  CHECK_THROWS(dispatcher.enqueue([] {}));
  dispatcher.wait_for_all();
  CHECK_THROWS(dispatcher.schedule([] {}));
  dispatcher.wait_for_all();
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
