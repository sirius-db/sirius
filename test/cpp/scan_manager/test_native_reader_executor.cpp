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
#include "scan_manager/preparation_fixture.hpp"

using namespace sirius::scan_manager;
using namespace sirius::test::preparation;
using namespace std::chrono_literals;

TEST_CASE("Native executor keeps the slot busy until the operation destructor returns",
          "[scan_preparation][native_lane]")
{
  native_reader_executor executor(1, 1);
  auto gate        = std::make_shared<preparation_gate>();
  auto observation = executor.observe(preparation_wakeup{gate});
  auto destroying  = std::make_shared<hold>();
  auto leases      = executor.reserve(1);
  std::thread::id worker, destroyed;
  std::atomic<bool> destructor_failed{false};
  struct blocking_operation : native_reader_executor::operation {
    std::shared_ptr<hold> destroying;
    std::thread::id& worker;
    std::thread::id& destroyed;
    std::atomic<bool>& failed;
    blocking_operation(std::shared_ptr<hold> h,
                       std::thread::id& w,
                       std::thread::id& d,
                       std::atomic<bool>& f)
      : destroying(std::move(h)), worker(w), destroyed(d), failed(f)
    {
    }
    void run() noexcept override { worker = std::this_thread::get_id(); }
    ~blocking_operation() override
    {
      destroyed = std::this_thread::get_id();
      try {
        destroying->wait();
      } catch (...) {
        failed = true;
      }
    }
  };
  auto retirement = std::shared_ptr<void>(nullptr, [&](void*) {
    destroying->release();
    std::unique_lock lock(gate->mutex);
    gate->cv.wait(lock, [&] { return leases.front()->idle(); });
  });
  leases.front()->submit(
    std::make_unique<blocking_operation>(destroying, worker, destroyed, destructor_failed));
  REQUIRE(destroying->await());
  CHECK(worker == destroyed);
  CHECK_FALSE(leases.front()->idle());
  CHECK_THROWS_WITH(executor.stop(), "cannot stop native executor with active readers");
  CHECK_THROWS_WITH(executor.reserve(1), "native reader capacity unavailable for complete attempt");
  retirement.reset();
  CHECK(leases.front()->idle());
  CHECK_FALSE(destructor_failed);
  leases.clear();
  auto reused = executor.reserve(1);
  CHECK(reused.front()->idle());
}

TEST_CASE("native runtime starts lazily and closes admission on stop",
          "[scan_preparation][native_runtime]")
{
  native_reader_executor executor(2, 4);
  size_t created = 0;
  executor.before_thread_for_testing([&](size_t) { ++created; });
  CHECK(executor.reserve(0).empty());
  CHECK(created == 0);
  CHECK_THROWS_WITH(executor.reserve(5), "native reader capacity unavailable for complete attempt");
  CHECK(created == 0);
  auto leases = executor.reserve(1);
  CHECK(created == 2);
  CHECK_THROWS_WITH(executor.stop(), "cannot stop native executor with active readers");
  // Rejected stop must not close admission or prevent an existing reader from retiring.
  auto other = executor.reserve(3);
  CHECK(created == 2);
  other.clear();
  leases.clear();
  CHECK_NOTHROW(executor.stop());
  CHECK_NOTHROW(executor.stop());
  CHECK_THROWS_WITH(executor.reserve(1), "native executor is stopped");
  CHECK_THROWS_WITH(executor.reserve(0), "native executor is stopped");
}

TEST_CASE("partial native worker startup rolls back before admitting readers",
          "[scan_preparation][native_runtime]")
{
  native_reader_executor executor(2, 2);
  bool fail      = true;
  size_t created = 0;
  executor.before_thread_for_testing([&](size_t i) {
    if (i == 1 && fail) throw std::runtime_error("native startup failed");
    ++created;
  });
  CHECK_THROWS_WITH(executor.reserve(2), "native startup failed");
  CHECK(created == 1);
  fail        = false;
  auto leases = executor.reserve(2);
  CHECK(created == 3);
  REQUIRE(leases.size() == 2);
  size_t executions = 0;
  struct probe : native_reader_executor::operation {
    size_t& executions;
    explicit probe(size_t& count) : executions(count) {}
    void run() noexcept override { ++executions; }
  };
  auto gate        = std::make_shared<preparation_gate>();
  auto observation = executor.observe(preparation_wakeup{gate});
  auto retirement  = std::shared_ptr<void>(nullptr, [&](void*) {
    std::unique_lock lock(gate->mutex);
    gate->cv.wait(lock, [&] { return leases.front()->idle(); });
  });
  leases.front()->submit(std::make_unique<probe>(executions));
  {
    std::unique_lock lock(gate->mutex);
    REQUIRE(gate->cv.wait_for(lock, 1s, [&] { return leases.front()->idle(); }));
  }
  CHECK(executions == 1);
  retirement.reset();
  leases.clear();
  CHECK(created == 3);
  CHECK_NOTHROW(executor.stop());
}
