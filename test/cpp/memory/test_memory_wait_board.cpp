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
#include "memory/memory_wait_board.hpp"

#include <memory>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using sirius::memory::memory_wait_board;

static_assert(!std::is_copy_constructible_v<memory_wait_board::waiter>);
static_assert(!std::is_copy_assignable_v<memory_wait_board::oom_token>);
static_assert(std::is_nothrow_move_constructible_v<memory_wait_board::waiter>);
static_assert(std::is_nothrow_move_assignable_v<memory_wait_board::oom_token>);

TEST_CASE("memory_wait_board counts waiters and OOM tokens per device", "[memory_wait_board]")
{
  auto board = std::make_shared<memory_wait_board>(2);
  REQUIRE(board->device_count() == 3);
  for (int d = 0; d < 3; ++d) {
    CHECK(board->waiting(d) == 0);
    CHECK(board->oom_pending(d) == 0);
  }

  {
    auto w0 = board->enter_wait(0);
    auto w1 = board->enter_wait(0);
    auto w2 = board->enter_wait(2);
    auto t1 = board->hold_oom(1);
    CHECK(w0.active());
    CHECK(w0.device() == 0);
    CHECK(t1.device() == 1);
    CHECK(board->waiting(0) == 2);
    CHECK(board->waiting(1) == 0);
    CHECK(board->waiting(2) == 1);
    CHECK(board->oom_pending(0) == 0);
    CHECK(board->oom_pending(1) == 1);
    CHECK(board->oom_pending(2) == 0);

    // The two counts are independent.
    {
      auto t0 = board->hold_oom(0);
      CHECK(board->oom_pending(0) == 1);
      CHECK(board->waiting(0) == 2);
    }
    CHECK(board->oom_pending(0) == 0);
  }
  for (int d = 0; d < 3; ++d) {
    CHECK(board->waiting(d) == 0);
    CHECK(board->oom_pending(d) == 0);
  }
}

TEST_CASE("memory_wait_board holds move without double counting or double release",
          "[memory_wait_board]")
{
  auto board = std::make_shared<memory_wait_board>(1);

  SECTION("move construction transfers the hold")
  {
    auto a = board->enter_wait(1);
    auto b = std::move(a);
    CHECK_FALSE(a.active());  // NOLINT(bugprone-use-after-move)
    CHECK(a.device() == -1);  // NOLINT(bugprone-use-after-move)
    CHECK(b.active());
    CHECK(board->waiting(1) == 1);
    {
      auto c = std::move(b);
    }
    CHECK(board->waiting(1) == 0);
  }

  SECTION("move assignment releases the target's previous hold")
  {
    auto a = board->hold_oom(0);
    auto b = board->hold_oom(1);
    CHECK(board->oom_pending(0) == 1);
    CHECK(board->oom_pending(1) == 1);
    b = std::move(a);
    CHECK(board->oom_pending(0) == 1);
    CHECK(board->oom_pending(1) == 0);
    CHECK(b.device() == 0);
    b = memory_wait_board::oom_token{};
    CHECK(board->oom_pending(0) == 0);
  }

  SECTION("self move assignment keeps the hold")
  {
    auto a  = board->enter_wait(0);
    auto& r = a;
    a       = std::move(r);
    CHECK(a.active());
    CHECK(board->waiting(0) == 1);
  }

  SECTION("a token moved through std::optional is counted once")
  {
    std::optional<memory_wait_board::oom_token> from = board->hold_oom(0);
    std::optional<memory_wait_board::oom_token> to;
    to = std::exchange(from, std::nullopt);
    CHECK_FALSE(from.has_value());
    CHECK(board->oom_pending(0) == 1);
    to.reset();
    CHECK(board->oom_pending(0) == 0);
  }

  SECTION("a default hold is inactive and releases nothing")
  {
    memory_wait_board::waiter w;
    CHECK_FALSE(w.active());
    auto live = board->enter_wait(0);
    w         = memory_wait_board::waiter{};
    CHECK(board->waiting(0) == 1);
  }
  CHECK(board->waiting(0) == 0);
  CHECK(board->waiting(1) == 0);
  CHECK(board->oom_pending(0) == 0);
  CHECK(board->oom_pending(1) == 0);
}

TEST_CASE("memory_wait_board ignores devices outside its range", "[memory_wait_board]")
{
  auto board = std::make_shared<memory_wait_board>(0);
  auto below = board->enter_wait(-1);
  auto above = board->hold_oom(1);
  CHECK_FALSE(below.active());
  CHECK_FALSE(above.active());
  CHECK(board->waiting(-1) == 0);
  CHECK(board->waiting(1) == 0);
  CHECK(board->oom_pending(1) == 0);
  CHECK(board->waiting(0) == 0);
  CHECK(board->oom_pending(0) == 0);

  auto empty = std::make_shared<memory_wait_board>(-1);
  CHECK(empty->device_count() == 0);
  auto none = empty->enter_wait(0);
  CHECK_FALSE(none.active());
  CHECK(empty->waiting(0) == 0);
}

TEST_CASE("memory_wait_board holds outlive the board's other owners", "[memory_wait_board]")
{
  auto board                                = std::make_shared<memory_wait_board>(0);
  std::weak_ptr<memory_wait_board> observer = board;
  auto token                                = board->hold_oom(0);
  auto waiter                               = board->enter_wait(0);
  board.reset();
  REQUIRE_FALSE(observer.expired());
  CHECK(observer.lock()->oom_pending(0) == 1);
  CHECK(observer.lock()->waiting(0) == 1);
  waiter = {};
  CHECK(observer.lock()->waiting(0) == 0);
  token = {};
  CHECK(observer.expired());
}

TEST_CASE("memory_wait_board requires shared ownership", "[memory_wait_board]")
{
  memory_wait_board unowned(0);
  CHECK_THROWS_AS(unowned.enter_wait(0), std::bad_weak_ptr);
  CHECK(unowned.waiting(0) == 0);
  // Out-of-range requests need no ownership.
  CHECK_FALSE(unowned.hold_oom(5).active());
}

TEST_CASE("memory_wait_board counts stay balanced under concurrent holds", "[memory_wait_board]")
{
  auto board            = std::make_shared<memory_wait_board>(1);
  constexpr int threads = 8;
  constexpr int rounds  = 2000;
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&board, t] {
      for (int i = 0; i < rounds; ++i) {
        auto w     = board->enter_wait(t % 2);
        auto tok   = board->hold_oom((t + 1) % 2);
        auto moved = std::move(w);
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  CHECK(board->waiting(0) == 0);
  CHECK(board->waiting(1) == 0);
  CHECK(board->oom_pending(0) == 0);
  CHECK(board->oom_pending(1) == 0);
}
