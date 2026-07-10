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

/**
 * @file test_single_assignment.cpp
 * @brief Contract tests for sirius::single_assignment (C1a-2): the write-once slot behind the
 *        dynamic-filter freeze seam. The load-bearing guarantees are (a) the commit path is
 *        statically noexcept, (b) an uncommitted token rolls its slot back so a failed multi-slot
 *        preparation changes nothing, and (c) reading an unassigned slot is a loud internal
 *        error, never a silent default.
 */

#include <catch.hpp>
#include <sirius/exception.hpp>
#include <sirius/single_assignment.hpp>

#include <atomic>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

/// The production payload shape (the frozen dynamic-filter plan is published as a shared_ptr).
using payload   = std::shared_ptr<int const>;
using slot_type = sirius::single_assignment<payload>;
using token     = slot_type::assignment_token;

payload make_value(int v) { return std::make_shared<int const>(v); }

}  // namespace

//===-----------------------------------------------------------------------------------------===//
// Compile-time contracts
//===-----------------------------------------------------------------------------------------===//

// The commit path must be statically non-throwing (the plan's allocation-free/noexcept commit).
static_assert(noexcept(std::declval<slot_type&>().commit_assignment(std::declval<token&&>())),
              "commit_assignment must be statically noexcept");

// One slot, one address: no copies of the slot; tokens are move-constructible only.
static_assert(!std::is_copy_constructible_v<slot_type>);
static_assert(!std::is_copy_assignable_v<slot_type>);
static_assert(!std::is_copy_constructible_v<token>);
static_assert(!std::is_copy_assignable_v<token>);
static_assert(std::is_move_constructible_v<token>);
static_assert(!std::is_move_assignable_v<token>);

//===-----------------------------------------------------------------------------------------===//
// Single-slot protocol
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("a committed two-phase assignment publishes the value", "[sirius][single_assignment]")
{
  slot_type slot;
  REQUIRE_FALSE(slot.is_assigned());

  auto tok = slot.prepare_assignment(make_value(42));
  REQUIRE_FALSE(slot.is_assigned());  // pending is not assigned: no reader may see it yet

  slot.commit_assignment(std::move(tok));
  REQUIRE(slot.is_assigned());
  REQUIRE(*slot.get() == 42);
}

TEST_CASE("reading an unassigned slot is an internal error", "[sirius][single_assignment]")
{
  slot_type slot;
  REQUIRE_THROWS_AS(slot.get(), sirius::internal_exception);

  // Still unreadable while merely pending: prepared is not published.
  auto tok = slot.prepare_assignment(make_value(1));
  REQUIRE_THROWS_AS(slot.get(), sirius::internal_exception);
  slot.commit_assignment(std::move(tok));
  REQUIRE(*slot.get() == 1);
}

TEST_CASE("a second prepare on a pending or assigned slot throws", "[sirius][single_assignment]")
{
  slot_type slot;
  auto tok = slot.prepare_assignment(make_value(1));
  REQUIRE_THROWS_AS(slot.prepare_assignment(make_value(2)), sirius::internal_exception);

  slot.commit_assignment(std::move(tok));
  REQUIRE_THROWS_AS(slot.prepare_assignment(make_value(3)), sirius::internal_exception);
  REQUIRE(*slot.get() == 1);  // the committed value is untouched by the failed prepares
}

TEST_CASE("destroying an uncommitted token rolls the slot back", "[sirius][single_assignment]")
{
  // The zero-slots-changed guarantee: a multi-slot preparation that throws midway destroys its
  // already-minted tokens, and every touched slot must return to empty — reusable, unreadable.
  slot_type slot;
  {
    auto tok = slot.prepare_assignment(make_value(7));
    // token destroyed uncommitted at scope end
  }
  REQUIRE_FALSE(slot.is_assigned());
  REQUIRE_THROWS_AS(slot.get(), sirius::internal_exception);

  // The rolled-back slot accepts a fresh preparation.
  auto tok = slot.prepare_assignment(make_value(8));
  slot.commit_assignment(std::move(tok));
  REQUIRE(*slot.get() == 8);
}

TEST_CASE("the checked one-shot assign is single-use", "[sirius][single_assignment]")
{
  slot_type slot;
  slot.assign(make_value(5));
  REQUIRE(*slot.get() == 5);
  REQUIRE_THROWS_AS(slot.assign(make_value(6)), sirius::internal_exception);
  REQUIRE(*slot.get() == 5);
}

TEST_CASE("a moved-from token neither commits nor rolls back", "[sirius][single_assignment]")
{
  slot_type slot;
  auto original = slot.prepare_assignment(make_value(9));
  auto moved    = std::move(original);

  slot.commit_assignment(std::move(moved));
  REQUIRE(slot.is_assigned());
  // `original` (moved-from) and `moved` (consumed by commit) both go out of scope here; neither
  // destructor may roll the assigned slot back.
}

TEST_CASE("token destruction after commit is a no-op", "[sirius][single_assignment]")
{
  slot_type slot;
  {
    auto tok = slot.prepare_assignment(make_value(3));
    slot.commit_assignment(std::move(tok));
    // consumed token destroyed at scope end
  }
  REQUIRE(slot.is_assigned());
  REQUIRE(*slot.get() == 3);
}

//===-----------------------------------------------------------------------------------------===//
// Publication visibility
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("concurrent readers observe the committed value exactly", "[sirius][single_assignment]")
{
  // The release-store in commit / acquire-load in is_assigned() must guarantee a reader that
  // observes the slot assigned sees the fully-constructed payload (the publisher-task pattern).
  slot_type slot;
  constexpr int reader_count = 4;
  std::atomic<int> correct{0};

  std::vector<std::thread> readers;
  readers.reserve(reader_count);
  for (int i = 0; i < reader_count; ++i) {
    readers.emplace_back([&] {
      while (!slot.is_assigned()) {
        std::this_thread::yield();
      }
      if (*slot.get() == 42) { correct.fetch_add(1, std::memory_order_relaxed); }
    });
  }

  slot.assign(make_value(42));
  for (auto& reader : readers) {
    reader.join();
  }
  REQUIRE(correct.load() == reader_count);
}
