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

// Host-only coverage of the reservation livelock guard: the conservative
// satisfiability arithmetic and the unevictable-bytes provider registry. No
// GPU work — the fake memory_space pointers below are identity tokens only and
// are never dereferenced (providers key off the pointer value).

// test
#include <catch.hpp>

// sirius
#include <memory/pinned_reservation_guard.hpp>

// standard library
#include <cstddef>
#include <stdexcept>

using sirius::memory::max_satisfiable_reservation;
using sirius::memory::register_unevictable_bytes_provider;
using sirius::memory::reservation_is_unsatisfiable;
using sirius::memory::unevictable_pinned_bytes;
using sirius::memory::unregister_unevictable_bytes_provider;

namespace {

/// Distinct opaque memory_space identity tokens; never dereferenced.
const cucascade::memory::memory_space* fake_space(int slot)
{
  static int anchors[4] = {};
  return reinterpret_cast<const cucascade::memory::memory_space*>(&anchors[slot]);
}

/// RAII provider registration so a failing assertion cannot leak a provider
/// into later test cases (the registry is process-global).
struct scoped_provider {
  const void* owner;
  scoped_provider(const void* owner_token, sirius::memory::unevictable_bytes_provider fn)
    : owner(owner_token)
  {
    register_unevictable_bytes_provider(owner, std::move(fn));
  }
  ~scoped_provider() { unregister_unevictable_bytes_provider(owner); }
};

}  // namespace

TEST_CASE("max_satisfiable_reservation saturating arithmetic", "[pinned_reservation_guard]")
{
  // No pins: the whole limit is satisfiable.
  CHECK(max_satisfiable_reservation(100, 0) == 100);
  // Pins carve the limit down.
  CHECK(max_satisfiable_reservation(100, 30) == 70);
  // Pins at exactly the limit: nothing satisfiable.
  CHECK(max_satisfiable_reservation(100, 100) == 0);
  // Pins beyond the limit (over-subscribed pool): saturate at 0, no underflow.
  CHECK(max_satisfiable_reservation(100, 130) == 0);
  CHECK(max_satisfiable_reservation(0, 0) == 0);
}

TEST_CASE("reservation_is_unsatisfiable conservative boundaries", "[pinned_reservation_guard]")
{
  // Demand exactly at the satisfiable maximum is NOT unsatisfiable (the
  // check must never fire on transient pressure or exact fits).
  CHECK_FALSE(reservation_is_unsatisfiable(70, 100, 30));
  CHECK(reservation_is_unsatisfiable(71, 100, 30));
  // No pins: only demand beyond the limit itself is unsatisfiable (that case
  // is already prevented upstream by the executor's space-max clamp).
  CHECK_FALSE(reservation_is_unsatisfiable(100, 100, 0));
  CHECK(reservation_is_unsatisfiable(101, 100, 0));
  // Pins >= limit: any non-zero demand is unsatisfiable; zero demand is fine.
  CHECK(reservation_is_unsatisfiable(1, 100, 100));
  CHECK_FALSE(reservation_is_unsatisfiable(0, 100, 130));
}

TEST_CASE("unevictable-bytes provider registry", "[pinned_reservation_guard]")
{
  auto* space_a = fake_space(0);
  auto* space_b = fake_space(1);
  int owner_a   = 0;
  int owner_b   = 0;

  SECTION("no provider registered -> 0") { CHECK(unevictable_pinned_bytes(space_a) == 0); }

  SECTION("single provider answers per space; unregister restores 0")
  {
    {
      scoped_provider provider(&owner_a, [&](const cucascade::memory::memory_space* space) {
        return space == space_a ? std::size_t{4096} : std::size_t{0};
      });
      CHECK(unevictable_pinned_bytes(space_a) == 4096);
      CHECK(unevictable_pinned_bytes(space_b) == 0);
    }
    CHECK(unevictable_pinned_bytes(space_a) == 0);
  }

  SECTION("multiple providers sum")
  {
    scoped_provider first(&owner_a,
                          [](const cucascade::memory::memory_space*) { return std::size_t{10}; });
    scoped_provider second(&owner_b,
                           [](const cucascade::memory::memory_space*) { return std::size_t{32}; });
    CHECK(unevictable_pinned_bytes(space_a) == 42);
  }

  SECTION("re-registering the same owner replaces, not duplicates")
  {
    scoped_provider provider(
      &owner_a, [](const cucascade::memory::memory_space*) { return std::size_t{10}; });
    register_unevictable_bytes_provider(
      &owner_a, [](const cucascade::memory::memory_space*) { return std::size_t{7}; });
    CHECK(unevictable_pinned_bytes(space_a) == 7);
  }

  SECTION("throwing provider contributes 0 and does not poison others")
  {
    scoped_provider thrower(&owner_a, [](const cucascade::memory::memory_space*) -> std::size_t {
      throw std::runtime_error("provider failure");
    });
    scoped_provider healthy(&owner_b,
                            [](const cucascade::memory::memory_space*) { return std::size_t{5}; });
    CHECK(unevictable_pinned_bytes(space_a) == 5);
  }

  SECTION("unregistering an unknown owner is a no-op")
  {
    unregister_unevictable_bytes_provider(&owner_b);
    CHECK(unevictable_pinned_bytes(space_a) == 0);
  }
}

TEST_CASE("reservation_wait_scope registers and releases without waiting",
          "[pinned_reservation_guard]")
{
  // Constructing/destroying scopes must be safe and leave the watchdog parked
  // (its first report only fires after ~10 s, far beyond this test's scope
  // lifetimes, so the fake space pointer is never dereferenced).
  auto* space = fake_space(2);
  {
    sirius::memory::reservation_wait_scope outer(space, 1024, /*pipeline_id=*/1, /*task_id=*/2);
    sirius::memory::reservation_wait_scope inner(space, 2048, /*pipeline_id=*/1, /*task_id=*/3);
  }
  {
    sirius::memory::reservation_wait_scope again(space, 512, /*pipeline_id=*/9, /*task_id=*/9);
  }
  SUCCEED("scopes registered and released");
}
