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

/*
 * Guards the two swappable third-party seams cucascade-io is built with (see
 * the CUCASCADE_* cache variables in the top-level CMakeLists.txt).
 *
 * Both seams are configured by compile definitions that cucascade propagates to
 * consumers. If a definition fails to reach this translation unit, the affected
 * cucascade header silently falls back to its standalone default — an in-tree
 * callable instead of absl::AnyInvocable, or the upstream `moodycamel`
 * namespace instead of duckdb's fork. Either would leave Sirius and
 * cucascade-io disagreeing about a type they pass across the library boundary,
 * which is an ODR violation the linker is under no obligation to diagnose.
 *
 * These are static_asserts, so a broken seam fails the build rather than the
 * test run. The TEST_CASE exists only to give the file a runtime presence.
 */

#include "catch.hpp"

#include <absl/functional/any_invocable.h>
#include <blockingconcurrentqueue.h>
#include <concurrentqueue.h>
#include <cucascade/exec/invocable.hpp>
#include <cucascade/io/concurrent_queue.hpp>

#include <type_traits>

// CUCASCADE_USE_ABSEIL_INVOCABLE + CUCASCADE_ABSEIL_INVOCABLE_TARGET: cucascade
// must resolve exec::invocable to the same abseil copy Sirius links, not to
// cucascade's in-tree fallback (which is also move-only but heap-allocates on
// every construction and rejects the &&/noexcept-qualified signatures Sirius
// spells).
static_assert(std::is_same_v<cucascade::exec::invocable<void()>, absl::AnyInvocable<void()>>,
              "cucascade::exec::invocable is not absl::AnyInvocable — the "
              "CUCASCADE_USE_ABSEIL_INVOCABLE seam did not reach this TU");

static_assert(
  std::is_same_v<cucascade::exec::invocable<int(double)>, absl::AnyInvocable<int(double)>>,
  "cucascade::exec::invocable does not forward its signature to absl::AnyInvocable");

// CUCASCADE_MOODYCAMEL_INCLUDE_DIR + CUCASCADE_MOODYCAMEL_NAMESPACE: cucascade's
// queue aliases must name duckdb's vendored fork, which renames the namespace to
// duckdb_moodycamel. Sirius's own headers reach these types through duckdb's
// global include path, so the two sides must agree.
static_assert(std::is_same_v<cucascade::io::blocking_concurrent_queue<int>,
                             duckdb_moodycamel::BlockingConcurrentQueue<int>>,
              "cucascade::io::blocking_concurrent_queue is not duckdb's "
              "moodycamel fork — check CUCASCADE_MOODYCAMEL_NAMESPACE");

static_assert(
  std::is_same_v<cucascade::io::concurrent_queue<int>, duckdb_moodycamel::ConcurrentQueue<int>>,
  "cucascade::io::concurrent_queue is not duckdb's moodycamel fork");

TEST_CASE("cucascade third-party seams are wired to Sirius's copies", "[cucascade_abi]")
{
  // The real assertions are the static_asserts above; this confirms the aliased
  // types are usable across the boundary rather than merely name-equal.
  cucascade::exec::invocable<int()> fn = [] { return 7; };
  CHECK(fn() == 7);

  cucascade::io::concurrent_queue<int> queue;
  queue.enqueue(42);
  int dequeued = 0;
  CHECK(queue.try_dequeue(dequeued));
  CHECK(dequeued == 42);
}
