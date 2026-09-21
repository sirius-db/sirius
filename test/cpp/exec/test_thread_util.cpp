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
#include "exec/thread_util.hpp"

// The thread-attribute knobs sit under privileged syscalls that can plausibly
// refuse a caller on some hosts (containers, restricted policies), so these
// assertions cover the failure-clean interface --- truncation, empty-list
// no-ops, out-of-range rejection --- rather than side-effects that depend on
// what the kernel is willing to let a non-root process do.

#include <sched.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <string_view>
#include <thread>
#include <vector>

using namespace sirius::exec::thread_util;

TEST_CASE("set_current_thread_name truncates to what the kernel accepts", "[exec][thread_util]")
{
  // A name longer than max_thread_name_len used to be silently kept as the
  // old one; the wrapper truncates up front so the kernel is never asked for
  // more than it will take.
  REQUIRE(set_current_thread_name("thread_util_test"));
  auto const seen = get_current_thread_name();
  REQUIRE_FALSE(seen.empty());
  CHECK(seen.size() <= max_thread_name_len);
  // Prefix matches the original request, so callers can search for the name
  // they asked for even after truncation.
  CHECK(seen == std::string_view{"thread_util_test"}.substr(0, seen.size()));
}

TEST_CASE("set_thread_name works against a live std::thread", "[exec][thread_util]")
{
  std::atomic<bool> proceed{false};
  std::atomic<bool> done{false};
  std::thread t([&] {
    while (!proceed.load()) {
      std::this_thread::yield();
    }
    done.store(true);
  });

  REQUIRE(set_thread_name(t, "tu_worker"));
  CHECK(get_thread_name(t.native_handle()) == "tu_worker");

  proceed.store(true);
  t.join();
  REQUIRE(done.load());
}

TEST_CASE("affinity accepts an empty list as a no-op", "[exec][thread_util]")
{
  // Empty is deliberately not "any CPU": callers who want a widening reset
  // should compute the id list themselves.  The wrapper reports success and
  // touches nothing.
  auto const before = get_current_thread_affinity();
  REQUIRE(set_current_thread_affinity({}));
  CHECK(get_current_thread_affinity() == before);
}

TEST_CASE("affinity round-trips through the first allowed CPU", "[exec][thread_util]")
{
  auto const before = get_current_thread_affinity();
  REQUIRE_FALSE(before.empty());
  // Pin to the first CPU the caller was already allowed on; asking for one it
  // was denied would fail regardless of the wrapper's own behaviour, which is
  // what this test is about.
  std::array<int, 1> single{before.front()};
  REQUIRE(set_current_thread_affinity(single));
  auto const pinned = get_current_thread_affinity();
  REQUIRE(pinned.size() == 1);
  CHECK(pinned.front() == single.front());
  // Reset so this test does not narrow the CPUs a later test can see.
  REQUIRE(set_current_thread_affinity(before));
}

TEST_CASE("affinity silently drops out-of-range CPU ids", "[exec][thread_util]")
{
  auto const before = get_current_thread_affinity();
  REQUIRE_FALSE(before.empty());
  // An id list with garbage on either side of a legal id must still land the
  // legal one, and must not fail: the wrapper is deliberately forgiving so a
  // caller does not have to prevalidate against CPU_SETSIZE.
  std::vector<int> ids{-1, before.front(), CPU_SETSIZE + 100};
  REQUIRE(set_current_thread_affinity(ids));
  auto const pinned = get_current_thread_affinity();
  REQUIRE(pinned.size() == 1);
  CHECK(pinned.front() == before.front());
  REQUIRE(set_current_thread_affinity(before));
}

TEST_CASE("scheduling policy rejects an out-of-range priority", "[exec][thread_util]")
{
  // SCHED_OTHER's only legal priority is 0; a caller who requests a real-time
  // level under it is a bug the wrapper catches before the syscall.
  CHECK_FALSE(set_current_thread_scheduling(scheduling_policy::other, 10));
  CHECK_FALSE(set_current_thread_scheduling(scheduling_policy::batch, 1));
  // And a real-time policy without a real-time priority is the same class of
  // mistake in the other direction.
  CHECK_FALSE(set_current_thread_scheduling(scheduling_policy::fifo, 0));
  CHECK_FALSE(set_current_thread_scheduling(scheduling_policy::round_robin, 100));
}

TEST_CASE("scheduling reads the calling thread's live policy", "[exec][thread_util]")
{
  // Whatever the process was started with, reading it back should succeed and
  // land on a known policy.
  auto const info = get_current_thread_scheduling();
  REQUIRE(info.has_value());
  CHECK((info->policy == scheduling_policy::other || info->policy == scheduling_policy::batch ||
         info->policy == scheduling_policy::idle || info->policy == scheduling_policy::fifo ||
         info->policy == scheduling_policy::round_robin));
}

TEST_CASE("niceness rejects values outside [-20, 19]", "[exec][thread_util]")
{
  CHECK_FALSE(set_current_thread_niceness(-21));
  CHECK_FALSE(set_current_thread_niceness(20));
}

TEST_CASE("niceness reads a value even without changing it", "[exec][thread_util]")
{
  auto const nv = get_current_thread_niceness();
  REQUIRE(nv.has_value());
  CHECK(*nv >= -20);
  CHECK(*nv <= 19);
}

TEST_CASE("current_thread_id names a live kernel TID", "[exec][thread_util]")
{
  auto const self_tid = current_thread_id();
  CHECK(self_tid != 0);

  // A different thread sees a different TID; the id is per-thread, not
  // per-process.
  std::uint64_t other_tid{0};
  std::thread t([&] { other_tid = current_thread_id(); });
  t.join();
  CHECK(other_tid != 0);
  CHECK(other_tid != self_tid);
}
