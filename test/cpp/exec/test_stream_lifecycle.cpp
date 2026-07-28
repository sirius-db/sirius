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

// Pure CPU tests — no GPU required.

#include "catch.hpp"
#include "exec/stream_lifecycle.hpp"
#include "sirius/exception.hpp"

#include <atomic>
#include <chrono>
#include <set>
#include <thread>

using namespace sirius::exec;
using namespace std::chrono_literals;

using availability = stream_lifecycle::availability;

// ============================================================================
// LIFE-1: classify() truth table over (terminal?, repo_empty?)
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-1: classify truth table", "[stream_lifecycle]")
{
  stream_lifecycle life{{0}};

  // Open + empty: more may still arrive.
  REQUIRE(life.classify(true) == availability::WAITING);
  // Open + non-empty.
  REQUIRE(life.classify(false) == availability::HAS_DATA);

  life.mark_sender_done(0);
  REQUIRE(life.terminal());

  // Terminal but data still queued: HAS_DATA wins — invariant 2.
  REQUIRE(life.classify(false) == availability::HAS_DATA);
  // Terminal + empty: the only EOS state.
  REQUIRE(life.classify(true) == availability::END_OF_STREAM);
}

// ============================================================================
// LIFE-2: drained() is terminal && repo_empty
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-2: drained requires terminal and empty", "[stream_lifecycle]")
{
  stream_lifecycle life{{0}};

  REQUIRE_FALSE(life.drained(true));   // open + empty
  REQUIRE_FALSE(life.drained(false));  // open + data

  life.mark_sender_done(0);
  REQUIRE_FALSE(life.drained(false));  // terminal but not drained
  REQUIRE(life.drained(true));
}

// ============================================================================
// LIFE-3: fan-in EOS is by sender identity, not by count
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-3: repeated close from one sender cannot reach EOS",
          "[stream_lifecycle]")
{
  stream_lifecycle life{{0, 1}};

  life.mark_sender_done(0);
  REQUIRE_FALSE(life.terminal());

  // The bug a bare counter would have: two closes from sender 0 must NOT stand in for {0, 1}.
  life.mark_sender_done(0);
  life.mark_sender_done(0);
  REQUIRE_FALSE(life.terminal());
  REQUIRE(life.classify(true) == availability::WAITING);

  life.mark_sender_done(1);
  REQUIRE(life.terminal());
  REQUIRE(life.classify(true) == availability::END_OF_STREAM);
}

// ============================================================================
// LIFE-4: an unexpected sender id is a defined error, not a silent count
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-4: unexpected sender id is rejected", "[stream_lifecycle]")
{
  stream_lifecycle life{{0, 1}};

  REQUIRE_THROWS_AS(life.mark_sender_done(7), sirius::invalid_input_exception);

  // The rejected close left no trace: EOS still needs both real senders.
  REQUIRE_FALSE(life.terminal());
  life.mark_sender_done(0);
  life.mark_sender_done(1);
  REQUIRE(life.terminal());
}

// ============================================================================
// LIFE-5: no batch is admitted after EOS
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-5: admit is rejected once terminal", "[stream_lifecycle]")
{
  stream_lifecycle life{{0}};

  int inserted = 0;
  REQUIRE(life.admit([&] { ++inserted; }));
  REQUIRE(inserted == 1);

  life.mark_sender_done(0);

  // Rejected, and the caller's insert must not have run — otherwise a batch would land in the
  // repository after the consumer already observed END_OF_STREAM.
  REQUIRE_FALSE(life.admit([&] { ++inserted; }));
  REQUIRE(inserted == 1);
}

// ============================================================================
// LIFE-6: an empty expected set is terminal from construction
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-6: no expected senders means immediate EOS", "[stream_lifecycle]")
{
  stream_lifecycle life{{}};

  REQUIRE(life.terminal());
  REQUIRE(life.classify(true) == availability::END_OF_STREAM);
  REQUIRE(life.drained(true));
  REQUIRE_FALSE(life.admit([] {}));
}

// ============================================================================
// LIFE-7: the end-of-stream hook fires exactly once, on the last close
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-7: end-of-stream hook fires once on the last close",
          "[stream_lifecycle]")
{
  stream_lifecycle life{{0, 1}};

  int fired = 0;
  life.set_on_end_of_stream([&] { ++fired; });

  life.mark_sender_done(0);
  REQUIRE(fired == 0);  // fan-in not complete

  life.mark_sender_done(1);
  REQUIRE(fired == 1);

  // Repeat closes after terminal must not re-fire it.
  life.mark_sender_done(0);
  life.mark_sender_done(1);
  REQUIRE(fired == 1);
}

// ============================================================================
// LIFE-8: a hook registered AFTER EOS still fires (raced-close recheck)
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-8: hook registered after EOS fires immediately",
          "[stream_lifecycle]")
{
  stream_lifecycle life{{0}};
  life.mark_sender_done(0);

  int fired = 0;
  life.set_on_end_of_stream([&] { ++fired; });
  REQUIRE(fired == 1);
}

// ============================================================================
// LIFE-9: the waker is one-shot per arm, and fires on admit
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-9: waker fires once per arm", "[stream_lifecycle]")
{
  stream_lifecycle life{{0}};

  int woken = 0;
  REQUIRE(life.arm_waker([&] { ++woken; }, [] { return true; }));

  REQUIRE(life.admit([] {}));
  REQUIRE(woken == 1);

  // Not re-armed: a second push must not fire the stale waker.
  REQUIRE(life.admit([] {}));
  REQUIRE(woken == 1);

  REQUIRE(life.arm_waker([&] { ++woken; }, [] { return true; }));
  REQUIRE(life.admit([] {}));
  REQUIRE(woken == 2);
}

// ============================================================================
// LIFE-10: arm_waker declines when the caller is no longer starved
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-10: arm_waker respects its predicate", "[stream_lifecycle]")
{
  stream_lifecycle life{{0}};

  int woken = 0;
  // The predicate stands in for "the repository is still empty", re-checked under the lock. A
  // false answer means a concurrent push already landed, so the caller must re-classify rather
  // than park.
  REQUIRE_FALSE(life.arm_waker([&] { ++woken; }, [] { return false; }));

  REQUIRE(life.admit([] {}));
  REQUIRE(woken == 0);
}

// ============================================================================
// LIFE-11: the insert runs under the lock, before the waker observes it
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-11: admit registers before it wakes", "[stream_lifecycle]")
{
  stream_lifecycle life{{0}};

  bool registered      = false;
  bool seen_registered = false;
  REQUIRE(life.arm_waker([&] { seen_registered = registered; }, [] { return true; }));

  REQUIRE(life.admit([&] { registered = true; }));
  // #839 invariant 1: register-then-push. A waker that ran first would schedule a task for a
  // batch not yet in the repository.
  REQUIRE(seen_registered);
}

// ============================================================================
// LIFE-12: wait() unblocks on a push
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-12: wait unblocks when a batch is admitted", "[stream_lifecycle]")
{
  stream_lifecycle life{{0}};
  std::atomic<bool> repo_empty{true};
  std::atomic<bool> returned{false};

  std::thread consumer([&] {
    life.wait([&] { return repo_empty.load(); });
    returned = true;
  });

  std::this_thread::sleep_for(20ms);
  REQUIRE_FALSE(returned.load());

  REQUIRE(life.admit([&] { repo_empty = false; }));
  consumer.join();

  REQUIRE(returned.load());
  REQUIRE(life.classify(repo_empty.load()) == availability::HAS_DATA);
}

// ============================================================================
// LIFE-13: wait() unblocks on the final close of an empty stream
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-13: wait unblocks on end-of-stream", "[stream_lifecycle]")
{
  stream_lifecycle life{{0, 1}};
  std::atomic<bool> returned{false};

  std::thread consumer([&] {
    life.wait([] { return true; });  // repository stays empty throughout
    returned = true;
  });

  std::this_thread::sleep_for(20ms);
  life.mark_sender_done(0);
  std::this_thread::sleep_for(20ms);
  REQUIRE_FALSE(returned.load());  // fan-in incomplete: still WAITING

  life.mark_sender_done(1);
  consumer.join();

  REQUIRE(returned.load());
  REQUIRE(life.classify(true) == availability::END_OF_STREAM);
}

// ============================================================================
// LIFE-14: concurrent producers — every accepted push is registered exactly once
//          and nothing is accepted after EOS
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-14: concurrent admits and a racing close", "[stream_lifecycle]")
{
  constexpr int kPerThread = 500;
  stream_lifecycle life{{0, 1}};

  std::atomic<int> admitted{0};
  std::atomic<int> inserted{0};

  auto producer = [&](sender_id_t id) {
    for (int i = 0; i < kPerThread; ++i) {
      if (life.admit([&] { inserted.fetch_add(1, std::memory_order_relaxed); })) {
        admitted.fetch_add(1, std::memory_order_relaxed);
      }
    }
    life.mark_sender_done(id);
  };

  std::thread t0(producer, 0);
  std::thread t1(producer, 1);
  t0.join();
  t1.join();

  // Every admitted push ran its insert exactly once, and no insert ran without admission.
  REQUIRE(admitted.load() == inserted.load());
  REQUIRE(life.terminal());
  // After both senders closed, the stream is shut for good.
  REQUIRE_FALSE(life.admit([&] { inserted.fetch_add(1, std::memory_order_relaxed); }));
  REQUIRE(admitted.load() == inserted.load());
}

// ============================================================================
// LIFE-15: sender_closed() reports per-sender progress
// ============================================================================

TEST_CASE("stream_lifecycle LIFE-15: sender_closed tracks individual senders", "[stream_lifecycle]")
{
  stream_lifecycle life{{0, 1, 2}};

  REQUIRE_FALSE(life.sender_closed(0));
  life.mark_sender_done(1);
  REQUIRE(life.sender_closed(1));
  REQUIRE_FALSE(life.sender_closed(0));
  REQUIRE_FALSE(life.sender_closed(2));
  REQUIRE_FALSE(life.terminal());
}
