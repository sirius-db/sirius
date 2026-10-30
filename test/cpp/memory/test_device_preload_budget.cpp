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
#include "memory/device_preload_budget.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

using sirius::memory::device_preload_budget;

namespace {

std::array<int const, 2> const two_gpus{0, 1};

}  // namespace

TEST_CASE("device_preload_budget charges and releases a single generation",
          "[device_preload_budget]")
{
  device_preload_budget board{two_gpus};
  CHECK(board.empty());
  CHECK(board.outstanding(0) == 0);

  board.watch(7);
  CHECK_FALSE(board.empty());
  CHECK(board.watched_count() == 1);

  REQUIRE(board.charge(7, /*job_seq=*/1, /*device=*/0, /*speculative=*/false, 100));
  CHECK(board.outstanding(0) == 100);
  CHECK(board.outstanding_speculative(0) == 0);

  SECTION("release of the same generation subtracts the charge; a wrong generation does not")
  {
    board.release(7, /*job_seq=*/2);  // wrong generation: no-op
    CHECK(board.outstanding(0) == 100);
    board.release(7, /*job_seq=*/1);
    CHECK(board.outstanding(0) == 0);
    board.release(7, /*job_seq=*/1);  // idempotent
    CHECK(board.outstanding(0) == 0);
  }
  SECTION("unwatch drops the entry and its charge")
  {
    board.unwatch(7);
    CHECK(board.outstanding(0) == 0);
    CHECK(board.empty());
  }
}

TEST_CASE("device_preload_budget tracks the speculative part and reclassifies it",
          "[device_preload_budget]")
{
  device_preload_budget board{two_gpus};
  board.watch(3);
  REQUIRE(board.charge(3, 1, 1, /*speculative=*/true, 64));
  CHECK(board.outstanding(1) == 64);
  CHECK(board.outstanding_speculative(1) == 64);

  board.reclassify_committed(3, 1);  // B -> A: drop the speculative part only
  CHECK(board.outstanding(1) == 64);
  CHECK(board.outstanding_speculative(1) == 0);

  board.release(3, 1);
  CHECK(board.outstanding(1) == 0);
}

TEST_CASE("device_preload_budget claim before charge refuses the charge", "[device_preload_budget]")
{
  device_preload_budget board{two_gpus};
  board.watch(5);
  board.claim(std::array<std::uint64_t, 1>{5});    // claimed before any charge
  CHECK(board.claimed_releases() == 0);            // nothing was charged yet
  CHECK_FALSE(board.charge(5, 1, 0, false, 100));  // refused: the task owns it
  CHECK(board.outstanding(0) == 0);
}

TEST_CASE("device_preload_budget claim during a copy releases the charge",
          "[device_preload_budget]")
{
  device_preload_budget board{two_gpus};
  board.watch(5);
  REQUIRE(board.charge(5, 1, 0, false, 100));  // the copy is running
  board.claim(std::array<std::uint64_t, 1>{5});
  CHECK(board.outstanding(0) == 0);
  CHECK(board.claimed_releases() == 1);
  CHECK(board.claimed_bytes_released() == 100);
  // The landing path's release is then a no-op.
  board.release(5, 1);
  CHECK(board.outstanding(0) == 0);
}

TEST_CASE("device_preload_budget claim after landing releases a persisted charge",
          "[device_preload_budget]")
{
  device_preload_budget board{two_gpus};
  board.watch(5);
  REQUIRE(board.charge(5, 1, 0, false, 100));
  // The promotion landed; the ledger keeps the charge until the task is claimed.
  board.claim(std::array<std::uint64_t, 1>{5});
  CHECK(board.outstanding(0) == 0);
  CHECK(board.claimed_releases() == 1);
}

TEST_CASE("device_preload_budget starts a new generation after a re-downgrade",
          "[device_preload_budget]")
{
  device_preload_budget board{two_gpus};
  board.watch(5);
  REQUIRE(board.charge(5, 1, 0, false, 100));
  board.claim(std::array<std::uint64_t, 1>{5});  // claimed, charge released
  CHECK(board.outstanding(0) == 0);

  board.watch(5);                             // re-downgraded: a fresh generation
  REQUIRE(board.charge(5, 2, 0, false, 80));  // claimed flag was cleared by watch
  CHECK(board.outstanding(0) == 80);
  // An old-generation release never touches the new charge.
  board.release(5, 1);
  CHECK(board.outstanding(0) == 80);
  board.release(5, 2);
  CHECK(board.outstanding(0) == 0);
}

TEST_CASE("device_preload_budget claim releases whatever device holds the charge",
          "[device_preload_budget]")
{
  device_preload_budget board{two_gpus};
  board.watch(5);
  REQUIRE(board.charge(5, 1, /*device=*/1, false, 100));  // charged to GPU 1
  board.claim(std::array<std::uint64_t, 1>{5});           // claimed on GPU 0 (no exception)
  CHECK(board.outstanding(1) == 0);
  CHECK(board.claimed_releases() == 1);
}

TEST_CASE("device_preload_budget handles shared and duplicate ids in one claim",
          "[device_preload_budget]")
{
  device_preload_budget board{two_gpus};
  board.watch(5);
  board.watch(6);
  REQUIRE(board.charge(5, 1, 0, false, 10));
  REQUIRE(board.charge(6, 1, 0, false, 20));
  // A task with a duplicated input id and an unknown id: the first claim releases, later no-ops.
  board.claim(std::array<std::uint64_t, 4>{5, 5, 6, 999});
  CHECK(board.outstanding(0) == 0);
  CHECK(board.claimed_releases() == 2);  // 5 and 6 once each
  CHECK(board.claimed_bytes_released() == 30);
}

TEST_CASE("device_preload_budget never underflows and ignores out-of-range devices",
          "[device_preload_budget]")
{
  device_preload_budget board{two_gpus};
  board.watch(5);
  REQUIRE(board.charge(5, 1, 0, false, 100));
  board.release(5, 1);
  board.release(5, 1);  // double release: no underflow
  CHECK(board.outstanding(0) == 0);

  // A device outside the board's range is ignored; its total reads as zero.
  board.watch(6);
  CHECK(board.charge(6, 1, /*device=*/0, false, 0));  // zero-byte charge succeeds
  CHECK(board.outstanding(9) == 0);                   // out-of-range device
}

TEST_CASE("device_preload_budget links concurrent claim and charge", "[device_preload_budget]")
{
  // Many rounds: a claim and a charge race on the same id; the board linearizes them, so the id
  // is either charged-then-released or never charged, and the total always returns to zero.
  for (int round = 0; round < 500; ++round) {
    device_preload_budget board{two_gpus};
    board.watch(1);
    std::atomic<bool> go{false};
    std::thread charger{[&] {
      while (!go.load()) {}
      if (board.charge(1, 1, 0, false, 100)) {
        // If the charge won the race, release it to mimic the landing path.
        board.release(1, 1);
      }
    }};
    std::thread claimer{[&] {
      while (!go.load()) {}
      board.claim(std::array<std::uint64_t, 1>{1});
    }};
    go.store(true);
    charger.join();
    claimer.join();
    CHECK(board.outstanding(0) == 0);
  }
}
