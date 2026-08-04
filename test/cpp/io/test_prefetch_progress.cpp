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

// progress_from and combine_prefetch_progress carry 100% of the prefetch state-model decision
// making: both are pure and total, which is what lets the whole model be tested with no I/O, no
// GPU and no mock ioctx. Everything left on the real types is trivial forwarding.

#include <catch.hpp>
#include <io/cache/types.hpp>

#include <algorithm>
#include <vector>

namespace {

using sirius::io::cache::entry_state;
using sirius::io::cache::prefetch_progress;

/// combine_prefetch_progress takes a span; every case here spells its input as a list of the
/// per-datasource progresses one split folded over.
prefetch_progress fold(const std::vector<prefetch_progress>& parts)
{
  return sirius::io::cache::combine_prefetch_progress(parts);
}

}  // namespace

TEST_CASE("progress_from maps every entry_state onto a prefetch_progress",
          "[io][prefetch_api][prefetch_progress]")
{
  using sirius::io::cache::progress_from;

  // queued and allocated collapse into prepared deliberately: the consumer cannot act on the
  // difference, and allocated is also where an IO failure reverts to.
  STATIC_REQUIRE(progress_from(entry_state::empty, false) == prefetch_progress::prepared);
  STATIC_REQUIRE(progress_from(entry_state::queued, false) == prefetch_progress::prepared);
  STATIC_REQUIRE(progress_from(entry_state::allocated, false) == prefetch_progress::prepared);
  STATIC_REQUIRE(progress_from(entry_state::loading, false) == prefetch_progress::loading);
  STATIC_REQUIRE(progress_from(entry_state::cached, false) == prefetch_progress::cached);
  STATIC_REQUIRE(progress_from(entry_state::in_use, false) == prefetch_progress::in_use);
  STATIC_REQUIRE(progress_from(entry_state::evicting, false) == prefetch_progress::evicting);
}

TEST_CASE("progress_from reports cancelled regardless of entry state",
          "[io][prefetch_api][prefetch_progress]")
{
  using sirius::io::cache::progress_from;

  // Consumer intent dominates: once cancelled, nothing is driving the entry_state to completion.
  STATIC_REQUIRE(progress_from(entry_state::empty, true) == prefetch_progress::cancelled);
  STATIC_REQUIRE(progress_from(entry_state::queued, true) == prefetch_progress::cancelled);
  STATIC_REQUIRE(progress_from(entry_state::allocated, true) == prefetch_progress::cancelled);
  STATIC_REQUIRE(progress_from(entry_state::loading, true) == prefetch_progress::cancelled);
  STATIC_REQUIRE(progress_from(entry_state::cached, true) == prefetch_progress::cancelled);
  STATIC_REQUIRE(progress_from(entry_state::in_use, true) == prefetch_progress::cancelled);
  STATIC_REQUIRE(progress_from(entry_state::evicting, true) == prefetch_progress::cancelled);
}

TEST_CASE("combine_prefetch_progress folds a multi-file split",
          "[io][prefetch_api][prefetch_progress]")
{
  SECTION("an empty handle list is empty") { CHECK(fold({}) == prefetch_progress::empty); }

  SECTION("one loading file makes the whole split loading")
  {
    CHECK(
      fold({prefetch_progress::cached, prefetch_progress::loading, prefetch_progress::cached}) ==
      prefetch_progress::loading);
    CHECK(fold({prefetch_progress::prepared, prefetch_progress::loading}) ==
          prefetch_progress::loading);
  }

  SECTION("a split is cached only when every file is")
  {
    CHECK(fold({prefetch_progress::cached, prefetch_progress::cached}) ==
          prefetch_progress::cached);
    CHECK(fold({prefetch_progress::cached, prefetch_progress::in_use}) ==
          prefetch_progress::cached);
    CHECK(fold({prefetch_progress::cached, prefetch_progress::prepared}) ==
          prefetch_progress::prepared);
  }

  SECTION("cancelled loses to prepared")
  {
    CHECK(fold({prefetch_progress::cancelled, prefetch_progress::prepared}) ==
          prefetch_progress::prepared);
  }

  SECTION("all cancelled is cancelled")
  {
    CHECK(fold({prefetch_progress::cancelled, prefetch_progress::cancelled}) ==
          prefetch_progress::cancelled);
  }

  SECTION("the fold is order independent")
  {
    // split_connector reads this value per queued split and never controls the order its
    // datasources were visited in, so every permutation of one multiset must agree.
    std::vector<prefetch_progress> parts{prefetch_progress::prepared,
                                         prefetch_progress::prepared,
                                         prefetch_progress::loading,
                                         prefetch_progress::cached};
    std::sort(parts.begin(), parts.end());
    do {
      CHECK(fold(parts) == prefetch_progress::loading);
    } while (std::next_permutation(parts.begin(), parts.end()));
  }
}
