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

// Unit tests for the pure decision helpers that drive the STANDARD / MIXED_JOIN
// partial-barrier hash-join path (batches arrive progressively on each side):
//   - next_cross_schedule_pair():    finds/claims the next per-partition
//                                    (probe, build) cross-product pair.
//   - collect_cross_schedule_discards(): frees batches once fully consumed.
//   - peek_cross_schedule_kind():    non-mutating hint classification.
//
// These are GPU-free and pipeline-free, so they exercise the highest-risk logic
// (progressive discovery, the exact pop-when-safe rule, completion) determin-
// istically. End-to-end multi-batch streaming is covered by the [mgpu] tests,
// which also exercise refresh_cross_schedule's non-INNER whole-side guard.

#include "op/sirius_physical_hash_join.hpp"

#include <catch.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

using sirius::op::collect_cross_schedule_discards;
using sirius::op::cross_schedule_discard;
using sirius::op::cross_schedule_kind;
using sirius::op::next_cross_schedule_pair;
using sirius::op::partition_cross_schedule;
using sirius::op::peek_cross_schedule_kind;

namespace {

// Build one partition's schedule pre-populated with the given probe/build batch IDs, mirroring what
// refresh_cross_schedule would have discovered (paired counts start at zero).
partition_cross_schedule make_partition(std::vector<uint64_t> const& probe_ids,
                                        std::vector<uint64_t> const& build_ids)
{
  partition_cross_schedule c;
  for (uint64_t id : probe_ids) {
    c.probe_id_seen.insert(id);
    c.probe_ids.push_back(id);
    c.probe_paired_count.push_back(0);
  }
  for (uint64_t id : build_ids) {
    c.build_id_seen.insert(id);
    c.build_ids.push_back(id);
    c.build_paired_count.push_back(0);
  }
  return c;
}

// Drain every schedulable pair, returning the emitted (partition, probe_idx, build_idx) triples.
std::vector<std::array<std::size_t, 3>> drain_all_pairs(
  std::vector<partition_cross_schedule>& cross, bool probe_finished, bool build_finished)
{
  std::vector<std::array<std::size_t, 3>> out;
  for (;;) {
    auto step = next_cross_schedule_pair(cross, probe_finished, build_finished);
    if (step.kind != cross_schedule_kind::emit_pair) { break; }
    out.push_back({step.partition, step.probe_idx, step.build_idx});
  }
  return out;
}

bool has_discard(std::vector<cross_schedule_discard> const& ds,
                 std::size_t partition,
                 bool is_build,
                 uint64_t batch_id)
{
  return std::any_of(ds.begin(), ds.end(), [&](cross_schedule_discard const& d) {
    return d.partition == partition && d.is_build == is_build && d.batch_id == batch_id;
  });
}

}  // namespace

//===----------------------------------------------------------------------===//
// next_cross_schedule_pair — enumeration & completion
//===----------------------------------------------------------------------===//

TEST_CASE("cross schedule enumerates the full per-partition grid once", "[cross_schedule]")
{
  // LEFT shape: build folded to one batch, probe streams three batches.
  std::vector<partition_cross_schedule> cross{make_partition({10, 11, 12}, {20})};

  auto pairs = drain_all_pairs(cross, /*probe_finished=*/false, /*build_finished=*/false);
  REQUIRE(pairs.size() == 3);
  // Each probe index paired exactly once against the single build index 0.
  REQUIRE(pairs == std::vector<std::array<std::size_t, 3>>{{0, 0, 0}, {0, 1, 0}, {0, 2, 0}});
  REQUIRE(cross[0].probe_paired_count == std::vector<uint32_t>{1, 1, 1});
  REQUIRE(cross[0].build_paired_count == std::vector<uint32_t>{3});

  // No pair remains; build producer not finished -> wait on build.
  auto again = next_cross_schedule_pair(cross, false, false);
  REQUIRE(again.kind == cross_schedule_kind::wait_build);
}

TEST_CASE("cross schedule produces the INNER cross product over both streaming sides",
          "[cross_schedule]")
{
  std::vector<partition_cross_schedule> cross{make_partition({10, 11}, {20, 21})};
  auto pairs = drain_all_pairs(cross, false, false);
  REQUIRE(pairs.size() == 4);
  REQUIRE(cross[0].probe_paired_count == std::vector<uint32_t>{2, 2});
  REQUIRE(cross[0].build_paired_count == std::vector<uint32_t>{2, 2});
}

TEST_CASE("cross schedule waits on the still-running producer, done only when both finish",
          "[cross_schedule]")
{
  std::vector<partition_cross_schedule> cross{make_partition({10}, {20})};
  (void)drain_all_pairs(cross, false, false);  // schedule the lone pair

  // Build still producing -> wait_build; then probe still producing -> wait_probe; both done ->
  // done.
  REQUIRE(next_cross_schedule_pair(cross, false, false).kind == cross_schedule_kind::wait_build);
  REQUIRE(next_cross_schedule_pair(cross, false, true).kind == cross_schedule_kind::wait_probe);
  REQUIRE(next_cross_schedule_pair(cross, true, true).kind == cross_schedule_kind::done);
}

TEST_CASE("cross schedule discovers newly-arrived batches on later calls", "[cross_schedule]")
{
  std::vector<partition_cross_schedule> cross{make_partition({10}, {20})};
  REQUIRE(next_cross_schedule_pair(cross, false, false).kind == cross_schedule_kind::emit_pair);
  REQUIRE(next_cross_schedule_pair(cross, false, false).kind != cross_schedule_kind::emit_pair);

  // A new probe batch arrives (as refresh_cross_schedule would append it).
  cross[0].probe_id_seen.insert(11);
  cross[0].probe_ids.push_back(11);
  cross[0].probe_paired_count.push_back(0);

  auto step = next_cross_schedule_pair(cross, false, false);
  REQUIRE(step.kind == cross_schedule_kind::emit_pair);
  REQUIRE(step.probe_idx == 1);
  REQUIRE(step.build_idx == 0);
  // The already-scheduled (0,0) pair is not re-emitted.
  REQUIRE(next_cross_schedule_pair(cross, false, false).kind != cross_schedule_kind::emit_pair);
}

//===----------------------------------------------------------------------===//
// peek_cross_schedule_kind — non-mutating hint classification
//===----------------------------------------------------------------------===//

TEST_CASE("peek reports emit_pair while any pair is unscheduled, even after both finish",
          "[cross_schedule]")
{
  std::vector<partition_cross_schedule> cross{make_partition({10}, {20})};
  // Unscheduled pair must be emitted (drained) before the operator can be done.
  REQUIRE(peek_cross_schedule_kind(cross, true, true) == cross_schedule_kind::emit_pair);
  (void)next_cross_schedule_pair(cross, true, true);
  REQUIRE(peek_cross_schedule_kind(cross, true, true) == cross_schedule_kind::done);
  REQUIRE(peek_cross_schedule_kind(cross, false, true) == cross_schedule_kind::wait_probe);
  REQUIRE(peek_cross_schedule_kind(cross, true, false) == cross_schedule_kind::wait_build);
}

//===----------------------------------------------------------------------===//
// collect_cross_schedule_discards — the exact pop-when-safe rule
//===----------------------------------------------------------------------===//

TEST_CASE("discards free a probe batch only after build finishes and it is fully paired",
          "[cross_schedule]")
{
  // LEFT shape: probe {10,11,12} x build {20}.
  std::vector<partition_cross_schedule> cross{make_partition({10, 11, 12}, {20})};
  (void)drain_all_pairs(cross, false, false);

  // Nothing frees while producers still run.
  REQUIRE(collect_cross_schedule_discards(cross, false, false).empty());

  // Build finished: every fully-paired probe batch frees; the build batch stays (probe not done).
  auto d1 =
    collect_cross_schedule_discards(cross, /*probe_finished=*/false, /*build_finished=*/true);
  REQUIRE(d1.size() == 3);
  REQUIRE(has_discard(d1, 0, /*is_build=*/false, 10));
  REQUIRE(has_discard(d1, 0, false, 11));
  REQUIRE(has_discard(d1, 0, false, 12));
  // Idempotent: already-popped probe batches are not returned again.
  REQUIRE(collect_cross_schedule_discards(cross, false, true).empty());

  // Probe finished too: the folded build batch (paired with all 3 probes) frees.
  auto d2 =
    collect_cross_schedule_discards(cross, /*probe_finished=*/true, /*build_finished=*/true);
  REQUIRE(d2.size() == 1);
  REQUIRE(has_discard(d2, 0, /*is_build=*/true, 20));
}

TEST_CASE("discards never free a batch while the opposite producer is still open",
          "[cross_schedule]")
{
  std::vector<partition_cross_schedule> cross{make_partition({10}, {20})};
  (void)drain_all_pairs(cross, false, false);

  // Probe finished but build still open: the build batch frees (paired with all probes), but the
  // probe batch must NOT free (a new build batch could still pair with it).
  auto d =
    collect_cross_schedule_discards(cross, /*probe_finished=*/true, /*build_finished=*/false);
  REQUIRE(d.size() == 1);
  REQUIRE(has_discard(d, 0, /*is_build=*/true, 20));
  REQUIRE_FALSE(has_discard(d, 0, /*is_build=*/false, 10));
}

TEST_CASE("discards skip the empty-opposite-side case (legacy draining behavior)",
          "[cross_schedule]")
{
  // A probe batch with no build batches at all: no pair exists, and it is never freed by the sweep
  // (matches the legacy grid, which likewise could not pair or pop it).
  std::vector<partition_cross_schedule> cross{make_partition({10}, {})};
  REQUIRE(next_cross_schedule_pair(cross, true, true).kind == cross_schedule_kind::done);
  REQUIRE(collect_cross_schedule_discards(cross, true, true).empty());
}
