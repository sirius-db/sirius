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

// Unit tests for the pure decision helpers that drive the multi-partition
// (one-hash-table-per-GPU) join path:
//   - compute_hash_join_partition_strategy(): the combined natural-count /
//     broadcast / BUILD_PROBE-eligibility decision a PARTITION operator asks the
//     hash join for (folds the former make_broadcast_partition_decision and
//     build_probe_mode_eligible helpers into one).
//   - select_build_probe_action(): the per-partition state-machine decision used
//     by get_next_task_hint / get_next_task_input_data_for_build_probe.
//
// These are GPU-free and pipeline-free, so they exercise the highest-risk logic
// (interleaved per-partition build/probe sequencing) deterministically. The
// end-to-end multi-GPU behavior is covered by the [mgpu] integration tests.

#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_partition.hpp"

#include <catch.hpp>

#include <stdexcept>

using sirius::op::broadcast_slots_to_discard;
using sirius::op::BUILD_HASH_TABLE_STATE;
using sirius::op::build_probe_action;
using sirius::op::build_probe_slot_view;
using sirius::op::compute_hash_join_partition_strategy;
using sirius::op::HASH_JOIN_MODE;
using sirius::op::partition_strategy;
using sirius::op::select_build_probe_action;

namespace {

// Defaults mirroring the production configuration used in the gate.
constexpr uint64_t kMaxBuildBytes = 500ull * 1024 * 1024;  // DEFAULT_MAX_BUILD_HASH_TABLE_BYTES
constexpr uint64_t k100MB         = 100ull * 1024 * 1024;

build_probe_slot_view slot(BUILD_HASH_TABLE_STATE state, bool has_build, bool has_probe)
{
  build_probe_slot_view v;
  v.state           = state;
  v.has_build_batch = has_build;
  v.has_probe_batch = has_probe;
  return v;
}

}  // namespace

//===----------------------------------------------------------------------===//
// compute_hash_join_partition_strategy
//===----------------------------------------------------------------------===//

namespace {

// A large hash_partition_bytes so the natural count is 1 unless a test deliberately makes the input
// exceed it; keeps the "small input" cases at a single natural partition.
constexpr uint64_t kBigPartitionBytes = 512ull * 1024 * 1024;  // 512 MB

// Default broadcast size cap (256 MB); some tests override it to exercise the ratio-based path.
constexpr uint64_t kMaxBroadcastBytes = 256ull * 1024 * 1024;

partition_strategy strategy(uint64_t total_bytes,
                            bool is_build_side,
                            bool build_foldable,
                            int num_gpus,
                            duckdb::JoinType join_type,
                            HASH_JOIN_MODE join_mode              = HASH_JOIN_MODE::STANDARD,
                            uint64_t hash_partition_bytes         = kBigPartitionBytes,
                            uint64_t max_build_hash_table_bytes   = kMaxBuildBytes,
                            uint64_t max_broadcast_join_size      = kMaxBroadcastBytes,
                            double estimated_probe_to_build_ratio = 0.0)
{
  return compute_hash_join_partition_strategy(total_bytes,
                                              is_build_side,
                                              build_foldable,
                                              num_gpus,
                                              hash_partition_bytes,
                                              max_build_hash_table_bytes,
                                              max_broadcast_join_size,
                                              join_type,
                                              join_mode,
                                              estimated_probe_to_build_ratio);
}

}  // namespace

TEST_CASE("compute_hash_join_partition_strategy - single-GPU small foldable inner is BUILD_PROBE",
          "[hash_join][build_probe][unit]")
{
  auto const s = strategy(
    k100MB, /*is_build_side=*/true, /*foldable=*/true, /*num_gpus=*/1, duckdb::JoinType::INNER);
  REQUIRE(s.num_partitions == 1);
  REQUIRE_FALSE(s.broadcast);
  REQUIRE(s.build_probe);
}

TEST_CASE(
  "compute_hash_join_partition_strategy - single-GPU build over the batch target still "
  "BUILD_PROBEs within the hash-table budget",
  "[hash_join][build_probe][unit]")
{
  // 4 natural partitions (400 MB / 100 MB), but the folded build fits the 500 MB hash-table
  // budget — the batch target must not veto BUILD_PROBE, so this collapses to one partition.
  auto const s = strategy(4 * k100MB,
                          true,
                          true,
                          /*num_gpus=*/1,
                          duckdb::JoinType::INNER,
                          HASH_JOIN_MODE::STANDARD,
                          /*hash_partition_bytes=*/k100MB,
                          /*max_build_hash_table_bytes=*/kMaxBuildBytes);
  REQUIRE(s.num_partitions == 1);
  REQUIRE_FALSE(s.broadcast);
  REQUIRE(s.build_probe);
}

TEST_CASE(
  "compute_hash_join_partition_strategy - single-GPU build over both targets splits and stays "
  "STANDARD",
  "[hash_join][build_probe][unit]")
{
  // Same 4 natural partitions, but the build also exceeds the hash-table budget (400 MB > 200 MB),
  // so there is no single-table shape to fall back to and the natural count is reported.
  auto const s = strategy(4 * k100MB,
                          true,
                          true,
                          /*num_gpus=*/1,
                          duckdb::JoinType::INNER,
                          HASH_JOIN_MODE::STANDARD,
                          /*hash_partition_bytes=*/k100MB,
                          /*max_build_hash_table_bytes=*/2 * k100MB);
  REQUIRE(s.num_partitions == 4);
  REQUIRE_FALSE(s.broadcast);
  REQUIRE_FALSE(s.build_probe);
}

TEST_CASE(
  "compute_hash_join_partition_strategy - single-GPU build over the per-GPU cap is STANDARD",
  "[hash_join][build_probe][unit]")
{
  // One natural partition (huge hash_partition_bytes), but the full build exceeds the cap.
  auto const s = strategy(/*total=*/6 * k100MB,
                          true,
                          true,
                          /*num_gpus=*/1,
                          duckdb::JoinType::INNER,
                          HASH_JOIN_MODE::STANDARD,
                          /*hash_partition_bytes=*/1024ull * 1024 * 1024,
                          /*max_build_hash_table_bytes=*/kMaxBuildBytes);
  REQUIRE(s.num_partitions == 1);
  REQUIRE_FALSE(s.build_probe);
}

TEST_CASE(
  "compute_hash_join_partition_strategy - build just over hash_partition_bytes is BUILD_PROBE",
  "[hash_join][build_probe][unit]")
{
  // TPC-H SF1000 q21: 5.94 GB build, 5 GiB batch target (natural count 2), 10 GiB hash-table
  // budget — fits one hash table, so BUILD_PROBE on one partition.
  auto const s = strategy(/*total=*/5'937'804'664ull,
                          true,
                          true,
                          /*num_gpus=*/1,
                          duckdb::JoinType::INNER,
                          HASH_JOIN_MODE::STANDARD,
                          /*hash_partition_bytes=*/5ull * 1024 * 1024 * 1024,
                          /*max_build_hash_table_bytes=*/10ull * 1024 * 1024 * 1024);
  REQUIRE(s.num_partitions == 1);
  REQUIRE_FALSE(s.broadcast);
  REQUIRE(s.build_probe);
}

TEST_CASE(
  "compute_hash_join_partition_strategy - multi-GPU medium build is one-per-GPU BUILD_PROBE",
  "[hash_join][build_probe][unit]")
{
  // 400 MB on 4 GPUs, 100 MB per partition: not small enough to broadcast, but each GPU's slice
  // fits the cap -> hash-partitioned BUILD_PROBE, one partition per GPU, no broadcast.
  auto const s = strategy(4 * k100MB,
                          true,
                          true,
                          /*num_gpus=*/4,
                          duckdb::JoinType::INNER,
                          HASH_JOIN_MODE::STANDARD,
                          /*hash_partition_bytes=*/k100MB);
  REQUIRE(s.num_partitions == 4);
  REQUIRE_FALSE(s.broadcast);
  REQUIRE(s.build_probe);
}

TEST_CASE("compute_hash_join_partition_strategy - multi-GPU small build broadcasts to every GPU",
          "[hash_join][build_probe][unit][broadcast]")
{
  // 10 MB build on 4 GPUs is below the small-table threshold (4 * 16 MB) -> replicate to every GPU.
  auto const s = strategy(10ull * 1024 * 1024, true, true, /*num_gpus=*/4, duckdb::JoinType::INNER);
  REQUIRE(s.num_partitions == 4);
  REQUIRE(s.broadcast);
  REQUIRE(s.build_probe);
}

TEST_CASE(
  "compute_hash_join_partition_strategy - medium build broadcasts when the probe dwarfs the build",
  "[hash_join][build_probe][unit][broadcast]")
{
  // 100 MB build on 4 GPUs: above the small-table threshold (64 MB) but below max_broadcast (256
  // MB), and the probe is 6x the build (>= 4 * 1.25 = 5.0) -> broadcast the medium build.
  auto const s = strategy(100ull * 1024 * 1024,
                          /*is_build_side=*/true,
                          /*foldable=*/true,
                          /*num_gpus=*/4,
                          duckdb::JoinType::INNER,
                          HASH_JOIN_MODE::STANDARD,
                          kBigPartitionBytes,
                          kMaxBuildBytes,
                          kMaxBroadcastBytes,
                          /*estimated_probe_to_build_ratio=*/6.0);
  REQUIRE(s.num_partitions == 4);
  REQUIRE(s.broadcast);
  REQUIRE(s.build_probe);
}

TEST_CASE(
  "compute_hash_join_partition_strategy - medium build hash-partitions when the ratio is too low",
  "[hash_join][build_probe][unit]")
{
  // Same 100 MB build on 4 GPUs, but the probe is only 3x the build (< 5.0) -> no broadcast; it
  // hash-partitions one-per-GPU instead (still BUILD_PROBE by size).
  auto const s = strategy(100ull * 1024 * 1024,
                          true,
                          true,
                          /*num_gpus=*/4,
                          duckdb::JoinType::INNER,
                          HASH_JOIN_MODE::STANDARD,
                          kBigPartitionBytes,
                          kMaxBuildBytes,
                          kMaxBroadcastBytes,
                          /*estimated_probe_to_build_ratio=*/3.0);
  REQUIRE(s.num_partitions == 4);
  REQUIRE_FALSE(s.broadcast);
  REQUIRE(s.build_probe);
}

TEST_CASE(
  "compute_hash_join_partition_strategy - build above max_broadcast_join_size never "
  "ratio-broadcasts",
  "[hash_join][build_probe][unit]")
{
  // 300 MB build exceeds max_broadcast (256 MB); even a huge probe-to-build ratio cannot broadcast.
  auto const s = strategy(300ull * 1024 * 1024,
                          true,
                          true,
                          /*num_gpus=*/4,
                          duckdb::JoinType::INNER,
                          HASH_JOIN_MODE::STANDARD,
                          kBigPartitionBytes,
                          kMaxBuildBytes,
                          kMaxBroadcastBytes,
                          /*estimated_probe_to_build_ratio=*/100.0);
  REQUIRE(s.num_partitions == 4);
  REQUIRE_FALSE(s.broadcast);
  REQUIRE(s.build_probe);
}

TEST_CASE("compute_hash_join_partition_strategy - ratio broadcast threshold scales with num_gpus",
          "[hash_join][build_probe][unit][broadcast]")
{
  // 200 MB build, probe 3x the build. Above the small-table threshold at both GPU counts and below
  // max_broadcast. The ratio bar is num_gpus * 1.25: 2.5 at 2 GPUs (3.0 clears it -> broadcast),
  // 10.0 at 8 GPUs (3.0 misses it -> hash-partition).
  auto const two_gpu = strategy(200ull * 1024 * 1024,
                                true,
                                true,
                                /*num_gpus=*/2,
                                duckdb::JoinType::INNER,
                                HASH_JOIN_MODE::STANDARD,
                                kBigPartitionBytes,
                                kMaxBuildBytes,
                                kMaxBroadcastBytes,
                                /*estimated_probe_to_build_ratio=*/3.0);
  REQUIRE(two_gpu.num_partitions == 2);
  REQUIRE(two_gpu.broadcast);

  auto const eight_gpu = strategy(200ull * 1024 * 1024,
                                  true,
                                  true,
                                  /*num_gpus=*/8,
                                  duckdb::JoinType::INNER,
                                  HASH_JOIN_MODE::STANDARD,
                                  kBigPartitionBytes,
                                  kMaxBuildBytes,
                                  kMaxBroadcastBytes,
                                  /*estimated_probe_to_build_ratio=*/3.0);
  REQUIRE(eight_gpu.num_partitions == 8);
  REQUIRE_FALSE(eight_gpu.broadcast);
}

TEST_CASE("compute_hash_join_partition_strategy - right-family joins never broadcast/build-probe",
          "[hash_join][build_probe][unit]")
{
  // Probe-driven right-family sizing: is_build_side is false -> plain natural count.
  auto const probe_driven = strategy(10ull * 1024 * 1024,
                                     /*is_build_side=*/false,
                                     true,
                                     /*num_gpus=*/4,
                                     duckdb::JoinType::RIGHT);
  REQUIRE(probe_driven.num_partitions == 1);
  REQUIRE_FALSE(probe_driven.broadcast);
  REQUIRE_FALSE(probe_driven.build_probe);

  // Even if the build side drives sizing, RIGHT is excluded from BUILD_PROBE, so a small build
  // falls back to the natural count instead of broadcasting.
  auto const build_driven = strategy(10ull * 1024 * 1024,
                                     /*is_build_side=*/true,
                                     true,
                                     /*num_gpus=*/4,
                                     duckdb::JoinType::RIGHT);
  REQUIRE(build_driven.num_partitions == 1);
  REQUIRE_FALSE(build_driven.broadcast);
  REQUIRE_FALSE(build_driven.build_probe);
}

TEST_CASE("compute_hash_join_partition_strategy - MARK single-GPU clamps to one partition",
          "[hash_join][build_probe][unit]")
{
  // Small foldable MARK build: clamped to a single partition, still BUILD_PROBE by size.
  auto const small = strategy(k100MB, true, true, /*num_gpus=*/1, duckdb::JoinType::MARK);
  REQUIRE(small.num_partitions == 1);
  REQUIRE_FALSE(small.broadcast);
  REQUIRE(small.build_probe);

  // A large MARK build stays clamped to one partition and is still forced into BUILD_PROBE even
  // though it is over the per-GPU hash-table cap: MARK must never fall back to a hash-partitioned
  // STANDARD join (that would split build rows and corrupt build_has_null). Capacity pressure is
  // handled by the downgrade/spill path, not by a mode switch.
  auto const large = strategy(8 * k100MB,
                              true,
                              true,
                              /*num_gpus=*/1,
                              duckdb::JoinType::MARK,
                              HASH_JOIN_MODE::STANDARD,
                              /*hash_partition_bytes=*/k100MB);
  REQUIRE(large.num_partitions == 1);
  REQUIRE_FALSE(large.broadcast);
  REQUIRE(large.build_probe);
}

TEST_CASE("compute_hash_join_partition_strategy - MARK multi-GPU forces broadcast",
          "[hash_join][build_probe][unit][broadcast]")
{
  // MARK cannot be hash-partitioned across batches, so multi-GPU forces broadcast when it fits.
  auto const s = strategy(k100MB, true, true, /*num_gpus=*/4, duckdb::JoinType::MARK);
  REQUIRE(s.num_partitions == 4);
  REQUIRE(s.broadcast);
  REQUIRE(s.build_probe);

  // A MARK build that is over the per-GPU / broadcast cap still forces broadcast BUILD_PROBE on
  // multi-GPU: MARK must never hash-partition its build (that would corrupt build_has_null), so it
  // broadcasts the full build to every GPU regardless of size and relies on downgrade for capacity.
  auto const big = strategy(6 * k100MB,
                            true,
                            true,
                            /*num_gpus=*/4,
                            duckdb::JoinType::MARK,
                            HASH_JOIN_MODE::STANDARD,
                            /*hash_partition_bytes=*/k100MB);
  REQUIRE(big.num_partitions == 4);
  REQUIRE(big.broadcast);
  REQUIRE(big.build_probe);
}

TEST_CASE("compute_hash_join_partition_strategy - mixed / full-outer / unfoldable stay STANDARD",
          "[hash_join][build_probe][unit]")
{
  // Mixed join (equality + inequality) never uses BUILD_PROBE.
  auto const mixed = strategy(
    k100MB, true, true, /*num_gpus=*/1, duckdb::JoinType::INNER, HASH_JOIN_MODE::MIXED_JOIN);
  REQUIRE_FALSE(mixed.build_probe);

  // Full outer over-emits unmatched build rows on the streamed path -> excluded.
  auto const outer = strategy(k100MB, true, true, /*num_gpus=*/1, duckdb::JoinType::OUTER);
  REQUIRE_FALSE(outer.build_probe);

  // Build side that cannot fold to a single batch -> excluded.
  auto const unfoldable =
    strategy(k100MB, true, /*foldable=*/false, /*num_gpus=*/1, duckdb::JoinType::INNER);
  REQUIRE_FALSE(unfoldable.build_probe);
}

TEST_CASE("compute_hash_join_partition_strategy - num_gpus < 1 is a precondition violation",
          "[hash_join][build_probe][unit]")
{
  REQUIRE_THROWS_AS(strategy(k100MB, true, true, /*num_gpus=*/0, duckdb::JoinType::INNER),
                    std::invalid_argument);
}

//===----------------------------------------------------------------------===//
// select_build_probe_action
//===----------------------------------------------------------------------===//

TEST_CASE("select_build_probe_action - no partitions means the operator is complete",
          "[hash_join][build_probe][unit]")
{
  auto const d = select_build_probe_action({});
  REQUIRE(d.action == build_probe_action::none);
}

TEST_CASE("select_build_probe_action - schedules a build for a ready NOT_BUILT partition",
          "[hash_join][build_probe][unit]")
{
  // NOT_BUILT with both its build batch and a probe batch -> build (and first probe) it.
  auto const d = select_build_probe_action({slot(BUILD_HASH_TABLE_STATE::NOT_BUILT, true, true)});
  REQUIRE(d.action == build_probe_action::schedule_build);
  REQUIRE(d.partition == 0);
}

TEST_CASE("select_build_probe_action - waits on build vs probe input when nothing is schedulable",
          "[hash_join][build_probe][unit]")
{
  // Missing build batch -> wait for the build producer so the build can start.
  REQUIRE(
    select_build_probe_action({slot(BUILD_HASH_TABLE_STATE::NOT_BUILT, false, true)}).action ==
    build_probe_action::wait_for_build);

  // Build batch present but no probe batch yet -> wait for probe input.
  REQUIRE(
    select_build_probe_action({slot(BUILD_HASH_TABLE_STATE::NOT_BUILT, true, false)}).action ==
    build_probe_action::wait_for_probe);

  // A build in flight (SCHEDULING/SCHEDULED) with no other work -> wait for probe input.
  REQUIRE(
    select_build_probe_action({slot(BUILD_HASH_TABLE_STATE::SCHEDULING, true, false)}).action ==
    build_probe_action::wait_for_probe);
  REQUIRE(
    select_build_probe_action({slot(BUILD_HASH_TABLE_STATE::SCHEDULED, false, false)}).action ==
    build_probe_action::wait_for_probe);

  // Built but drained of probe data -> wait for probe input (this is also how the op idles until
  // the probe pipeline signals completion).
  REQUIRE(select_build_probe_action({slot(BUILD_HASH_TABLE_STATE::BUILT, false, false)}).action ==
          build_probe_action::wait_for_probe);
}

TEST_CASE("select_build_probe_action - probes a built partition that has probe data",
          "[hash_join][build_probe][unit]")
{
  auto const d = select_build_probe_action({slot(BUILD_HASH_TABLE_STATE::BUILT, false, true)});
  REQUIRE(d.action == build_probe_action::schedule_probe);
  REQUIRE(d.partition == 0);
}

TEST_CASE("select_build_probe_action - prefers starting a build over probing a built partition",
          "[hash_join][build_probe][unit]")
{
  // Partition 0 is BUILT with probe data; partition 1 is NOT_BUILT and ready. Building 1 first
  // gets its GPU busy sooner, so build wins.
  auto const d = select_build_probe_action({
    slot(BUILD_HASH_TABLE_STATE::BUILT, false, true),
    slot(BUILD_HASH_TABLE_STATE::NOT_BUILT, true, true),
  });
  REQUIRE(d.action == build_probe_action::schedule_build);
  REQUIRE(d.partition == 1);
}

TEST_CASE("select_build_probe_action - interleaves: probes a built partition while another builds",
          "[hash_join][build_probe][unit]")
{
  // Partition 0 is still building (SCHEDULED); partition 1 is BUILT with probe data. With no
  // NOT_BUILT partition ready to build, we probe partition 1 on its GPU while 0 finishes.
  auto const d = select_build_probe_action({
    slot(BUILD_HASH_TABLE_STATE::SCHEDULED, false, false),
    slot(BUILD_HASH_TABLE_STATE::BUILT, false, true),
  });
  REQUIRE(d.action == build_probe_action::schedule_probe);
  REQUIRE(d.partition == 1);
}

TEST_CASE("select_build_probe_action - a SCHEDULING partition is not re-scheduled",
          "[hash_join][build_probe][unit]")
{
  // Partition 0 was just claimed for build (SCHEDULING) by a prior hint; it must not be picked
  // again. Partition 1 is BUILT with probe data, so that is the next action.
  auto const d = select_build_probe_action({
    slot(BUILD_HASH_TABLE_STATE::SCHEDULING, true, true),
    slot(BUILD_HASH_TABLE_STATE::BUILT, false, true),
  });
  REQUIRE(d.action == build_probe_action::schedule_probe);
  REQUIRE(d.partition == 1);
}

TEST_CASE("select_build_probe_action - all partitions destroyed reports completion",
          "[hash_join][build_probe][unit]")
{
  auto const d = select_build_probe_action({
    slot(BUILD_HASH_TABLE_STATE::DESTROYED, false, false),
    slot(BUILD_HASH_TABLE_STATE::DESTROYED, false, false),
  });
  REQUIRE(d.action == build_probe_action::none);
}

TEST_CASE("select_build_probe_action - first ready build wins across many partitions",
          "[hash_join][build_probe][unit]")
{
  // Partition 0 built+drained, 1 built+drained, 2 NOT_BUILT+ready, 3 NOT_BUILT+ready. The first
  // schedulable build (partition 2) is chosen.
  auto const d = select_build_probe_action({
    slot(BUILD_HASH_TABLE_STATE::BUILT, false, false),
    slot(BUILD_HASH_TABLE_STATE::BUILT, false, false),
    slot(BUILD_HASH_TABLE_STATE::NOT_BUILT, true, true),
    slot(BUILD_HASH_TABLE_STATE::NOT_BUILT, true, true),
  });
  REQUIRE(d.action == build_probe_action::schedule_build);
  REQUIRE(d.partition == 2);
}

//===----------------------------------------------------------------------===//
// broadcast_slots_to_discard
//===----------------------------------------------------------------------===//

TEST_CASE("broadcast_slots_to_discard - nothing is discarded until the probe side finishes",
          "[build_probe]")
{
  // Build-only slots exist, but probe may still deliver data — discard nothing yet.
  auto const d = broadcast_slots_to_discard(
    {
      slot(BUILD_HASH_TABLE_STATE::NOT_BUILT, true, false),
      slot(BUILD_HASH_TABLE_STATE::NOT_BUILT, true, false),
    },
    /*probe_finished=*/false);
  REQUIRE(d.empty());
}

TEST_CASE("broadcast_slots_to_discard - discards only NOT_BUILT slots with build but no probe",
          "[build_probe]")
{
  // p0: has probe -> will be built, keep.  p1: build-only -> discard.  p2: already BUILT -> keep.
  // p3: build-only -> discard.  p4: no build batch at all -> nothing to discard.
  auto const d = broadcast_slots_to_discard(
    {
      slot(BUILD_HASH_TABLE_STATE::NOT_BUILT, true, true),
      slot(BUILD_HASH_TABLE_STATE::NOT_BUILT, true, false),
      slot(BUILD_HASH_TABLE_STATE::BUILT, false, false),
      slot(BUILD_HASH_TABLE_STATE::NOT_BUILT, true, false),
      slot(BUILD_HASH_TABLE_STATE::NOT_BUILT, false, false),
    },
    /*probe_finished=*/true);
  REQUIRE(d == std::vector<std::size_t>{1, 3});
}

TEST_CASE("broadcast_slots_to_discard - a DESTROYED slot is not rediscarded", "[build_probe]")
{
  auto const d = broadcast_slots_to_discard(
    {
      slot(BUILD_HASH_TABLE_STATE::DESTROYED, false, false),
      slot(BUILD_HASH_TABLE_STATE::NOT_BUILT, true, false),
    },
    /*probe_finished=*/true);
  REQUIRE(d == std::vector<std::size_t>{1});
}
