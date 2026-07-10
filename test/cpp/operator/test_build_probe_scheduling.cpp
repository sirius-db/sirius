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

// Unit tests for the pure BUILD_PROBE scheduling helpers that drive the
// multi-partition (one-hash-table-per-GPU) join path:
//   - build_probe_mode_eligible(): the eligibility gate used by
//     update_join_exec_mode to admit BUILD_PROBE for up to one partition per GPU.
//   - select_build_probe_action(): the per-partition state-machine decision used
//     by get_next_task_hint / get_next_task_input_data_for_build_probe.
//
// These are GPU-free and pipeline-free, so they exercise the highest-risk logic
// (interleaved per-partition build/probe sequencing) deterministically. The
// end-to-end multi-GPU behavior is covered by the [mgpu] integration tests.

#include "op/sirius_physical_hash_join.hpp"

#include <catch.hpp>

using sirius::op::BUILD_HASH_TABLE_STATE;
using sirius::op::build_probe_action;
using sirius::op::build_probe_mode_eligible;
using sirius::op::build_probe_slot_view;
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
// build_probe_mode_eligible
//===----------------------------------------------------------------------===//

TEST_CASE("build_probe_mode_eligible - single GPU keeps the historical single-partition rule",
          "[hash_join][build_probe][unit]")
{
  // 1 partition on 1 GPU, small foldable build, plain inner join -> eligible.
  REQUIRE(build_probe_mode_eligible(/*num_partitions=*/1,
                                    /*build_side_bytes=*/k100MB,
                                    /*build_foldable_to_single_batch=*/true,
                                    /*is_right_family=*/false,
                                    /*is_mixed_join=*/false,
                                    /*num_gpus=*/1,
                                    kMaxBuildBytes));

  // On a single GPU, more than one partition is never eligible (num_partitions > num_gpus).
  REQUIRE_FALSE(build_probe_mode_eligible(2, k100MB, true, false, false, 1, kMaxBuildBytes));
}

TEST_CASE("build_probe_mode_eligible - multi-GPU admits up to one partition per GPU",
          "[hash_join][build_probe][unit]")
{
  // 4 partitions on 4 GPUs, each partition's average build side under the cap -> eligible.
  REQUIRE(build_probe_mode_eligible(/*num_partitions=*/4,
                                    /*build_side_bytes=*/4 * k100MB,  // 400 MB total, 100 MB/part
                                    /*build_foldable_to_single_batch=*/true,
                                    /*is_right_family=*/false,
                                    /*is_mixed_join=*/false,
                                    /*num_gpus=*/4,
                                    kMaxBuildBytes));

  // More partitions than GPUs -> not eligible (a partition would share a GPU / large joins stay
  // in STANDARD mode).
  REQUIRE_FALSE(build_probe_mode_eligible(5, 5 * k100MB, true, false, false, 4, kMaxBuildBytes));
}

TEST_CASE("build_probe_mode_eligible - the size cap is per-partition, not total",
          "[hash_join][build_probe][unit]")
{
  // Total build side is 4x the cap, but split across 4 partitions each partition is under it.
  uint64_t const total = 4 * (kMaxBuildBytes - 1);
  REQUIRE(build_probe_mode_eligible(4, total, true, false, false, 4, kMaxBuildBytes));

  // A single partition holding that same total exceeds the cap -> not eligible.
  REQUIRE_FALSE(build_probe_mode_eligible(1, total, true, false, false, 4, kMaxBuildBytes));

  // Per-partition average at or above the cap -> not eligible.
  REQUIRE_FALSE(
    build_probe_mode_eligible(2, 2 * kMaxBuildBytes, true, false, false, 2, kMaxBuildBytes));
}

TEST_CASE("build_probe_mode_eligible - excluded join shapes and unfoldable builds",
          "[hash_join][build_probe][unit]")
{
  // Right-family joins are excluded (they emit build-side output).
  REQUIRE_FALSE(
    build_probe_mode_eligible(1, k100MB, true, /*is_right_family=*/true, false, 1, kMaxBuildBytes));
  // Mixed joins never use BUILD_PROBE.
  REQUIRE_FALSE(
    build_probe_mode_eligible(1, k100MB, true, false, /*is_mixed_join=*/true, 1, kMaxBuildBytes));
  // Build side that cannot fold to a single batch per partition.
  REQUIRE_FALSE(build_probe_mode_eligible(
    1, k100MB, /*build_foldable=*/false, false, false, 1, kMaxBuildBytes));
  // Degenerate counts are rejected rather than dividing by zero.
  REQUIRE_FALSE(build_probe_mode_eligible(0, k100MB, true, false, false, 1, kMaxBuildBytes));
  REQUIRE_FALSE(build_probe_mode_eligible(1, k100MB, true, false, false, 0, kMaxBuildBytes));
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
