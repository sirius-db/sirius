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
#include "memory/memory_wait_board.hpp"
#include "preload/config.hpp"
#include "preload/preload_policy.hpp"
#include "preload/preload_stats.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

using cucascade::memory::Tier;
using sirius::memory::memory_wait_board;
using namespace sirius::preload;

namespace {

constexpr std::size_t GiB = std::size_t{1} << 30;
constexpr std::size_t MiB = std::size_t{1} << 20;

std::size_t refused(assignment_result const& r, refusal why)
{
  return r.refusals[static_cast<std::size_t>(why)];
}

/// A GPU with room for everything unless a test narrows it.
gpu_budget_state open_gpu(std::size_t room = 100 * GiB)
{
  gpu_budget_state g;
  g.room_committed     = room;
  g.room_speculative   = room;
  g.window             = room;
  g.speculative_window = room;
  g.max_batch_bytes    = room;
  return g;
}

}  // namespace

// ---- plan order --------------------------------------------------------------------------------

TEST_CASE("plan_key orders by band, priority, class, partition, position, then batch id",
          "[preload_policy]")
{
  plan_key const base{band_gpu, 10, klass::A, 0, 5, 7};

  auto staging     = base;
  staging.band     = band_host_staging;
  staging.priority = 0;
  CHECK(base < staging);  // any GPU item before any staging item

  auto later               = base;
  later.priority           = 11;
  later.k                  = klass::A;
  auto b_same_priority     = base;
  b_same_priority.k        = klass::B;
  b_same_priority.position = 0;
  CHECK(base < b_same_priority);   // A before B at equal priority
  CHECK(b_same_priority < later);  // priority dominates class

  auto next_partition      = b_same_priority;
  next_partition.partition = 1;
  next_partition.position  = 0;
  auto front               = b_same_priority;
  front.partition          = 0;
  front.position           = 3;
  CHECK(front < next_partition);

  auto tiebreak     = base;
  tiebreak.batch_id = 8;
  CHECK(base < tiebreak);
  CHECK(base == plan_key{band_gpu, 10, klass::A, 0, 5, 7});

  std::vector<plan_key> keys{later, staging, tiebreak, b_same_priority, base};
  std::sort(keys.begin(), keys.end());
  CHECK(keys == std::vector<plan_key>{base, tiebreak, b_same_priority, later, staging});
}

// ---- arithmetic --------------------------------------------------------------------------------

TEST_CASE("preload size arithmetic saturates and estimates with slack", "[preload_policy]")
{
  constexpr auto max = std::numeric_limits<std::size_t>::max();
  STATIC_REQUIRE(sat_add(max, 1) == max);
  STATIC_REQUIRE(sat_sub(1, 2) == 0);
  STATIC_REQUIRE(sat_mul(max / 2 + 1, 2) == max);
  STATIC_REQUIRE(sat_mul(0, max) == 0);

  STATIC_REQUIRE(with_slack(0) == MiB);
  STATIC_REQUIRE(with_slack(64 * MiB) == 65 * MiB);
  STATIC_REQUIRE(with_slack(1024 * MiB) == 1040 * MiB);

  // There is no OOM multiplier: the first GPU OOM bans the candidate, so the estimate is never
  // inflated. reserve_estimate is just with_slack(max(est, 1)).
  STATIC_REQUIRE(reserve_estimate(0) == with_slack(1));
  STATIC_REQUIRE(reserve_estimate(64 * MiB) == with_slack(64 * MiB));

  STATIC_REQUIRE(fraction_of(0.5, 100) == 50);
  STATIC_REQUIRE(fraction_of(-0.1, 100) == 0);
  STATIC_REQUIRE(fraction_of(2.0, 100) == 100);

  STATIC_REQUIRE(consumed_bytes(100, 30) == 70);
  STATIC_REQUIRE(consumed_bytes(100, 130) == 0);
}

// ---- ceilings ----------------------------------------------------------------------------------

TEST_CASE("preload ceilings follow the device preloading ceiling, in consumed bytes",
          "[preload_policy]")
{
  preload_executor_config cfg;  // ceiling absent (= stop), speculative headroom 0.1, host 0.1
  space_limits const gpu{100 * GiB, 100 * GiB, 60 * GiB};

  SECTION("defaults: committed at the stop, speculative one headroom below")
  {
    auto const c = gpu_ceilings_bytes(cfg, gpu);
    CHECK(c.committed == 60 * GiB);  // absent ceiling defaults to the stop fraction
    CHECK(c.speculative == fraction_of(0.5, 100 * GiB));
  }

  SECTION("an explicit ceiling below the stop is used; above it, it is clamped")
  {
    cfg.device_preloading_ceiling = 0.4;
    CHECK(gpu_ceilings_bytes(cfg, gpu).committed == fraction_of(0.4, 100 * GiB));
    cfg.device_preloading_ceiling = 0.9;  // above the 0.6 stop
    CHECK(gpu_ceilings_bytes(cfg, gpu).committed == 60 * GiB);
  }

  SECTION("host ceiling is stop minus host headroom")
  {
    space_limits const host{100 * GiB, 100 * GiB, 80 * GiB};
    CHECK(host_ceiling_bytes(cfg, host) == fraction_of(0.7, 100 * GiB));
  }

  SECTION("a speculative ceiling at or below zero disables class B")
  {
    cfg.gpu_speculative_headroom_fraction = 0.7;  // more than the committed ceiling
    auto const c                          = gpu_ceilings_bytes(cfg, gpu);
    CHECK(c.committed == 60 * GiB);
    CHECK(c.speculative == 0);

    cfg.host_headroom_fraction = 0.8;
    CHECK(host_ceiling_bytes(cfg, space_limits{100 * GiB, 100 * GiB, 80 * GiB}) == 0);
  }

  SECTION("ceilings never exceed the reservation limit or the stop threshold")
  {
    // Limit 50 GiB of a 100 GiB capacity: stop (60 GiB) is unreachable, so clamp to 50.
    space_limits const small_limit{100 * GiB, 50 * GiB, 60 * GiB};
    auto const c = gpu_ceilings_bytes(cfg, small_limit);
    CHECK(c.committed == 50 * GiB);
    CHECK(c.speculative == 50 * GiB);
    CHECK(ceiling_bytes(0.9, gpu) == 60 * GiB);
  }

  SECTION("a zero-capacity space has no room")
  {
    CHECK(gpu_ceilings_bytes(cfg, space_limits{}).committed == 0);
    CHECK(host_ceiling_bytes(cfg, space_limits{}) == 0);
  }

  SECTION("heterogeneous GPUs get per-GPU committed ceilings")
  {
    space_limits const big{100 * GiB, 100 * GiB, 60 * GiB};  // stop 0.6
    space_limits const small{50 * GiB, 50 * GiB, 20 * GiB};  // stop 0.4
    // Absent ceiling: each GPU's committed ceiling is its own downgrade stop.
    CHECK(gpu_ceilings_bytes(cfg, big).committed == 60 * GiB);
    CHECK(gpu_ceilings_bytes(cfg, small).committed == 20 * GiB);
    // An explicit ceiling is clamped per GPU to that GPU's downgrade stop.
    cfg.device_preloading_ceiling = 0.5;
    CHECK(gpu_ceilings_bytes(cfg, big).committed == fraction_of(0.5, 100 * GiB));  // below 0.6 stop
    CHECK(gpu_ceilings_bytes(cfg, small).committed == 20 * GiB);  // clamped to the 0.4 stop
    // The speculative headroom is subtracted per GPU from each committed ceiling.
    cfg.gpu_speculative_headroom_fraction = 0.1;
    CHECK(gpu_ceilings_bytes(cfg, big).speculative == fraction_of(0.4, 100 * GiB));   // 0.5 - 0.1
    CHECK(gpu_ceilings_bytes(cfg, small).speculative == fraction_of(0.3, 50 * GiB));  // 0.4 - 0.1
  }
}

// ---- budgets -----------------------------------------------------------------------------------

TEST_CASE("gpu_budget is zero under a hold-off, a gate or the trigger, else the binding limit",
          "[preload_policy]")
{
  auto board = std::make_shared<memory_wait_board>(1);
  gpu_budget_inputs in;
  in.device            = 0;
  in.consumed          = 10 * GiB;
  in.mem_epoch         = 4;
  in.gate_epoch        = 3;
  in.unreserved_bytes  = 5 * GiB;
  in.ceilings          = {60 * GiB, 50 * GiB};
  in.lookahead_bytes   = 20 * GiB;
  in.outstanding       = 2 * GiB;
  in.speculative_bytes = 5 * GiB;
  in.outstanding_b     = 1 * GiB;
  in.max_batch_bytes   = 10 * GiB;

  SECTION("open: the window binds class A, the speculative window binds class B")
  {
    auto const a = gpu_budget(in, *board, klass::A);
    CHECK(a.bytes == 18 * GiB);
    CHECK(a.reason == refusal::window);
    auto const b = gpu_budget(in, *board, klass::B);
    CHECK(b.bytes == 4 * GiB);
    CHECK(b.reason == refusal::speculative_window);
  }

  SECTION("consumed plus in-flight bytes count against the ceiling")
  {
    in.consumed  = 50 * GiB;
    auto const a = gpu_budget(in, *board, klass::A);
    CHECK(a.bytes == 5 * GiB);
    CHECK(a.reason == refusal::ceiling);
    CHECK(gpu_budget(in, *board, klass::B).bytes == 0);
  }

  SECTION("a board waiter holds the GPU off")
  {
    auto waiter  = board->enter_wait(0);
    auto const a = gpu_budget(in, *board, klass::A);
    CHECK(a.bytes == 0);
    CHECK(a.reason == refusal::held_off);
  }

  SECTION("an OOM token holds the GPU off; another GPU's token does not")
  {
    {
      auto other = board->hold_oom(1);
      CHECK(gpu_budget(in, *board, klass::A).bytes > 0);
    }
    auto token = board->hold_oom(0);
    CHECK(gpu_budget(in, *board, klass::B).reason == refusal::held_off);
  }

  SECTION("a worker refusal since the last memory event gates the GPU")
  {
    in.gate_epoch = in.mem_epoch;
    CHECK(gpu_budget(in, *board, klass::A).reason == refusal::epoch);
  }

  SECTION("at or above the trigger the GPU is closed")
  {
    in.above_trigger = true;
    CHECK(gpu_budget(in, *board, klass::A).reason == refusal::trigger);
  }

  SECTION("a full window leaves nothing")
  {
    in.outstanding = 20 * GiB;
    auto const a   = gpu_budget(in, *board, klass::A);
    CHECK(a.bytes == 0);
    CHECK(a.reason == refusal::window);
  }
}

TEST_CASE("host_budget is zero while a NUMA-local GPU is pressed", "[preload_policy]")
{
  auto board = std::make_shared<memory_wait_board>(1);
  std::array<gpu_pressure, 2> local{gpu_pressure{0, false}, gpu_pressure{1, false}};
  host_budget_inputs in;
  in.consumed         = 10 * GiB;
  in.mem_epoch        = 1;
  in.gate_epoch       = 0;
  in.unreserved_bytes = 2 * GiB;
  in.ceiling          = 70 * GiB;
  in.local_gpus       = local;

  CHECK(host_budget(in, *board).bytes == 58 * GiB);
  CHECK(host_budget(in, *board).reason == refusal::ceiling);

  SECTION("a waiter on a local GPU")
  {
    auto waiter = board->enter_wait(1);
    CHECK(host_budget(in, *board).bytes == 0);
    CHECK(host_budget(in, *board).reason == refusal::held_off);
  }
  SECTION("an OOM token on a local GPU")
  {
    auto token = board->hold_oom(0);
    CHECK(host_budget(in, *board).reason == refusal::held_off);
  }
  SECTION("a local GPU at its trigger")
  {
    local[0].above_trigger = true;
    CHECK(host_budget(in, *board).reason == refusal::held_off);
  }
  SECTION("a gate")
  {
    in.gate_epoch = in.mem_epoch;
    CHECK(host_budget(in, *board).reason == refusal::epoch);
  }
  SECTION("the host's own trigger")
  {
    in.above_trigger = true;
    CHECK(host_budget(in, *board).reason == refusal::trigger);
  }
  SECTION("a starting state carries the closure")
  {
    in.above_trigger = true;
    auto const s     = make_host_budget_state(in, *board);
    CHECK(s.room == 0);
    CHECK(s.closed == refusal::trigger);
  }
}

// ---- head-of-line assignment -------------------------------------------------------------------

TEST_CASE("head-of-line assignment closes a target at the first item that does not fit",
          "[preload_policy]")
{
  std::vector<gpu_budget_state> gpus{open_gpu(10 * GiB)};
  std::vector<host_budget_state> hosts{{100 * GiB, refusal::none}};

  SECTION("a refusal blocks every later item on that GPU, however small")
  {
    std::vector<assign_item> items{
      {klass::A, 0, std::nullopt, 6 * GiB, 0},
      {klass::A, 0, std::nullopt, 6 * GiB, 0},  // does not fit: closes GPU 0
      {klass::A, 0, std::nullopt, 1 * GiB, 0},  // would fit, but behind the head
    };
    auto const r = assign_head_of_line(items, gpus, hosts);
    REQUIRE(r.assigned.size() == 1);
    CHECK(r.assigned[0].item == 0);
    CHECK(r.assigned[0].t == target{Tier::GPU, 0});
    CHECK(refused(r, refusal::ceiling) == 2);
  }

  SECTION("a refused item always closes its GPU (there is no OOM-retry exception)")
  {
    gpus[0].max_batch_bytes = 100 * GiB;
    std::vector<assign_item> items{
      {klass::A, 0, std::nullopt, 20 * GiB, 0},  // does not fit: closes GPU 0
      {klass::A, 0, std::nullopt, 1 * GiB, 0},   // behind the closed head, even though it fits
    };
    auto const r = assign_head_of_line(items, gpus, hosts);
    CHECK(r.assigned.empty());
    CHECK(gpus[0].closed_a == refusal::ceiling);
  }

  SECTION("a class-B refusal leaves class A open; a class-A refusal closes both")
  {
    gpus[0].speculative_window = 1 * GiB;
    std::vector<assign_item> items{
      {klass::B, 0, std::nullopt, 2 * GiB, 0},  // speculative window: closes B
      {klass::A, 0, std::nullopt, 2 * GiB, 0},  // A still fits
      {klass::B, 0, std::nullopt, 1 * GiB, 0},  // B closed
      {klass::A, 0, std::nullopt, 9 * GiB, 0},  // 8 GiB left: closes A
      {klass::A, 0, std::nullopt, 1 * GiB, 0},
    };
    auto const r = assign_head_of_line(items, gpus, hosts);
    REQUIRE(r.assigned.size() == 1);
    CHECK(r.assigned[0].item == 1);
    CHECK(refused(r, refusal::speculative_window) == 2);
    CHECK(refused(r, refusal::ceiling) == 2);
    CHECK(gpus[0].closed_a == refusal::ceiling);
    CHECK(gpus[0].closed_b == refusal::speculative_window);
  }

  SECTION("a DISK item refused for room falls back to its HOST target")
  {
    gpus[0].max_batch_bytes = 100 * GiB;
    std::vector<assign_item> items{
      {klass::A, 0, 0, 11 * GiB, 11 * GiB},
    };
    auto const r = assign_head_of_line(items, gpus, hosts);
    REQUIRE(r.assigned.size() == 1);
    CHECK(r.assigned[0].t == target{Tier::HOST, 0});
    CHECK(r.assigned[0].fallback);
    CHECK(hosts[0].room == 89 * GiB);
  }

  SECTION("a held-off GPU closes without falling back")
  {
    gpus[0].closed_a = gpus[0].closed_b = refusal::held_off;
    std::vector<assign_item> items{
      {klass::A, 0, 0, 1 * GiB, 1 * GiB},
    };
    auto const r = assign_head_of_line(items, gpus, hosts);
    CHECK(r.assigned.empty());
    CHECK(refused(r, refusal::held_off) == 1);
  }

  SECTION("an item above the per-batch limit never targets the GPU and closes nothing")
  {
    gpus[0].max_batch_bytes = 2 * GiB;
    std::vector<assign_item> items{
      {klass::A, 0, std::nullopt, 3 * GiB, 0},
      {klass::A, 0, 0, 4 * GiB, 4 * GiB},
      {klass::A, 0, std::nullopt, 1 * GiB, 0},
    };
    auto const r = assign_head_of_line(items, gpus, hosts);
    REQUIRE(r.assigned.size() == 2);
    CHECK(r.assigned[0].item == 1);
    CHECK(r.assigned[0].t == target{Tier::HOST, 0});
    CHECK(r.assigned[1].item == 2);
    CHECK(r.assigned[1].t == target{Tier::GPU, 0});
    CHECK(refused(r, refusal::batch_too_big) == 2);
  }

  SECTION("HOST targets are head-of-line too")
  {
    hosts[0].room = 5 * GiB;
    std::vector<assign_item> items{
      {klass::B, std::nullopt, 0, 0, 3 * GiB},
      {klass::B, std::nullopt, 0, 0, 3 * GiB},
      {klass::B, std::nullopt, 0, 0, 1 * GiB},
    };
    auto const r = assign_head_of_line(items, gpus, hosts);
    REQUIRE(r.assigned.size() == 1);
    CHECK_FALSE(r.assigned[0].fallback);
    CHECK(hosts[0].closed == refusal::ceiling);
    CHECK(refused(r, refusal::ceiling) == 2);
  }

  SECTION("items assigned to a GPU consume its windows")
  {
    std::vector<assign_item> items{
      {klass::B, 0, std::nullopt, 3 * GiB, 0},
      {klass::A, 0, std::nullopt, 3 * GiB, 0},
    };
    auto const r = assign_head_of_line(items, gpus, hosts);
    CHECK(r.assigned.size() == 2);
    CHECK(gpus[0].room_committed == 4 * GiB);
    CHECK(gpus[0].window == 4 * GiB);
    CHECK(gpus[0].speculative_window == 7 * GiB);
  }

  SECTION("an item with no usable target is counted")
  {
    std::vector<assign_item> items{{klass::A, 5, std::nullopt, 1, 0}};
    auto const r = assign_head_of_line(items, gpus, hosts);
    CHECK(r.assigned.empty());
    CHECK(refused(r, refusal::no_target) == 1);
  }
}

// ---- target choice -----------------------------------------------------------------------------

TEST_CASE("choose_target follows the class, bucket, readiness and ban rules", "[preload_policy]")
{
  preload_executor_config const cfg;  // stage_to_host, lookahead 1
  std::array<int, 1> const one_gpu{3};
  std::array<int, 2> const two_gpus{0, 1};

  target_request a;
  a.k             = klass::A;
  a.location      = Tier::HOST;
  a.bucket        = bucket_kind::exact;
  a.bucket_device = 1;
  a.active_gpus   = two_gpus;

  SECTION("class A in an exact bucket targets that GPU with the committed ceiling")
  {
    auto const c = choose_target(a, cfg);
    CHECK(c.decision == target_decision::gpu);
    CHECK(c.device == 1);
    CHECK(c.band == band_gpu);
    CHECK_FALSE(c.speculative);
  }

  SECTION("class A in the shared bucket: the single GPU, or staging with several")
  {
    a.bucket          = bucket_kind::shared;
    a.bucket_device   = std::nullopt;
    a.active_gpus     = one_gpu;
    auto const single = choose_target(a, cfg);
    CHECK(single.decision == target_decision::gpu);
    CHECK(single.device == 3);

    a.active_gpus      = two_gpus;
    a.location         = Tier::DISK;
    auto const several = choose_target(a, cfg);
    CHECK(several.decision == target_decision::host_staging);
    CHECK(several.band == band_host_staging);
    CHECK(several.device == -1);

    a.location = Tier::HOST;  // nothing to stage from HOST
    CHECK(choose_target(a, cfg).decision == target_decision::not_planned);
  }

  SECTION("a GPU-banned class A item is staged toward its bucket GPU's host when on DISK")
  {
    a.gpu_banned = true;
    a.location   = Tier::DISK;
    auto const c = choose_target(a, cfg);
    CHECK(c.decision == target_decision::host_staging);
    CHECK(c.device == 1);

    a.location = Tier::HOST;
    CHECK(choose_target(a, cfg).decision == target_decision::not_planned);
  }

  SECTION("speculative_banned does not affect class A")
  {
    a.speculative_banned = true;
    CHECK(choose_target(a, cfg).decision == target_decision::gpu);
  }

  target_request b;
  b.k              = klass::B;
  b.location       = Tier::DISK;
  b.consumer_ready = false;
  b.active_gpus    = two_gpus;

  SECTION("class B behind an unfinished barrier is staged only")
  {
    CHECK(choose_target(b, cfg).decision == target_decision::host_staging);
    b.location = Tier::HOST;
    CHECK(choose_target(b, cfg).decision == target_decision::not_planned);
  }

  SECTION("a ready partition consumer: the next partition is skipped, the one after targeted")
  {
    b.consumer_ready     = true;
    b.partition_consumer = true;
    b.placement_device   = 1;

    b.partition_rank = 0;
    auto const next  = choose_target(b, cfg);
    CHECK(next.decision == target_decision::not_planned);
    CHECK(next.reason == skip_reason::creator_next);

    b.partition_rank = 1;
    auto const after = choose_target(b, cfg);
    CHECK(after.decision == target_decision::gpu);
    CHECK(after.device == 1);
    CHECK(after.speculative);
    CHECK(after.band == band_gpu);

    b.partition_rank = 2;  // beyond repository_partition_lookahead = 1
    CHECK(choose_target(b, cfg).decision == target_decision::not_planned);
  }

  SECTION("a missing placement falls back to the single active GPU, else nothing")
  {
    b.consumer_ready     = true;
    b.partition_consumer = true;
    b.partition_rank     = 1;
    b.placement_device   = std::nullopt;
    b.active_gpus        = one_gpu;
    auto const single    = choose_target(b, cfg);
    CHECK(single.decision == target_decision::gpu);
    CHECK(single.device == 3);

    b.active_gpus   = two_gpus;
    auto const none = choose_target(b, cfg);
    CHECK(none.decision == target_decision::not_planned);
    CHECK(none.reason == skip_reason::no_target);
  }

  SECTION("a ready unpartitioned consumer is left to the task creator")
  {
    b.consumer_ready     = true;
    b.partition_consumer = false;
    CHECK(choose_target(b, cfg).decision == target_decision::not_planned);
  }

  SECTION("banned class B items are only staged")
  {
    b.consumer_ready     = true;
    b.partition_consumer = true;
    b.partition_rank     = 1;
    b.placement_device   = 0;
    for (bool const gpu_banned : {true, false}) {
      b.gpu_banned         = gpu_banned;
      b.speculative_banned = !gpu_banned;
      auto const c         = choose_target(b, cfg);
      CHECK(c.decision == target_decision::host_staging);
      CHECK(c.device == 0);
    }
  }

  SECTION("stage_to_host = false never targets HOST")
  {
    auto no_staging          = cfg;
    no_staging.stage_to_host = false;
    CHECK(choose_target(b, no_staging).decision == target_decision::not_planned);
    a.bucket        = bucket_kind::shared;
    a.bucket_device = std::nullopt;
    a.location      = Tier::DISK;
    CHECK(choose_target(a, no_staging).decision == target_decision::not_planned);
  }

  SECTION("a candidate not on HOST or DISK is never planned")
  {
    a.location = Tier::GPU;
    CHECK(choose_target(a, cfg).decision == target_decision::not_planned);
  }
}

TEST_CASE("choose_staging_host prefers the target GPU's NUMA node, then the origins",
          "[preload_policy]")
{
  STATIC_REQUIRE(choose_staging_host(1, 0, 0) == 1);
  STATIC_REQUIRE(choose_staging_host(-1, 2, 0) == 2);
  STATIC_REQUIRE(choose_staging_host(-1, std::nullopt, 3) == 3);
  STATIC_REQUIRE(choose_staging_host(-1, -1, -1) == -1);
  STATIC_REQUIRE(choose_staging_host(-1, std::nullopt, -1) == -1);

  // Nodes without a HOST space are skipped, down the same chain.
  constexpr auto only_two = [](int node) { return node == 2; };
  STATIC_REQUIRE(choose_staging_host(1, 2, 0, only_two) == 2);
  STATIC_REQUIRE(choose_staging_host(2, 0, 0, only_two) == 2);
  STATIC_REQUIRE(choose_staging_host(1, 0, 3, only_two) == -1);
  STATIC_REQUIRE(choose_staging_host(-1, std::nullopt, 2, only_two) == 2);
}

// ---- margins and windows -----------------------------------------------------------------------

TEST_CASE("queue margins: H for a device's bucket, H x GPUs for the shared bucket",
          "[preload_policy]")
{
  preload_executor_config cfg;
  CHECK(resolve_head_margin(cfg, 4) == 4);  // auto = pipeline threads
  cfg.head_margin_tasks = 0;
  CHECK(resolve_head_margin(cfg, 4) == 0);
  cfg.head_margin_tasks = 2;
  CHECK(resolve_head_margin(cfg, 4) == 2);
  cfg.head_margin_tasks = -1;
  CHECK(resolve_head_margin(cfg, -3) == 0);

  auto const exact = exact_bucket_window(4, 8);
  CHECK_FALSE(exact.contains(3));
  CHECK(exact.contains(4));
  CHECK(exact.contains(11));
  CHECK_FALSE(exact.contains(12));

  auto const shared = shared_bucket_window(4, 8, 3);
  CHECK_FALSE(shared.contains(11));
  CHECK(shared.contains(12));
  CHECK(shared.contains(19));
  CHECK_FALSE(shared.contains(20));

  // One GPU: the shared bucket pops after the GPU's own bucket, so its positions continue there.
  auto const single = shared_bucket_window(4, 8, 1);
  CHECK(shared_bucket_position(0, 2, 1) == 2);
  CHECK_FALSE(single.contains(shared_bucket_position(1, 2, 1)));
  CHECK(single.contains(shared_bucket_position(2, 2, 1)));
  CHECK(shared_bucket_position(5, 2, 3) == 5);  // several GPUs: judged on its own

  STATIC_REQUIRE_FALSE(class_b_partition_admitted(0, 1));
  STATIC_REQUIRE(class_b_partition_admitted(1, 1));
  STATIC_REQUIRE_FALSE(class_b_partition_admitted(2, 1));
  STATIC_REQUIRE(class_b_partition_admitted(2, 2));
  STATIC_REQUIRE_FALSE(class_b_partition_admitted(1, 0));
  STATIC_REQUIRE(fifo_admitted(1, 2));
  STATIC_REQUIRE_FALSE(fifo_admitted(2, 2));
  STATIC_REQUIRE_FALSE(fifo_admitted(0, 0));
}

// ---- lane caps ---------------------------------------------------------------------------------

TEST_CASE("lane caps: C2C x overrides", "[preload_policy]")
{
  preload_executor_config cfg;
  CHECK(h2g_cap(cfg, true) == 5);
  CHECK(h2g_cap(cfg, false) == 1);
  // The auto DISK width is always 1 at this pin: there is no GPUDirect Storage path.
  CHECK(disk_cap(cfg, true) == 1);
  CHECK(disk_cap(cfg, false) == 1);

  // concurrent_preloads is one per-lane width for every GPU and DISK lane alike.
  cfg.concurrent_preloads = 3;
  for (bool const hw : {true, false}) {
    CHECK(h2g_cap(cfg, hw) == 3);
    CHECK(disk_cap(cfg, hw) == 3);
  }

  // Every copy into a GPU over the host link takes an h2g token.
  STATIC_REQUIRE(takes_h2g_token(Tier::HOST, Tier::GPU, false));
  STATIC_REQUIRE(takes_h2g_token(Tier::HOST, Tier::GPU, true));
  STATIC_REQUIRE(takes_h2g_token(Tier::DISK, Tier::GPU, false));
  STATIC_REQUIRE_FALSE(takes_h2g_token(Tier::DISK, Tier::GPU, true));
  STATIC_REQUIRE_FALSE(takes_h2g_token(Tier::DISK, Tier::HOST, false));
}

// ---- stats -------------------------------------------------------------------------------------

TEST_CASE("preload counters add, raise and snapshot", "[preload_policy]")
{
  STATIC_REQUIRE(counter_name(counter::events_received) == "events_received");
  STATIC_REQUIRE(counter_name(counter::wake_to_dispatch_us_max) == "wake_to_dispatch_us_max");
  STATIC_REQUIRE(counter_count == static_cast<std::size_t>(counter::wake_to_dispatch_us_max) + 1);

  preload_counters counters(2, 1);
  counters.add(counter::hit);
  counters.add(counter::hit, 2);
  counters.raise(counter::promote_us_max_h2g, 7);
  counters.raise(counter::promote_us_max_h2g, 3);
  counters.raise_lane_concurrency(1, 4);
  counters.raise_lane_concurrency(1, 2);
  counters.raise_lane_concurrency(9, 9);  // ignored
  counters.raise_gpu_inflight_bytes(0, 100);

  auto const s = counters.snapshot();
  CHECK(s[counter::hit] == 3);
  CHECK(s[counter::promote_us_max_h2g] == 7);
  CHECK(s.lane_concurrency_max == std::vector<std::uint64_t>{0, 4});
  CHECK(s.gpu_inflight_bytes_max == std::vector<std::uint64_t>{100});

  per_query_stats q;
  q.add(counter::bounce);
  CHECK(q[counter::bounce] == 1);
  CHECK(q[counter::hit] == 0);
}
