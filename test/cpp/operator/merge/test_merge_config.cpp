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

// Unit tests for the sort-merge sizing in merge_config: derived window rows, output chunk target,
// one-shot fraction and fan-in, the route choice, pass planning and the task memory estimate; and
// for the packing of MERGE_SORT output positions in merge_chunk_position. GPU-free.

#include "op/merge/bounded_kway_merge.hpp"
#include "op/merge/merge_chunk_position.hpp"
#include "op/merge/merge_config.hpp"
#include "sirius/exception.hpp"

#include <cudf/types.hpp>

#include <catch.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

using sirius::sort_merge_mode;
using sirius::op::merge::choose_route;
using sirius::op::merge::derive_merge_config;
using sirius::op::merge::device_budget;
using sirius::op::merge::estimate_task_bytes;
using sirius::op::merge::estimate_window_slot_bytes;
using sirius::op::merge::kBudgetFloorFraction;
using sirius::op::merge::kMaxKwayMergeInputs;
using sirius::op::merge::kMinExplicitWindowRows;
using sirius::op::merge::kOneShotRowLimit;
using sirius::op::merge::kWindowRowsFloor;
using sirius::op::merge::make_merge_chunk_position;
using sirius::op::merge::merge_route;
using sirius::op::merge::merge_rule;
using sirius::op::merge::merge_settings;
using sirius::op::merge::plan_pass;
using sirius::op::merge::round_rows_for;
using sirius::op::merge::route_inputs;
using sirius::op::merge::run_summary;
using sirius::op::merge::split_merge_chunk_position;

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;
constexpr std::size_t kGiB = std::size_t{1} << 30;
constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();

device_budget device(std::size_t gpu_bytes, std::size_t concurrent_tasks = 4)
{
  return {.gpu_available_bytes = gpu_bytes,
          .gpu_total_bytes     = gpu_bytes,
          .concurrent_tasks    = concurrent_tasks};
}

std::vector<run_summary> runs_of(std::vector<std::size_t> const& bytes)
{
  std::vector<run_summary> runs;
  for (auto b : bytes) {
    runs.push_back({.bytes = b, .rows = b / 32});
  }
  return runs;
}

/// Settings with @p rule switched off.
merge_settings without(merge_rule rule)
{
  merge_settings settings;
  settings.disabled_rules = static_cast<std::uint64_t>(rule);
  return settings;
}

/// What merging a partition to the end with plan_pass costs and looks like.
struct plan_trace {
  std::size_t rows_written{0};          ///< rows every merge wrote, the final pass included
  std::vector<std::size_t> merge_runs;  ///< runs each merge read, in order; the last is the final
};

/// Merges @p rows (the row count of each run; a run's bytes are its rows) the way the operator
/// does: plan_pass on the current runs, the selected runs become one run, until the final pass.
plan_trace run_plan(std::vector<std::size_t> rows, std::size_t fan_in, bool short_first_merge)
{
  plan_trace trace;
  while (true) {
    std::vector<run_summary> runs;
    for (auto r : rows) {
      runs.push_back({.bytes = r, .rows = r});
    }
    auto const plan = plan_pass(runs, fan_in, short_first_merge);
    std::size_t merged{0};
    for (auto i : plan.runs) {
      merged += rows[i];
    }
    trace.rows_written += merged;
    trace.merge_runs.push_back(plan.runs.size());
    if (plan.final_pass) { return trace; }
    // The intermediate run joins the partition's runs behind the ones that are left.
    std::vector<std::size_t> next;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (!std::binary_search(plan.runs.begin(), plan.runs.end(), i)) { next.push_back(rows[i]); }
    }
    next.push_back(merged);
    rows = std::move(next);
  }
}

/// The fewest rows any merge tree of fan-in at most @p fan_in writes for @p rows: the k-ary
/// Huffman cost, computed the textbook way (pad with empty runs until (R - 1) mod (f - 1) == 0,
/// then merge the f smallest until one run is left).
std::size_t huffman_rows_written(std::vector<std::size_t> rows, std::size_t fan_in)
{
  if (rows.size() <= 1) { return rows.empty() ? 0 : rows.front(); }
  while ((rows.size() - 1) % (fan_in - 1) != 0) {
    rows.push_back(0);
  }
  std::size_t written{0};
  while (rows.size() > 1) {
    std::sort(rows.begin(), rows.end());
    auto const merged = std::accumulate(
      rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(fan_in), std::size_t{0});
    written += merged;
    rows.erase(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(fan_in));
    rows.push_back(merged);
  }
  return written;
}

}  // namespace

TEST_CASE("derive_merge_config sizes a 16-run merge of 32-byte rows at T = 4",
          "[merge_config][unit]")
{
  struct row {
    std::size_t gpu_gib;
    double window_rows;
    std::size_t output_bytes;
    double fraction;
  };
  // B = G / 2 / T; n = B / (3 * k * w); output = clamp(B / 8, 64 MiB, 1 GiB); fraction by total G.
  auto const r = GENERATE(row{24, 2.1e6, 384 * kMiB, 0.25},
                          row{80, 7.0e6, 1 * kGiB, 0.33},
                          row{128, 11.2e6, 1 * kGiB, 0.33});

  auto const cfg =
    derive_merge_config(merge_settings{.max_fan_in = 16}, device(r.gpu_gib * kGiB), 16, 32);

  CHECK(cfg.fan_in == 16);
  CHECK(static_cast<double>(cfg.window_rows) == Approx(r.window_rows).epsilon(0.01));
  CHECK(static_cast<double>(cfg.output_target_bytes) ==
        Approx(static_cast<double>(r.output_bytes)).epsilon(0.01));
  CHECK(cfg.one_shot_fraction == Approx(r.fraction));
}

TEST_CASE("derive_merge_config caps the fan-in so the window stays at its floor",
          "[merge_config][unit]")
{
  // 4096 runs of 32-byte rows on a 24 GiB device: B = 3 GiB.
  SECTION("the max_fan_in setting caps it")
  {
    // The default cap, and an explicit one.
    auto const by_default = derive_merge_config(merge_settings{}, device(24 * kGiB), 4096, 32);
    CHECK(by_default.fan_in == sirius::config::DEFAULT_SORT_MERGE_MAX_FAN_IN);
    CHECK(by_default.window_rows ==
          3 * kGiB / (3 * sirius::config::DEFAULT_SORT_MERGE_MAX_FAN_IN * 32));

    auto const cfg =
      derive_merge_config(merge_settings{.max_fan_in = 64}, device(24 * kGiB), 4096, 32);
    CHECK(cfg.fan_in == 64);
    CHECK(cfg.window_rows == 3 * kGiB / (3 * 64 * 32));
    CHECK(cfg.window_rows >= kWindowRowsFloor);
  }

  SECTION("the task budget caps it below a large max_fan_in")
  {
    merge_settings settings;
    settings.max_fan_in = 4096;
    auto const cfg      = derive_merge_config(settings, device(24 * kGiB), 4096, 32);
    // 3 * F * 65536 * 32 <= 3 GiB gives F = 512; uncapped, the ceiling would be ~12 K rows.
    CHECK(cfg.fan_in == 512);
    CHECK(cfg.window_rows == kWindowRowsFloor);
  }

  SECTION("a tiny device keeps a fan-in of 2 and drops the window below the floor")
  {
    auto const cfg = derive_merge_config(merge_settings{}, device(32 * kMiB), 4096, 32);
    CHECK(cfg.fan_in == 2);
    // The ceiling G / (16 * F * w) wins over the floor.
    CHECK(cfg.window_rows == 32 * kMiB / (16 * 2 * 32));
    CHECK(cfg.window_rows < kWindowRowsFloor);
    CHECK(cfg.output_target_bytes == 64 * kMiB);
    CHECK(cfg.one_shot_fraction == Approx(0.20));
  }

  SECTION("the fan-in never exceeds the number of runs and the window is never zero")
  {
    CHECK(derive_merge_config(merge_settings{}, device(24 * kGiB), 3, 32).fan_in == 3);
    CHECK(derive_merge_config(merge_settings{}, device(24 * kGiB), 1, 32).fan_in == 1);

    auto const empty = derive_merge_config(merge_settings{}, device(0, 0), 0, 0);
    CHECK(empty.fan_in == 1);
    CHECK(empty.window_rows == 1);
    CHECK(empty.output_target_bytes == 64 * kMiB);
  }

  SECTION("a max_fan_in below 2 is treated as 2")
  {
    merge_settings settings;
    settings.max_fan_in = 0;
    CHECK(derive_merge_config(settings, device(24 * kGiB), 100, 32).fan_in == 2);
  }

  SECTION("a max_fan_in above the merge kernel's limit is clamped to it")
  {
    merge_settings settings;
    settings.max_fan_in = 1'000'000;
    // 1-byte rows: the task budget alone would allow far more than 4096 runs.
    CHECK(derive_merge_config(settings, device(80 * kGiB), 100'000, 1).fan_in ==
          kMaxKwayMergeInputs);
  }
}

TEST_CASE("derive_merge_config splits the budget among concurrent tasks", "[merge_config][unit]")
{
  // 80 GiB, k = 16, w = 32: B = 40 GiB / T.
  merge_settings const settings{.max_fan_in = 16};
  auto const one  = derive_merge_config(settings, device(80 * kGiB, 1), 16, 32);
  auto const four = derive_merge_config(settings, device(80 * kGiB, 4), 16, 32);
  // With T = 1 the fit (~28 M rows) exceeds the ceiling G / (16 * k * w) = 10.5 M rows.
  CHECK(one.window_rows == 80 * kGiB / (16 * 16 * 32));
  CHECK(four.window_rows == 10 * kGiB / (3 * 16 * 32));
  CHECK(one.output_target_bytes == 1 * kGiB);
  CHECK(four.output_target_bytes == 1 * kGiB);
  // Without the round_rows rule the output target is B / 16: a round is then at most a window.
  CHECK(derive_merge_config(without(merge_rule::round_rows), device(80 * kGiB, 4), 16, 32)
          .output_target_bytes == 640 * kMiB);
  CHECK(derive_merge_config(without(merge_rule::round_rows), device(24 * kGiB, 4), 16, 32)
          .output_target_bytes == 192 * kMiB);

  // Zero concurrent tasks is treated as one.
  auto const zero = derive_merge_config(settings, device(80 * kGiB, 0), 16, 32);
  CHECK(zero.window_rows == one.window_rows);
  CHECK(zero.output_target_bytes == one.output_target_bytes);
}

TEST_CASE("derive_merge_config sizes from at least a quarter of the device", "[merge_config][unit]")
{
  // An 80 GiB device, k = 8, w = 16, T = 4.
  auto const budget_with = [](std::size_t available) {
    return device_budget{
      .gpu_available_bytes = available, .gpu_total_bytes = 80 * kGiB, .concurrent_tasks = 4};
  };
  auto const window = [&](std::size_t available, const merge_settings& settings = {}) {
    return derive_merge_config(settings, budget_with(available), 8, 16).window_rows;
  };
  // G_eff = max(available, 20 GiB); B = G_eff / 8; n = B / (3 * 8 * 16).
  auto const floor_bytes  = 20 * kGiB;
  auto const floor_window = (floor_bytes / 8) / (3 * 8 * 16);

  SECTION("a device full of the merge's own runs is sized from the floor")
  {
    // What the readings of one larger-than-GPU query looked like: anything from nothing free
    // to a few GiB. They all give the same sizes.
    for (auto available : {std::size_t{0}, 1 * kGiB, 8 * kGiB, 19 * kGiB, 20 * kGiB}) {
      CHECK(window(available) == floor_window);
      CHECK(
        derive_merge_config(merge_settings{}, budget_with(available), 8, 16).output_target_bytes ==
        (floor_bytes / 8) / 8);
    }
    CHECK(kBudgetFloorFraction == 0.25);
    // A reading above the floor is used as it is.
    CHECK(window(21 * kGiB) > floor_window);
  }

  SECTION("more free memory than the floor is used as before")
  {
    CHECK(window(48 * kGiB) == 6 * kGiB / (3 * 8 * 16));
    CHECK(window(80 * kGiB) == 10 * kGiB / (3 * 8 * 16));
    CHECK(window(48 * kGiB) == window(48 * kGiB, without(merge_rule::budget_floor)));
  }

  SECTION("without the rule the sizes follow the reading")
  {
    auto const off = without(merge_rule::budget_floor);
    CHECK(window(8 * kGiB, off) == 1 * kGiB / (3 * 8 * 16));
    CHECK(window(1 * kGiB, off) == (kGiB / 8) / (3 * 8 * 16));
    CHECK(window(8 * kGiB, off) < window(8 * kGiB));
  }

  SECTION("the ceiling follows the floor too")
  {
    // T = 1: the fit (20 GiB / 2 / (3 * 8 * 16)) is above the ceiling G_eff / (16 * k * w).
    device_budget const alone{
      .gpu_available_bytes = 0, .gpu_total_bytes = 80 * kGiB, .concurrent_tasks = 1};
    CHECK(derive_merge_config(merge_settings{}, alone, 8, 16).window_rows ==
          20 * kGiB / (16 * 8 * 16));
  }

  SECTION("a device of unknown size has no floor")
  {
    device_budget const unknown{
      .gpu_available_bytes = 24 * kGiB, .gpu_total_bytes = 0, .concurrent_tasks = 4};
    // 16 runs in one merge: above the default fan-in, so the cap is set to it.
    CHECK(derive_merge_config(merge_settings{.max_fan_in = 16}, unknown, 16, 32).window_rows ==
          3 * kGiB / (3 * 16 * 32));
  }
}

TEST_CASE("derive_merge_config derives the one-shot fraction from the total GPU memory",
          "[merge_config][unit]")
{
  auto fraction = [](std::size_t total) {
    device_budget budget{
      .gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = total, .concurrent_tasks = 1};
    return derive_merge_config(merge_settings{}, budget, 16, 32).one_shot_fraction;
  };
  CHECK(fraction(16 * kGiB - 1) == Approx(0.20));
  CHECK(fraction(16 * kGiB) == Approx(0.25));
  CHECK(fraction(64 * kGiB - 1) == Approx(0.25));
  CHECK(fraction(64 * kGiB) == Approx(0.33));
}

TEST_CASE("derive_merge_config keeps explicit settings", "[merge_config][unit]")
{
  merge_settings settings;
  settings.window_rows           = 1'000'000;
  settings.output_bytes          = 123 * kMiB;
  settings.one_shot_max_fraction = 0.5;
  settings.max_fan_in            = 4;

  auto const cfg = derive_merge_config(settings, device(24 * kGiB), 16, 32);
  CHECK(cfg.window_rows == 1'000'000);
  CHECK(cfg.output_target_bytes == 123 * kMiB);
  CHECK(cfg.one_shot_fraction == 0.5);
  CHECK(cfg.fan_in == 4);

  SECTION("an explicit window below the accepted minimum is clamped up")
  {
    settings.window_rows = 10;
    CHECK(derive_merge_config(settings, device(24 * kGiB), 16, 32).window_rows ==
          kMinExplicitWindowRows);
  }

  SECTION("an explicit window is kept even when the device is tiny")
  {
    CHECK(derive_merge_config(settings, device(64 * kMiB), 16, 32).window_rows == 1'000'000);
  }
}

TEST_CASE("derive_merge_config keeps the window within the cudf row limit", "[merge_config][unit]")
{
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());

  SECTION("derived: 2 runs of 1-byte rows on an 80 GiB device")
  {
    // Unclamped, the ceiling G / (16 * k * w) would give 2.7e9 rows.
    auto const cfg = derive_merge_config(merge_settings{}, device(80 * kGiB, 1), 2, 1);
    CHECK(cfg.window_rows == limit);
  }

  SECTION("explicit")
  {
    merge_settings settings;
    settings.window_rows = 3'000'000'000;
    CHECK(derive_merge_config(settings, device(24 * kGiB), 16, 32).window_rows == limit);
    settings.window_rows = limit;
    CHECK(derive_merge_config(settings, device(24 * kGiB), 16, 32).window_rows == limit);
  }
}

TEST_CASE("choose_route follows the mode, the inputs and the one-shot threshold",
          "[merge_config][unit]")
{
  constexpr std::size_t total = 100 * kGiB;
  constexpr double fraction   = 0.25;
  route_inputs const small{.partition_bytes = 10 * kGiB, .output_bytes = 10 * kGiB};
  route_inputs const large{.partition_bytes = 20 * kGiB, .output_bytes = 20 * kGiB};

  SECTION("forced modes ignore the size")
  {
    CHECK(choose_route(sort_merge_mode::one_shot, fraction, total, large) == merge_route::one_shot);
    CHECK(choose_route(sort_merge_mode::windowed, fraction, total, small) == merge_route::windowed);
  }

  SECTION("automatic compares partition + output against the threshold")
  {
    CHECK(choose_route(sort_merge_mode::automatic, fraction, total, small) ==
          merge_route::one_shot);
    CHECK(choose_route(sort_merge_mode::automatic, fraction, total, large) ==
          merge_route::windowed);

    // Exactly at the threshold stays one-shot; one byte over goes windowed.
    route_inputs at{.partition_bytes = 20 * kGiB, .output_bytes = 5 * kGiB};
    CHECK(choose_route(sort_merge_mode::automatic, fraction, total, at) == merge_route::one_shot);
    at.output_bytes += 1;
    CHECK(choose_route(sort_merge_mode::automatic, fraction, total, at) == merge_route::windowed);

    route_inputs const huge{.partition_bytes = kMax, .output_bytes = kMax};
    CHECK(choose_route(sort_merge_mode::automatic, 1.0, total, huge) == merge_route::windowed);
  }

  SECTION("automatic goes windowed when any input is off the GPU")
  {
    auto inputs        = small;
    inputs.any_non_gpu = true;
    CHECK(choose_route(sort_merge_mode::automatic, fraction, total, inputs) ==
          merge_route::windowed);
    CHECK(choose_route(sort_merge_mode::one_shot, fraction, total, inputs) ==
          merge_route::one_shot);
  }

  SECTION("nested and late-materialized columns force one-shot in every mode")
  {
    auto const mode =
      GENERATE(sort_merge_mode::automatic, sort_merge_mode::windowed, sort_merge_mode::one_shot);
    auto nested        = large;
    nested.has_nested  = true;
    nested.any_non_gpu = true;
    CHECK(choose_route(mode, fraction, total, nested) == merge_route::one_shot);

    auto late_mat         = large;
    late_mat.has_late_mat = true;
    late_mat.any_non_gpu  = true;
    CHECK(choose_route(mode, fraction, total, late_mat) == merge_route::one_shot);
  }
}

TEST_CASE("plan_pass merges everything when the runs fit the fan-in", "[merge_config][unit]")
{
  auto const runs = runs_of({500, 100, 300});

  auto const exact = plan_pass(runs, 3);
  CHECK(exact.final_pass);
  CHECK(exact.runs == std::vector<std::size_t>{0, 1, 2});

  auto const roomy = plan_pass(runs, 64);
  CHECK(roomy.final_pass);
  CHECK(roomy.runs == std::vector<std::size_t>{0, 1, 2});

  auto const none = plan_pass({}, 4);
  CHECK(none.final_pass);
  CHECK(none.runs.empty());
}

TEST_CASE("plan_pass picks the smallest runs for an intermediate pass", "[merge_config][unit]")
{
  SECTION("the fan_in smallest by bytes, indices ascending")
  {
    // Seven runs at fan-in 3: (7 - 1) mod 2 == 0, so the merge is a full one.
    auto const runs = runs_of({500, 100, 400, 50, 300, 200, 600});
    auto const plan = plan_pass(runs, 3);
    CHECK_FALSE(plan.final_pass);
    CHECK(plan.runs == std::vector<std::size_t>{1, 3, 5});
  }

  SECTION("a short first merge when the run count leaves a remainder")
  {
    // Six runs at fan-in 3: (6 - 1) mod 2 == 1, so the first merge takes ((6 - 2) mod 2) + 2 = 2
    // runs; the five runs left are then merged 3 + 3.
    auto const runs = runs_of({500, 100, 400, 50, 300, 200});
    auto const plan = plan_pass(runs, 3);
    CHECK_FALSE(plan.final_pass);
    CHECK(plan.runs == std::vector<std::size_t>{1, 3});

    auto const legacy = plan_pass(runs, 3, false);
    CHECK_FALSE(legacy.final_pass);
    CHECK(legacy.runs == std::vector<std::size_t>{1, 3, 5});
  }

  SECTION("ties go to the lower index")
  {
    auto const runs = runs_of({100, 100, 50, 100, 100});
    auto const plan = plan_pass(runs, 3);
    CHECK_FALSE(plan.final_pass);
    CHECK(plan.runs == std::vector<std::size_t>{0, 1, 2});
  }

  SECTION("a fan-in below 2 still merges two runs")
  {
    auto const runs = runs_of({300, 100, 200});
    auto const plan = plan_pass(runs, 1);
    CHECK_FALSE(plan.final_pass);
    CHECK(plan.runs == std::vector<std::size_t>{1, 2});
  }
}

TEST_CASE("plan_pass writes the fewest rows a fan-in allows", "[merge_config][unit]")
{
  SECTION("equal runs: rows written for a table of (R, f)")
  {
    struct row {
      std::size_t runs;
      std::size_t fan_in;
      std::size_t written;         ///< with the short first merge
      std::size_t written_legacy;  ///< always fan_in runs
    };
    // One row per run, so `written` is also writes per row times R. The f = 8 rows are the
    // run counts of the measurements: 268 runs were written 3.597 times each, now 2.873.
    auto const r = GENERATE(row{9, 8, 11, 17},
                            row{65, 8, 132, 186},
                            row{267, 8, 766, 766},
                            row{268, 8, 770, 964},
                            row{269, 8, 773, 895},
                            row{274, 8, 788, 788},
                            row{513, 8, 1541, 1979},
                            row{268, 4, 1088, 1088},
                            row{269, 4, 1094, 1264},
                            row{5, 4, 7, 9},
                            row{3, 2, 5, 5},
                            row{100, 2, 672, 672},
                            row{17, 16, 19, 33});
    CAPTURE(r.runs, r.fan_in);
    std::vector<std::size_t> const rows(r.runs, 1);

    auto const plan = run_plan(rows, r.fan_in, true);
    CHECK(plan.rows_written == r.written);
    CHECK(plan.rows_written == huffman_rows_written(rows, r.fan_in));
    CHECK(run_plan(rows, r.fan_in, false).rows_written == r.written_legacy);
    CHECK(r.written <= r.written_legacy);
  }

  SECTION("every merge after the first is full, and so is the final pass")
  {
    auto const fan_in = GENERATE(std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{8});
    for (std::size_t runs = fan_in + 1; runs <= 6 * fan_in + 3; ++runs) {
      CAPTURE(runs, fan_in);
      auto const plan = run_plan(std::vector<std::size_t>(runs, 1), fan_in, true);
      REQUIRE(plan.merge_runs.size() >= 2);
      auto const remainder = (runs - 1) % (fan_in - 1);
      CHECK(plan.merge_runs.front() == (remainder == 0 ? fan_in : (runs - 2) % (fan_in - 1) + 2));
      CHECK(plan.merge_runs.front() >= 2);
      CHECK(plan.merge_runs.front() <= fan_in);
      for (std::size_t i = 1; i < plan.merge_runs.size(); ++i) {
        CHECK(plan.merge_runs[i] == fan_in);
      }
    }
  }

  SECTION("R = f + 1 merges two runs and then all of them")
  {
    auto const plan = run_plan(std::vector<std::size_t>(9, 100), 8, true);
    CHECK(plan.merge_runs == std::vector<std::size_t>{2, 8});
    // Without the rule: eight runs, then a final pass of two behind a rewrite of 8/9 of the rows.
    auto const legacy = run_plan(std::vector<std::size_t>(9, 100), 8, false);
    CHECK(legacy.merge_runs == std::vector<std::size_t>{8, 2});
  }

  SECTION("unequal runs: the plan is the Huffman optimum for any sizes")
  {
    auto const fan_in   = GENERATE(std::size_t{2}, std::size_t{3}, std::size_t{5}, std::size_t{8});
    std::uint64_t state = 12345;
    auto const next     = [&state] {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      return static_cast<std::size_t>(state >> 33);
    };
    for (std::size_t runs = 1; runs <= 70; ++runs) {
      CAPTURE(runs, fan_in);
      // Sizes over three orders of magnitude, with one run fifty times the largest of the rest.
      std::vector<std::size_t> rows;
      for (std::size_t i = 0; i < runs; ++i) {
        rows.push_back(1 + next() % 1000);
      }
      rows[runs / 2]  = 50'000;
      auto const plan = run_plan(rows, fan_in, true);
      CHECK(plan.rows_written == huffman_rows_written(rows, fan_in));
      CHECK(plan.rows_written <= run_plan(rows, fan_in, false).rows_written);
    }
  }

  SECTION("an intermediate run that was written keeps the count on the Huffman track")
  {
    // The operator plans again after each intermediate run: with 268 runs at fan-in 8 the calls
    // see 268, 267, 260, ... runs. Only the first has a remainder.
    std::vector<std::size_t> seen;
    std::vector<std::size_t> rows(268, 1);
    while (true) {
      std::vector<run_summary> runs;
      for (auto r : rows) {
        runs.push_back({.bytes = r, .rows = r});
      }
      auto const plan = plan_pass(runs, 8);
      seen.push_back(plan.runs.size());
      if (plan.final_pass) { break; }
      std::size_t merged{0};
      std::vector<std::size_t> next;
      for (std::size_t i = 0; i < rows.size(); ++i) {
        if (std::binary_search(plan.runs.begin(), plan.runs.end(), i)) {
          merged += rows[i];
        } else {
          next.push_back(rows[i]);
        }
      }
      next.push_back(merged);
      rows = std::move(next);
    }
    CHECK(seen.front() == 2);
    CHECK(std::all_of(seen.begin() + 1, seen.end(), [](auto n) { return n == 8; }));
    // 1 short merge + 37 full ones + the final pass.
    CHECK(seen.size() == 39);
  }
}

TEST_CASE("choose_route windows a partition a one-shot merge cannot hold in one table",
          "[merge_config][unit]")
{
  constexpr std::size_t total = 160 * kGiB;
  constexpr double fraction   = 0.33;
  // Narrow rows: 2^31 rows of 8 bytes are 16 GiB, under the byte threshold of a 160 GiB device.
  route_inputs inputs{.partition_bytes = 16 * kGiB, .output_bytes = 16 * kGiB};

  inputs.partition_rows = kOneShotRowLimit - 1;
  CHECK(choose_route(sort_merge_mode::automatic, fraction, total, inputs) == merge_route::one_shot);

  inputs.partition_rows = kOneShotRowLimit;
  CHECK(choose_route(sort_merge_mode::automatic, fraction, total, inputs) == merge_route::windowed);
  inputs.partition_rows = std::size_t{1} << 33;
  CHECK(choose_route(sort_merge_mode::automatic, fraction, total, inputs) == merge_route::windowed);

  // The guard is a rule of the automatic mode only, and it can be switched off.
  CHECK(choose_route(sort_merge_mode::automatic, fraction, total, inputs, false) ==
        merge_route::one_shot);
  CHECK(choose_route(sort_merge_mode::one_shot, fraction, total, inputs) == merge_route::one_shot);
  // Nested and late-materialized columns have no windowed path.
  auto nested       = inputs;
  nested.has_nested = true;
  CHECK(choose_route(sort_merge_mode::automatic, fraction, total, nested) == merge_route::one_shot);
  // An unknown row count (0) does not trigger it.
  inputs.partition_rows = 0;
  CHECK(choose_route(sort_merge_mode::automatic, fraction, total, inputs) == merge_route::one_shot);
  CHECK(kOneShotRowLimit == static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()));
}

TEST_CASE("round_rows_for sizes a round from the output target", "[merge_config][unit]")
{
  // 512 MiB of 16-byte rows.
  CHECK(round_rows_for(16'000'000, 512 * kMiB, 16) == 512 * kMiB / 16);
  // Rounded up, so that one round reaches the target.
  CHECK(round_rows_for(1000, 100, 16) == 1000);
  CHECK(round_rows_for(1, 100, 16) == 7);
  // Never below the window: a round emitted a window's worth of rows before the rule.
  CHECK(round_rows_for(50'000'000, 512 * kMiB, 16) == 50'000'000);
  // At most half the cudf row limit, so that the two rounds a task may concatenate fit one
  // table; and defined for a zero row width.
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());
  CHECK(sirius::op::merge::kMaxRoundRows == limit / 2);
  CHECK(round_rows_for(1000, kMax, 1) == limit / 2);
  CHECK(round_rows_for(1000, 4 * kGiB, 0) == limit / 2);
  CHECK(round_rows_for(limit, 1000, 16) == limit / 2);
  CHECK(2 * round_rows_for(1000, kGiB, 1) <= limit);
  CHECK(round_rows_for(1000, 1000, 0) == 1000);
}

TEST_CASE("estimate_window_slot_bytes follows what a window buffer allocates",
          "[merge_config][unit]")
{
  constexpr std::size_t n = 100'000;

  SECTION("fixed-width columns: the rows and a mask per column")
  {
    CHECK(estimate_window_slot_bytes(n, 32, 0, 0) == n * 32);
    CHECK(estimate_window_slot_bytes(n, 32, 5, 0) == n * 32 + 5 * (n / 8 + 64));
    // The STRING width means nothing without a STRING column.
    CHECK(estimate_window_slot_bytes(n, 32, 5, 0, 20) == n * 32 + 5 * (n / 8 + 64));
  }

  SECTION("STRING columns of a known width: slack on their chars only, 4-byte offsets")
  {
    // Three columns, two of them STRING, 12 + 20 bytes wide of a 40-byte row.
    constexpr std::size_t w = 40, sw = 32, columns = 3, strings = 2;
    auto const chars = n * (sw + strings);
    CHECK(estimate_window_slot_bytes(n, w, columns, strings, sw) ==
          n * (w - sw) + columns * (n / 8 + 64) + chars + chars / 4 + strings * 64 +
            strings * (n + 1) * 4);
    // Less than the whole-row slack with 8-byte offsets.
    CHECK(estimate_window_slot_bytes(n, w, columns, strings, sw) <
          estimate_window_slot_bytes(n, w, columns, strings));
  }

  SECTION("chars that could pass 2 GiB take 8-byte offsets")
  {
    constexpr std::size_t rows = 100'000'000, w = 40, sw = 32;
    auto const chars = rows * (sw + 1);
    CHECK(estimate_window_slot_bytes(rows, w, 2, 1, sw) ==
          rows * (w - sw) + 2 * (rows / 8 + 64) + chars + chars / 4 + 64 + (rows + 1) * 8);
  }

  SECTION("an unknown or inconsistent STRING width keeps the whole-row slack")
  {
    constexpr std::size_t w = 40, columns = 3, strings = 2;
    auto const legacy =
      n * w + columns * (n / 8 + 64) + n * w / 4 + strings * ((n + 1) * 8 + 2 * n + 64);
    CHECK(estimate_window_slot_bytes(n, w, columns, strings) == legacy);
    CHECK(estimate_window_slot_bytes(n, w, columns, strings, 0) == legacy);
    CHECK(estimate_window_slot_bytes(n, w, columns, strings, w + 1) == legacy);
  }

  SECTION("huge inputs saturate")
  {
    CHECK(estimate_window_slot_bytes(kMax, 32, 3, 1, 16) == kMax);
    CHECK(estimate_window_slot_bytes(n, kMax, 3, 1, kMax) == kMax);
    CHECK(estimate_window_slot_bytes(n, 32, kMax, 1, 16) == kMax);
  }
}

TEST_CASE("estimate_task_bytes sizes the output and the scratch from the round",
          "[merge_config][unit]")
{
  constexpr std::size_t k = 8, n = 100'000, round = 1'000'000, w = 16;
  constexpr std::size_t out     = 64 * kMiB;
  auto const output_and_scratch = [&](std::size_t rows) {
    return 2 * (out + rows * w) + rows * 8 + (rows / 2048 + 2) * (4 + 12 * k);
  };

  // Every run sliced in place: only the round's output and scratch.
  CHECK(estimate_task_bytes(k, k, n, w, 0, out, 0, 0, 0, round) == output_and_scratch(round));
  // The window buffers stay sized by the window.
  CHECK(estimate_task_bytes(k, 5, n, w, 0, out, 0, 0, 0, round) ==
        2 * 3 * n * w + output_and_scratch(round));
  // 0 means a round of a window's rows, as before the rule.
  CHECK(estimate_task_bytes(k, 5, n, w, 0, out, 0, 0, 0, 0) ==
        2 * 3 * n * w + output_and_scratch(n));
  CHECK(estimate_task_bytes(k, 5, n, w, 0, out, 0, 0, 0, n) ==
        estimate_task_bytes(k, 5, n, w, 0, out, 0));
  // The STRING width reaches the slot estimate.
  CHECK(estimate_task_bytes(k, 5, n, 40, 0, out, 0, 3, 2, 0, 32) ==
        2 * 3 * estimate_window_slot_bytes(n, 40, 3, 2, 32) + 2 * (out + n * 40) + n * 8 +
          (n / 2048 + 2) * (4 + 12 * k));
  CHECK(estimate_task_bytes(k, k, n, w, 0, out, 0, 0, 0, kMax) == kMax);
}

TEST_CASE("merge_settings switches each rule off by its bit", "[merge_config][unit]")
{
  merge_settings settings;
  CHECK(settings.disabled_rules == sirius::config::DEFAULT_SORT_MERGE_DISABLED_RULES);
  // By default every rule is in force but the one-shot cold-start estimate.
  for (auto rule : {merge_rule::huffman_plan,
                    merge_rule::budget_floor,
                    merge_rule::partition_budget,
                    merge_rule::in_place_reservation,
                    merge_rule::round_rows,
                    merge_rule::row_count_guard,
                    merge_rule::single_run_passthrough,
                    merge_rule::string_slot_estimate}) {
    CHECK(settings.enabled(rule));
  }
  CHECK_FALSE(settings.enabled(merge_rule::one_shot_estimate));
  for (auto rule : {merge_rule::huffman_plan,
                    merge_rule::budget_floor,
                    merge_rule::partition_budget,
                    merge_rule::in_place_reservation,
                    merge_rule::round_rows,
                    merge_rule::row_count_guard,
                    merge_rule::one_shot_estimate,
                    merge_rule::single_run_passthrough,
                    merge_rule::string_slot_estimate}) {
    settings.disabled_rules = 0;
    CHECK(settings.enabled(rule));
    settings.disabled_rules = static_cast<std::uint64_t>(rule);
    CHECK_FALSE(settings.enabled(rule));
    settings.disabled_rules =
      sirius::config::sort_merge_rule::ALL & ~static_cast<std::uint64_t>(rule);
    CHECK(settings.enabled(rule));
  }

  sirius::operator_params params;
  params.sort_merge_disabled_rules =
    sirius::config::sort_merge_rule::HUFFMAN_PLAN | sirius::config::sort_merge_rule::ROUND_ROWS;
  auto const from = merge_settings::from(params);
  CHECK_FALSE(from.enabled(merge_rule::huffman_plan));
  CHECK_FALSE(from.enabled(merge_rule::round_rows));
  CHECK(from.enabled(merge_rule::budget_floor));
}

TEST_CASE("estimate_reader_state_bytes covers the decode scratch of compressed runs",
          "[merge_config][unit]")
{
  using sirius::op::merge::estimate_reader_state_bytes;
  constexpr std::size_t runs = 6, columns = 9, strings = 2, n = 100'000;

  // Two int64 per row of the chunks covering a window, per column; 2 MiB per STRING column.
  CHECK(estimate_reader_state_bytes(runs, columns, strings, n) ==
        runs * (columns * 16 * (n + 2048) + strings * 2 * kMiB));
  CHECK(estimate_reader_state_bytes(0, columns, strings, n) == 0);
  CHECK(estimate_reader_state_bytes(runs, 0, 0, n) == 0);
  CHECK(estimate_reader_state_bytes(kMax, columns, strings, n) == kMax);
  CHECK(estimate_reader_state_bytes(runs, columns, strings, kMax) == kMax);
}

TEST_CASE("estimate_task_bytes sums window buffers, output, merge scratch and fallback batches",
          "[merge_config][unit]")
{
  constexpr std::size_t k = 16, g = 4, n = 100'000, w = 32, kw = 12;
  constexpr std::size_t out = 64 * kMiB, fallback = 10 * kMiB;
  // The round outputs (target plus one round) and their concatenation; the gather map; a rank and
  // three values per run at each of the n / 2048 + 2 tile boundaries; the fallback batches.
  constexpr std::size_t fixed =
    2 * (out + n * w) + n * 8 + (n / 2048 + 2) * (4 + 12 * k) + fallback;

  SECTION("fixed-width rows: two slots of n rows per buffered run")
  {
    CHECK(estimate_task_bytes(k, g, n, w, kw, out, fallback) == 2 * (k - g) * n * w + fixed);
  }

  SECTION("every column adds a validity mask to each slot")
  {
    constexpr std::size_t columns = 5;
    auto const slot               = n * w + columns * (n / 8 + 64);
    CHECK(estimate_task_bytes(k, g, n, w, kw, out, fallback, columns) ==
          2 * (k - g) * slot + fixed);
  }

  SECTION("STRING columns add chars slack and offsets to each slot")
  {
    constexpr std::size_t columns = 5, strings = 2;
    auto const slot =
      n * w + columns * (n / 8 + 64) + n * w / 4 + strings * ((n + 1) * 8 + 2 * n + 64);
    CHECK(estimate_task_bytes(k, g, n, w, kw, out, fallback, columns, strings) ==
          2 * (k - g) * slot + fixed);
  }

  SECTION("the key width does not change the estimate")
  {
    CHECK(estimate_task_bytes(k, g, n, w, 0, out, fallback) ==
          estimate_task_bytes(k, g, n, w, 1000, out, fallback));
  }

  SECTION("the fallback bytes are added as given")
  {
    CHECK(estimate_task_bytes(k, g, n, w, kw, out, 3 * fallback) ==
          estimate_task_bytes(k, g, n, w, kw, out, 0) + 3 * fallback);
  }

  SECTION("all runs on the GPU need no window buffers")
  {
    CHECK(estimate_task_bytes(k, k, n, w, kw, out, fallback, 5, 2) == fixed);
    // More GPU runs than runs is clamped.
    CHECK(estimate_task_bytes(k, k + 5, n, w, kw, out, fallback, 5, 2) == fixed);
  }

  SECTION("huge inputs saturate")
  {
    CHECK(estimate_task_bytes(kMax, 0, kMax, kMax, kMax, kMax, kMax, kMax, kMax) == kMax);
    CHECK(estimate_task_bytes(k, g, kMax / 2, w, kw, out, fallback) == kMax);
    CHECK(estimate_task_bytes(k, g, n, w, kw, kMax, fallback) == kMax);
    CHECK(estimate_task_bytes(k, g, n, w, kw, out, kMax) == kMax);
    CHECK(estimate_task_bytes(k, g, n, w, kw, out, fallback, kMax, 1) == kMax);
    CHECK(estimate_task_bytes(k, g, n, w, kw, out, fallback, 1, kMax) == kMax);
    // Only the two tile boundaries of an empty merge remain.
    CHECK(estimate_task_bytes(0, 0, 0, 0, 0, 0, 0) == 2 * 4);
  }
}

TEST_CASE("make_merge_chunk_position packs partition and chunk and rejects values >= 2^32",
          "[merge_chunk_position][unit]")
{
  constexpr std::size_t last = (std::size_t{1} << 32) - 1;

  SECTION("round trip, up to the largest values")
  {
    CHECK(make_merge_chunk_position(0, 0) == 0);
    CHECK(make_merge_chunk_position(3, 7) == ((std::size_t{3} << 32) | 7));
    CHECK(split_merge_chunk_position(make_merge_chunk_position(3, 7)) ==
          std::pair<std::size_t, std::size_t>{3, 7});
    CHECK(make_merge_chunk_position(last, last) == kMax);
    CHECK(split_merge_chunk_position(make_merge_chunk_position(last, last)) ==
          std::pair<std::size_t, std::size_t>{last, last});
    CHECK(split_merge_chunk_position(make_merge_chunk_position(0, last)) ==
          std::pair<std::size_t, std::size_t>{0, last});
    CHECK(split_merge_chunk_position(make_merge_chunk_position(last, 0)) ==
          std::pair<std::size_t, std::size_t>{last, 0});
  }

  SECTION("positions order by partition, then by chunk")
  {
    CHECK(make_merge_chunk_position(1, 0) < make_merge_chunk_position(1, 1));
    CHECK(make_merge_chunk_position(1, last) < make_merge_chunk_position(2, 0));
    CHECK(make_merge_chunk_position(0, last) < make_merge_chunk_position(1, 0));
  }

  SECTION("a partition or chunk of 2^32 or more throws")
  {
    CHECK_THROWS_AS(make_merge_chunk_position(last + 1, 0), sirius::internal_exception);
    CHECK_THROWS_AS(make_merge_chunk_position(0, last + 1), sirius::internal_exception);
    CHECK_THROWS_AS(make_merge_chunk_position(kMax, kMax), sirius::internal_exception);
  }
}
