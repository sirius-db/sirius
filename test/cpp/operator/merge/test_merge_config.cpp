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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

using sirius::sort_merge_mode;
using sirius::op::merge::choose_route;
using sirius::op::merge::derive_merge_config;
using sirius::op::merge::device_budget;
using sirius::op::merge::estimate_task_bytes;
using sirius::op::merge::kMaxKwayMergeInputs;
using sirius::op::merge::kMinExplicitWindowRows;
using sirius::op::merge::kWindowRowsFloor;
using sirius::op::merge::make_merge_chunk_position;
using sirius::op::merge::merge_route;
using sirius::op::merge::merge_settings;
using sirius::op::merge::plan_pass;
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
  // B = G / 2 / T; n = B / (3 * k * w); output = clamp(B / 16, 64 MiB, 1 GiB); fraction by total G.
  auto const r = GENERATE(row{24, 2.1e6, 192 * kMiB, 0.25},
                          row{80, 7.0e6, 640 * kMiB, 0.33},
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
  CHECK(four.output_target_bytes == 640 * kMiB);

  // Zero concurrent tasks is treated as one.
  auto const zero = derive_merge_config(settings, device(80 * kGiB, 0), 16, 32);
  CHECK(zero.window_rows == one.window_rows);
  CHECK(zero.output_target_bytes == one.output_target_bytes);
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
    auto const runs = runs_of({500, 100, 400, 50, 300, 200});
    auto const plan = plan_pass(runs, 3);
    CHECK_FALSE(plan.final_pass);
    CHECK(plan.runs == std::vector<std::size_t>{1, 3, 5});
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
