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

#include "op/merge/merge_config.hpp"

#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/merge/bounded_kway_merge.hpp"

#include <algorithm>
#include <numeric>

namespace sirius::op::merge {

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;
constexpr std::size_t kGiB = std::size_t{1} << 30;

constexpr std::size_t kMinOutputTargetBytes = 64 * kMiB;
constexpr std::size_t kMaxOutputTargetBytes = kGiB;

/// Share of the available GPU memory the merge tasks of a device may use together.
constexpr double kTaskBudgetFraction = 0.5;
/// Windows a task budgets per input: the window itself, merge scratch and output.
constexpr std::size_t kWindowCopies = 3;
/// A task's windows together stay under 1/kWindowCeilingDivisor of the available GPU memory.
constexpr std::size_t kWindowCeilingDivisor = 16;
/// The output chunk target is 1/kOutputBudgetDivisor of the task budget.
constexpr std::size_t kOutputBudgetDivisor = 16;

using sirius::memory::saturating_add;
using sirius::memory::saturating_mul;

/// The largest fan-in whose windows fit the task budget with n at its floor, within [2, max].
/// The merge kernel takes at most kMaxKwayMergeInputs windows, whatever the setting says.
std::size_t fan_in_cap(std::size_t task_budget, std::size_t row_width, std::size_t max_fan_in)
{
  auto const max_f = std::clamp<std::size_t>(max_fan_in, 2, kMaxKwayMergeInputs);
  auto const bytes_per_run =
    saturating_mul(saturating_mul(kWindowCopies, kWindowRowsFloor), row_width);
  return std::clamp<std::size_t>(task_budget / bytes_per_run, 2, max_f);
}

double derive_one_shot_fraction(std::size_t gpu_total_bytes) noexcept
{
  if (gpu_total_bytes < 16 * kGiB) { return 0.20; }
  if (gpu_total_bytes < 64 * kGiB) { return 0.25; }
  return 0.33;
}

}  // namespace

derived_merge_config derive_merge_config(const merge_settings& settings,
                                         const device_budget& budget,
                                         std::size_t num_runs,
                                         std::size_t row_width)
{
  auto const g = budget.gpu_available_bytes;
  auto const t = std::max<std::size_t>(budget.concurrent_tasks, 1);
  auto const w = std::max<std::size_t>(row_width, 1);
  auto const b = static_cast<std::size_t>(static_cast<double>(g) * kTaskBudgetFraction) / t;

  // A partition with fewer runs than the cap reads them all; fan_in stays at least 1 so the
  // divisions below are defined for an empty partition.
  auto const fan_in =
    std::max<std::size_t>(std::min(fan_in_cap(b, w, settings.max_fan_in), num_runs), 1);

  std::size_t window_rows{};
  if (settings.window_rows != 0) {
    window_rows = std::max<std::size_t>(settings.window_rows, kMinExplicitWindowRows);
  } else {
    auto const run_bytes = saturating_mul(fan_in, w);
    auto const fit       = b / saturating_mul(kWindowCopies, run_bytes);
    auto const ceiling   = g / saturating_mul(kWindowCeilingDivisor, run_bytes);
    // The ceiling is applied last, so it wins when it is below the floor: memory comes first.
    window_rows = std::max<std::size_t>(std::min(std::max(fit, kWindowRowsFloor), ceiling), 1);
  }

  auto const output_target_bytes =
    settings.output_bytes != 0
      ? static_cast<std::size_t>(settings.output_bytes)
      : std::clamp(b / kOutputBudgetDivisor, kMinOutputTargetBytes, kMaxOutputTargetBytes);

  auto const one_shot_fraction = settings.one_shot_max_fraction != 0
                                   ? settings.one_shot_max_fraction
                                   : derive_one_shot_fraction(budget.gpu_total_bytes);

  SIRIUS_LOG_DEBUG(
    "sort merge config: window_rows={} output_target_bytes={} fan_in={} one_shot_fraction={} "
    "(gpu_available_bytes={} gpu_total_bytes={} concurrent_tasks={} task_budget_bytes={} "
    "num_runs={} row_width={})",
    window_rows,
    output_target_bytes,
    fan_in,
    one_shot_fraction,
    g,
    budget.gpu_total_bytes,
    t,
    b,
    num_runs,
    w);

  return {.window_rows         = window_rows,
          .output_target_bytes = output_target_bytes,
          .fan_in              = fan_in,
          .one_shot_fraction   = one_shot_fraction};
}

merge_route choose_route(sort_merge_mode mode,
                         double one_shot_fraction,
                         std::size_t gpu_total_bytes,
                         const route_inputs& inputs)
{
  // The windowed path handles neither nested columns nor late-materialized ones.
  if (inputs.has_nested || inputs.has_late_mat) { return merge_route::one_shot; }
  switch (mode) {
    case sort_merge_mode::one_shot: return merge_route::one_shot;
    case sort_merge_mode::windowed: return merge_route::windowed;
    case sort_merge_mode::automatic: break;
  }
  if (inputs.any_non_gpu) { return merge_route::windowed; }
  auto const needed    = saturating_add(inputs.partition_bytes, inputs.output_bytes);
  auto const threshold = one_shot_fraction * static_cast<double>(gpu_total_bytes);
  return static_cast<double>(needed) > threshold ? merge_route::windowed : merge_route::one_shot;
}

pass_plan plan_pass(std::span<const run_summary> runs, std::size_t fan_in)
{
  pass_plan plan;
  plan.runs.resize(runs.size());
  std::iota(plan.runs.begin(), plan.runs.end(), std::size_t{0});
  // A pass that merges fewer than two runs would not reduce the run count.
  auto const f = std::max<std::size_t>(fan_in, 2);
  if (runs.size() <= f) { return plan; }

  plan.final_pass = false;
  std::stable_sort(plan.runs.begin(), plan.runs.end(), [&](std::size_t a, std::size_t b) {
    return runs[a].bytes < runs[b].bytes;
  });
  plan.runs.resize(f);
  std::sort(plan.runs.begin(), plan.runs.end());
  return plan;
}

std::size_t estimate_task_bytes(std::size_t k,
                                std::size_t gpu_runs,
                                std::size_t window_rows,
                                std::size_t row_width,
                                std::size_t key_row_width,
                                std::size_t output_target_bytes,
                                std::size_t largest_fallback_run_bytes)
{
  auto const buffered_runs = k - std::min(gpu_runs, k);
  // Window buffers (two copies each) of the runs that are not sliced in place.
  auto total =
    saturating_mul(saturating_mul(2, buffered_runs), saturating_mul(window_rows, row_width));
  // One window of gathered output rows plus its 8-byte gather-map entries.
  total = saturating_add(total, saturating_mul(window_rows, saturating_add(row_width, 8)));
  // Key columns of every run's window.
  total = saturating_add(total, saturating_mul(saturating_mul(k, window_rows), key_row_width));
  total = saturating_add(total, saturating_mul(2, output_target_bytes));
  return saturating_add(total, largest_fallback_run_bytes);
}

std::size_t estimate_reader_state_bytes(std::size_t compressed_runs,
                                        std::size_t fetch_columns,
                                        std::size_t string_columns,
                                        std::size_t window_rows)
{
  // Rows of the 1024-row chunks that cover a window starting anywhere in a chunk.
  constexpr std::size_t chunk_slack_rows = 2048;
  // Two int64 scratch values per row: the run values and run lengths of a delta column.
  constexpr std::size_t scratch_bytes_per_row = 16;
  // Offsets scratch of a str_split column.
  constexpr std::size_t string_scratch_bytes = std::size_t{2} << 20;
  auto const per_column =
    saturating_mul(scratch_bytes_per_row, saturating_add(window_rows, chunk_slack_rows));
  auto const per_run = saturating_add(saturating_mul(fetch_columns, per_column),
                                      saturating_mul(string_columns, string_scratch_bytes));
  return saturating_mul(compressed_runs, per_run);
}

}  // namespace sirius::op::merge
