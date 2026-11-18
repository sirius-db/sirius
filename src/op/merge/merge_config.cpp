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
#include "op/merge/detail/kway_merge_detail.hpp"

#include <cudf/types.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

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
/// The output chunk target is 1/kOutputBudgetDivisor of the task budget: a round is then no
/// larger than a window, which is what bounds it without the round_rows rule.
constexpr std::size_t kOutputBudgetDivisor = 16;
/// The same with the round_rows rule, where one round fills the output and so the target is
/// also the round size: fewer, larger rounds save the fixed cost of a round (about 0.56 ms).
constexpr std::size_t kRoundOutputBudgetDivisor = 8;

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

/// Whether a partition alone is small enough for the one-shot merge of the automatic mode.
bool within_one_shot_threshold(std::size_t input_bytes,
                               std::size_t output_bytes,
                               std::size_t rows,
                               double one_shot_fraction,
                               std::size_t gpu_total_bytes,
                               bool row_count_guard) noexcept
{
  // cudf::merge writes one table: a one-shot merge of this many rows fails as soon as it starts.
  if (row_count_guard && rows >= kOneShotRowLimit) { return false; }
  auto const needed    = saturating_add(input_bytes, output_bytes);
  auto const threshold = one_shot_fraction * static_cast<double>(gpu_total_bytes);
  return static_cast<double>(needed) <= threshold;
}

}  // namespace

derived_merge_config derive_merge_config(const merge_settings& settings,
                                         const device_budget& budget,
                                         std::size_t num_runs,
                                         std::size_t row_width)
{
  auto g = budget.gpu_available_bytes;
  if (settings.enabled(merge_rule::budget_floor)) {
    g = std::max(
      g,
      static_cast<std::size_t>(static_cast<double>(budget.gpu_total_bytes) * kBudgetFloorFraction));
  }
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
  // The merger and its window buffers address a window with cudf::size_type.
  window_rows = std::min<std::size_t>(
    window_rows, static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()));

  auto const output_divisor =
    settings.enabled(merge_rule::round_rows) ? kRoundOutputBudgetDivisor : kOutputBudgetDivisor;
  auto const output_target_bytes =
    settings.output_bytes != 0
      ? static_cast<std::size_t>(settings.output_bytes)
      : std::clamp(b / output_divisor, kMinOutputTargetBytes, kMaxOutputTargetBytes);

  auto const one_shot_fraction = settings.one_shot_max_fraction != 0
                                   ? settings.one_shot_max_fraction
                                   : derive_one_shot_fraction(budget.gpu_total_bytes);

  SIRIUS_LOG_DEBUG(
    "sort merge config: window_rows={} output_target_bytes={} fan_in={} one_shot_fraction={} "
    "(gpu_available_bytes={} gpu_total_bytes={} concurrent_tasks={} task_budget_bytes={} "
    "num_runs={} row_width={} gpu_free_or_reserved_bytes={} disabled_rules={})",
    window_rows,
    output_target_bytes,
    fan_in,
    one_shot_fraction,
    g,
    budget.gpu_total_bytes,
    t,
    b,
    num_runs,
    w,
    budget.gpu_available_bytes,
    settings.disabled_rules);

  return {.window_rows         = window_rows,
          .output_target_bytes = output_target_bytes,
          .fan_in              = fan_in,
          .one_shot_fraction   = one_shot_fraction};
}

std::size_t one_shot_working_bytes(const run_summary& partition) noexcept
{
  return saturating_add(saturating_mul(2, partition.bytes), saturating_mul(8, partition.rows));
}

std::size_t one_shot_peak_bytes(std::span<const run_summary> partitions,
                                std::size_t concurrent_tasks,
                                std::size_t num_devices,
                                double one_shot_fraction,
                                std::size_t gpu_total_bytes,
                                bool row_count_guard)
{
  std::size_t sort_bytes = 0;
  std::vector<std::size_t> working;
  working.reserve(partitions.size());
  for (auto const& partition : partitions) {
    sort_bytes = saturating_add(sort_bytes, partition.bytes);
    if (partition.bytes > 0 && within_one_shot_threshold(partition.bytes,
                                                         partition.bytes,
                                                         partition.rows,
                                                         one_shot_fraction,
                                                         gpu_total_bytes,
                                                         row_count_guard)) {
      working.push_back(one_shot_working_bytes(partition));
    }
  }
  // The largest merges that can be in flight together.
  auto const at_once = std::min(working.size(), std::max<std::size_t>(concurrent_tasks, 1));
  std::partial_sort(working.begin(),
                    working.begin() + static_cast<std::ptrdiff_t>(at_once),
                    working.end(),
                    std::greater<>{});
  auto peak = sort_bytes / std::max<std::size_t>(num_devices, 1);
  for (std::size_t i = 0; i < at_once; ++i) {
    peak = saturating_add(peak, working[i]);
  }
  return peak;
}

merge_route choose_route(sort_merge_mode mode,
                         double one_shot_fraction,
                         std::size_t gpu_total_bytes,
                         const route_inputs& inputs,
                         bool row_count_guard)
{
  // The windowed path handles neither nested columns nor late-materialized ones.
  if (inputs.has_nested || inputs.has_late_mat) { return merge_route::one_shot; }
  switch (mode) {
    case sort_merge_mode::one_shot: return merge_route::one_shot;
    case sort_merge_mode::windowed: return merge_route::windowed;
    case sort_merge_mode::automatic: break;
  }
  if (inputs.any_non_gpu) { return merge_route::windowed; }
  if (!within_one_shot_threshold(inputs.partition_bytes,
                                 inputs.output_bytes,
                                 inputs.partition_rows,
                                 one_shot_fraction,
                                 gpu_total_bytes,
                                 row_count_guard)) {
    return merge_route::windowed;
  }
  if (inputs.sort_partitions.empty()) { return merge_route::one_shot; }
  auto const peak = one_shot_peak_bytes(inputs.sort_partitions,
                                        inputs.concurrent_tasks,
                                        inputs.num_devices,
                                        one_shot_fraction,
                                        gpu_total_bytes,
                                        row_count_guard);
  return peak > gpu_total_bytes ? merge_route::windowed : merge_route::one_shot;
}

pass_plan plan_pass(std::span<const run_summary> runs, std::size_t fan_in, bool short_first_merge)
{
  pass_plan plan;
  plan.runs.resize(runs.size());
  std::iota(plan.runs.begin(), plan.runs.end(), std::size_t{0});
  // A pass that merges fewer than two runs would not reduce the run count.
  auto const f = std::max<std::size_t>(fan_in, 2);
  if (runs.size() <= f) { return plan; }

  plan.final_pass = false;
  // The k-ary Huffman rule: a merge of f runs lowers the run count by f - 1, so only a count
  // with (R - 1) mod (f - 1) == 0 ends in a final pass of exactly f runs. Any other count is
  // brought there by one short merge of the smallest runs, now, instead of by a short final pass
  // behind a merge that rewrote nearly every row.
  auto take = f;
  if (short_first_merge && (runs.size() - 1) % (f - 1) != 0) {
    take = (runs.size() - 2) % (f - 1) + 2;
  }
  std::stable_sort(plan.runs.begin(), plan.runs.end(), [&](std::size_t a, std::size_t b) {
    return runs[a].bytes < runs[b].bytes;
  });
  plan.runs.resize(take);
  std::sort(plan.runs.begin(), plan.runs.end());
  return plan;
}

std::size_t round_rows_for(std::size_t window_rows,
                           std::size_t output_target_bytes,
                           std::size_t row_width)
{
  auto const w    = std::max<std::size_t>(row_width, 1);
  auto const rows = output_target_bytes / w + (output_target_bytes % w != 0 ? 1 : 0);
  return std::min<std::size_t>(std::max(window_rows, rows), kMaxRoundRows);
}

std::size_t estimate_window_slot_bytes(std::size_t window_rows,
                                       std::size_t row_width,
                                       std::size_t fetch_columns,
                                       std::size_t string_columns,
                                       std::size_t string_row_width)
{
  // A validity mask per column: cudf pads the mask to a multiple of 64 bytes.
  auto const masks = saturating_mul(fetch_columns, window_rows / 8 + 64);
  if (string_columns == 0) { return saturating_add(saturating_mul(window_rows, row_width), masks); }

  if (string_row_width == 0 || string_row_width > row_width) {
    // The chars share of row_width is not known: the 25% slack is taken on the whole row, and
    // 8-byte offsets are counted on top of the row.
    auto const window_bytes = saturating_mul(window_rows, row_width);
    auto const per_string =
      saturating_add(saturating_mul(saturating_add(window_rows, 1), 8),    // offsets
                     saturating_add(saturating_mul(window_rows, 2), 64));  // the + 1 and minimum
    auto slot = saturating_add(window_bytes, masks);
    slot      = saturating_add(slot, window_bytes / 4);
    return saturating_add(slot, saturating_mul(string_columns, per_string));
  }

  // What window_buffer's constructor allocates. The merger asks for each STRING column's chars
  // at its bytes per row plus one; summed over the columns that is string_row_width plus one
  // byte per column. window_buffer adds a quarter and allocates at least 64 bytes per column.
  auto const fixed = saturating_mul(window_rows, row_width - string_row_width);
  auto const chars_exact =
    saturating_mul(window_rows, saturating_add(string_row_width, string_columns));
  auto const chars = saturating_add(saturating_add(chars_exact, chars_exact / 4),
                                    saturating_mul(string_columns, 64));
  // int32 offsets unless a column's chars could pass 2 GiB; no column's can while their sum
  // does not.
  auto const offset_width =
    chars_exact > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ? 8U : 4U;
  auto const offsets =
    saturating_mul(string_columns, saturating_mul(saturating_add(window_rows, 1), offset_width));
  return saturating_add(saturating_add(fixed, masks), saturating_add(chars, offsets));
}

std::size_t estimate_task_bytes(std::size_t k,
                                std::size_t gpu_runs,
                                std::size_t window_rows,
                                std::size_t row_width,
                                std::size_t /*key_row_width*/,
                                std::size_t output_target_bytes,
                                std::size_t fallback_runs_bytes,
                                std::size_t fetch_columns,
                                std::size_t string_columns,
                                std::size_t round_rows,
                                std::size_t string_row_width)
{
  auto const buffered_runs = k - std::min(gpu_runs, k);
  auto const round         = round_rows == 0 ? window_rows : round_rows;
  auto const round_bytes   = saturating_mul(round, row_width);

  // Two slots for each run that is not sliced in place.
  auto const slot = estimate_window_slot_bytes(
    window_rows, row_width, fetch_columns, string_columns, string_row_width);
  auto total = saturating_mul(saturating_mul(2, buffered_runs), slot);

  // The round outputs, up to the target plus one round, and their concatenation into one table.
  total =
    saturating_add(total, saturating_mul(2, saturating_add(output_target_bytes, round_bytes)));

  // Scratch of one bounded merge: 8-byte gather-map entries per output row and, per tile boundary,
  // its rank plus three 4-byte values per run (the co-rank and corank_split's two scratch values).
  auto const boundaries =
    saturating_add(round / static_cast<std::size_t>(detail::kDefaultTileRows), 2);
  total = saturating_add(total, saturating_mul(round, 8));
  total =
    saturating_add(total, saturating_mul(boundaries, saturating_add(4, saturating_mul(12, k))));

  // Every fallback run can hold one converted batch at once.
  return saturating_add(total, fallback_runs_bytes);
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
  // Offsets scratch of a str_split column: its decode cursor's int64 offsets chunks per window
  // (see str_split_chunks.cu); keep the two in step.
  constexpr std::size_t string_scratch_bytes = std::size_t{2} << 20;
  auto const per_column =
    saturating_mul(scratch_bytes_per_row, saturating_add(window_rows, chunk_slack_rows));
  auto const per_run = saturating_add(saturating_mul(fetch_columns, per_column),
                                      saturating_mul(string_columns, string_scratch_bytes));
  return saturating_mul(compressed_runs, per_run);
}

}  // namespace sirius::op::merge
