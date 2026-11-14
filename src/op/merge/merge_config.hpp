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

#pragma once

#include "sirius_config.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sirius::op::merge {

/**
 * @brief The sizing and planning rules of the merge that can be switched off one by one.
 *
 * Each rule is a bit of the sort_merge_disabled_rules setting (config::sort_merge_rule holds the
 * values); a set bit restores the behaviour the merge had before the rule. They exist so that
 * the effect of each rule can be measured alone.
 */
enum class merge_rule : std::uint64_t {
  /// plan_pass makes the first merge of a partition short, so every later merge and the final
  /// pass read fan_in runs (the k-ary Huffman plan).
  huffman_plan = config::sort_merge_rule::HUFFMAN_PLAN,
  /// derive_merge_config sizes from at least kBudgetFloorFraction (a quarter) of the device,
  /// not only from the bytes free when the partition is armed.
  budget_floor = config::sort_merge_rule::BUDGET_FLOOR,
  /// The budget is divided by the partitions that can be open (at most the executor width), and
  /// the open-partition cap is that same number.
  partition_budget = config::sort_merge_rule::PARTITION_BUDGET,
  /// A task reserves window buffers only for the runs it cannot slice in place, plus one spare.
  in_place_reservation = config::sort_merge_rule::IN_PLACE_RESERVATION,
  /// A round emits output_target_bytes worth of rows, whatever the window of a buffered run is,
  /// and the derived output target is B / 8 instead of B / 16.
  round_rows = config::sort_merge_rule::ROUND_ROWS,
  /// In automatic mode a partition of 2^31 - 1 rows or more is windowed.
  row_count_guard = config::sort_merge_rule::ROW_COUNT_GUARD,
  /// The cold-start estimate of a one-shot merge is 2.5 x its input instead of 2 x. Off by
  /// default: it removes the first task's OOM retry, but an 8-partition one-shot sort of twice
  /// the GPU limit ran 39% longer with it (its sink drained 19 s after the last merge).
  one_shot_estimate = config::sort_merge_rule::ONE_SHOT_ESTIMATE,
  /// In automatic mode a partition with one GPU-resident run is forwarded without a merge,
  /// also when its size would send it to the windowed route.
  single_run_passthrough = config::sort_merge_rule::SINGLE_RUN_PASSTHROUGH,
  /// The STRING part of a window buffer slot is estimated from the STRING columns' own width.
  string_slot_estimate = config::sort_merge_rule::STRING_SLOT_ESTIMATE,
};

/**
 * @brief The sort_merge_* settings as the user set them (0 = derive).
 */
struct merge_settings {
  sort_merge_mode mode{sort_merge_mode::automatic};
  std::uint64_t window_rows{0};     ///< n; 0 = derive per device
  std::uint64_t output_bytes{0};    ///< output chunk per task; 0 = derive
  double one_shot_max_fraction{0};  ///< auto threshold; 0 = derive from device size
  std::uint64_t max_fan_in{config::DEFAULT_SORT_MERGE_MAX_FAN_IN};
  /// Bit set of the merge_rule values that are switched off.
  std::uint64_t disabled_rules{config::DEFAULT_SORT_MERGE_DISABLED_RULES};

  /// Whether @p rule is in force.
  [[nodiscard]] bool enabled(merge_rule rule) const noexcept
  {
    return (disabled_rules & static_cast<std::uint64_t>(rule)) == 0;
  }

  /// Snapshot the sort_merge_* fields of @p params.
  static merge_settings from(const operator_params& params) noexcept
  {
    return {.mode                  = params.sort_merge_mode_setting,
            .window_rows           = params.sort_merge_window_rows,
            .output_bytes          = params.sort_merge_output_bytes,
            .one_shot_max_fraction = params.sort_merge_one_shot_max_fraction,
            .max_fan_in            = params.sort_merge_max_fan_in,
            .disabled_rules        = params.sort_merge_disabled_rules};
  }
};

/**
 * @brief GPU memory of the device a merge task runs on.
 */
struct device_budget {
  std::size_t gpu_available_bytes{0};  ///< GPU memory free or reserved by tasks in flight
  std::size_t gpu_total_bytes{0};      ///< total GPU memory of the device (its configured limit)
  /// Windowed partitions that can be open at once on the device: the executor width, lowered to
  /// the number of partitions the operator has when the partition_budget rule is in force.
  std::size_t concurrent_tasks{1};
};

/**
 * @brief The sizes a merge runs with, after derivation.
 */
struct derived_merge_config {
  std::size_t window_rows;          ///< n: maximum rows per input window
  std::size_t output_target_bytes;  ///< output chunk target per task
  std::size_t fan_in;               ///< most runs one task reads
  double one_shot_fraction;         ///< auto threshold as a fraction of total GPU memory
};

inline constexpr std::size_t kWindowRowsFloor       = 65536;  // derived n floor (yields to memory)
inline constexpr std::size_t kMinExplicitWindowRows = 1024;   // smallest SET value accepted
inline constexpr std::size_t kShrinkWindowRowsFloor = 1024;   // shrink_for_retry floor

/// Share of the device's GPU memory the budget is derived from at least (the budget_floor rule).
/// The runs of the merge are spilled first (input_residency_policy), so the memory they hold
/// when a partition is armed is not lost to the merge. Without a floor a partition armed on a
/// full device is sized from whatever a downgrade happened to have freed at that instant: the
/// same query then runs with windows several times smaller or larger from one execution to
/// the next.
///
/// A quarter, measured on sorts of twice the GPU limit: it fixed the sizes in every run, cut
/// the wall time by about a fifth where the host tier holds the spilled runs, and changed
/// nothing beyond noise where they go to disk. A half, the first choice, sized windows the
/// device has no room for under pressure: more runs were evicted to make room for them, and
/// with the disk tier in use the sort got 7% to 24% slower.
inline constexpr double kBudgetFloorFraction = 0.25;

/// Most rows a round sized from the output emits. A task whose round stops short of the output
/// target (a truncated window) runs another one, and the rounds of a task are concatenated
/// into one table: with rounds of at most half the cudf row limit, a task that reaches its
/// target in two rounds cannot pass that limit.
inline constexpr std::size_t kMaxRoundRows = ((std::size_t{1} << 31) - 1) / 2;

/// Rows a one-shot merge can emit: cudf::merge writes one table. A partition with this many rows
/// or more is windowed in automatic mode (the row_count_guard rule).
inline constexpr std::size_t kOneShotRowLimit = (std::size_t{1} << 31) - 1;

/**
 * @brief Resolve the settings against a device: derive every value left at 0.
 *
 * window_rows and output_target_bytes derive from the device's available GPU memory (at least
 * kBudgetFloorFraction of its total with the budget_floor rule) shared among its concurrent
 * tasks, one_shot_fraction from its total GPU memory. The derived output target is 1/8 of the
 * task budget with the round_rows rule and 1/16 without it, within [64 MiB, 1 GiB]. fan_in is
 * num_runs capped by max_fan_in, by kMaxKwayMergeInputs and by the floor-sized windows that fit the
 * task budget (never below 2 unless num_runs is). An explicit window_rows below
 * kMinExplicitWindowRows is raised to it, and window_rows, explicit or derived, never exceeds the
 * cudf row limit (2^31 - 1).
 *
 * @param settings The user's sort_merge_* settings.
 * @param budget GPU memory of the device and the number of concurrent merge tasks on it.
 * @param num_runs Runs in the partition to merge.
 * @param row_width Estimated row width in bytes of the fetched columns.
 */
derived_merge_config derive_merge_config(const merge_settings& settings,
                                         const device_budget& budget,
                                         std::size_t num_runs,
                                         std::size_t row_width);

/// The merge path chosen for one partition.
enum class merge_route : uint8_t { one_shot, windowed };

/**
 * @brief What choose_route needs to know about a partition.
 */
struct route_inputs {
  std::size_t partition_bytes{0};  ///< bytes of the partition's runs
  std::size_t output_bytes{0};     ///< bytes of the merged output
  std::size_t partition_rows{0};   ///< rows of the partition's runs (0 = not known)
  bool any_non_gpu{false};         ///< some input batch is not on the GPU tier
  bool has_nested{false};          ///< some fetched column has a nested type
  bool has_late_mat{false};        ///< the sort carries late-materialized columns
};

/**
 * @brief Choose the merge path for one partition.
 *
 * Nested or late-materialized columns always take the one-shot path, whatever @p mode says. In
 * automatic mode a partition with a batch off the GPU is windowed, and so is one with
 * kOneShotRowLimit rows or more when @p row_count_guard is set (a one-shot merge of it fails at
 * once); otherwise it is windowed when its input plus output bytes exceed one_shot_fraction of
 * gpu_total_bytes.
 *
 * @param mode The sort_merge_mode setting.
 * @param one_shot_fraction The derived auto threshold (derived_merge_config::one_shot_fraction).
 * @param gpu_total_bytes Total GPU memory of the device.
 * @param inputs Size and shape of the partition.
 * @param row_count_guard Whether the row_count_guard rule is in force.
 */
merge_route choose_route(sort_merge_mode mode,
                         double one_shot_fraction,
                         std::size_t gpu_total_bytes,
                         const route_inputs& inputs,
                         bool row_count_guard = true);

/**
 * @brief Size of one run, for pass planning.
 */
struct run_summary {
  std::size_t bytes{0};
  std::size_t rows{0};
};

/**
 * @brief The runs one merge task of a pass reads.
 */
struct pass_plan {
  bool final_pass{true};          ///< false when the task writes an intermediate run
  std::vector<std::size_t> runs;  ///< indices into the input span
};

/**
 * @brief Plan the next merge task of a partition under the fan-in cap.
 *
 * With at most fan_in runs the task merges all of them and is the final pass; otherwise it
 * merges the smallest runs by bytes (ties to the lower index, indices returned ascending) into an
 * intermediate run. A fan_in below 2 is treated as 2.
 *
 * How many runs an intermediate merge takes, with R runs and f = fan_in: when @p short_first_merge
 * is set and (R - 1) mod (f - 1) != 0 it takes ((R - 2) mod (f - 1)) + 2 runs (between 2 and
 * f - 1), otherwise f. This is the k-ary Huffman rule, which writes the fewest rows for any run
 * sizes: after the one short merge R' = R - take + 1 satisfies (R' - 1) mod (f - 1) == 0, so
 * every later call takes f runs and the final pass reads exactly f. Without it the shortfall
 * lands in the final pass, after a merge that rewrote nearly every row. The function keeps no
 * state: the remainder is evaluated on the runs it is given, once per pass.
 *
 * @param runs The partition's current runs.
 * @param fan_in Most runs one task reads.
 * @param short_first_merge Whether the huffman_plan rule is in force.
 */
pass_plan plan_pass(std::span<const run_summary> runs,
                    std::size_t fan_in,
                    bool short_first_merge = true);

/**
 * @brief Rows one round of a windowed task emits at most, when it is sized from the output.
 *
 * max(window_rows, ceil(output_target_bytes / row_width)), at most kMaxRoundRows: one round
 * fills the task's output when no truncated window stops it earlier. When one does, the task
 * runs another round, so its output is below output_target_bytes plus one round. Runs that are
 * sliced in place offer the round this many rows; a run read through a window buffer still
 * offers at most window_rows.
 *
 * @param window_rows n: maximum rows per window of a buffered run.
 * @param output_target_bytes Output chunk target of the task.
 * @param row_width Estimated row width in bytes of the fetched columns.
 */
std::size_t round_rows_for(std::size_t window_rows,
                           std::size_t output_target_bytes,
                           std::size_t row_width);

/**
 * @brief Estimate the device bytes of ONE slot of a window buffer (it allocates two).
 *
 * Follows detail::window_buffer's constructor: the rows of the fixed-width columns, a validity
 * mask per column and, per STRING column, its offsets and its chars at 1.25 x the estimate.
 *
 * With @p string_row_width (the string_slot_estimate rule) the chars are sized like the merger
 * sizes them for a batch without column metadata: each STRING column's share of the row width
 * plus one byte, times 1.25, with 4-byte offsets unless the chars could pass 2 GiB. The row
 * width of a STRING column includes its 4-byte offsets, so that share is an upper bound of its
 * chars. With string_row_width == 0 the chars share is unknown: the 25% slack is taken on the
 * whole row and the offsets are counted at 8 bytes on top of it.
 *
 * @param window_rows Row capacity of the slot.
 * @param row_width Estimated row width in bytes of the fetched columns.
 * @param fetch_columns Columns in the buffer.
 * @param string_columns How many of them are STRING.
 * @param string_row_width Sum of the estimated widths of the STRING columns; 0 = not known.
 */
std::size_t estimate_window_slot_bytes(std::size_t window_rows,
                                       std::size_t row_width,
                                       std::size_t fetch_columns,
                                       std::size_t string_columns,
                                       std::size_t string_row_width = 0);

/**
 * @brief Estimate the GPU bytes one windowed merge task needs.
 *
 * The sum of:
 * - the window buffers of the runs that are not sliced in place: two slots per run
 *   (estimate_window_slot_bytes);
 * - the output: round outputs accumulate until they reach output_target_bytes, so up to the
 *   target plus one round of round_rows rows, and are then concatenated into one table of the
 *   same size;
 * - the scratch of one bounded merge of round_rows rows: its gather map and the co-ranks at the
 *   tile boundaries;
 * - the batches converted whole to the GPU tier, which stay resident until the task ends.
 *
 * The merge compares keys through views of the windows, so the key columns cost nothing extra.
 *
 * @param k Runs the task reads.
 * @param gpu_runs How many of them are GPU-resident (sliced, no window buffer).
 * @param window_rows n: maximum rows per input window.
 * @param row_width Estimated row width in bytes of the fetched columns.
 * @param key_row_width Unused: no key column is copied. Kept until the callers stop passing it.
 * @param output_target_bytes Output chunk target of the task.
 * @param fallback_runs_bytes Sum, over the runs that may be converted whole to the GPU tier (a
 *        batch without a row-range reader), of each such run's largest batch. A task can hold one
 *        converted batch per fallback run at once, so this is a sum, not the largest of them.
 * @param fetch_columns Columns the task fetches (each has a validity mask per buffer slot).
 * @param string_columns How many of them are STRING (chars slack and offsets per buffer slot).
 * @param round_rows Rows one round emits at most; 0 = window_rows.
 * @param string_row_width Sum of the estimated widths of the STRING columns; 0 = not known (see
 *        estimate_window_slot_bytes).
 */
std::size_t estimate_task_bytes(std::size_t k,
                                std::size_t gpu_runs,
                                std::size_t window_rows,
                                std::size_t row_width,
                                std::size_t key_row_width,
                                std::size_t output_target_bytes,
                                std::size_t fallback_runs_bytes,
                                std::size_t fetch_columns    = 0,
                                std::size_t string_columns   = 0,
                                std::size_t round_rows       = 0,
                                std::size_t string_row_width = 0);

/**
 * @brief Estimate the GPU bytes the row-range readers of compressed runs keep between refills.
 *
 * A compressed run is read through one decode cursor per fetched column. A cursor keeps grow-only
 * scratch sized by the chunks that cover one window: up to two int64 values per row (delta with
 * run lengths), and a fixed scratch for a STRING column's offsets. Dictionary keys, which a
 * cursor also keeps, depend on the data and are not estimated.
 *
 * @param compressed_runs Runs of the task that hold a compressed batch.
 * @param fetch_columns Columns the task fetches.
 * @param string_columns How many of them are STRING.
 * @param window_rows n: maximum rows per input window.
 */
std::size_t estimate_reader_state_bytes(std::size_t compressed_runs,
                                        std::size_t fetch_columns,
                                        std::size_t string_columns,
                                        std::size_t window_rows);

}  // namespace sirius::op::merge
