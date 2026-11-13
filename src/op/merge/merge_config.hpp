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
 * @brief The sort_merge_* settings as the user set them (0 = derive).
 */
struct merge_settings {
  sort_merge_mode mode{sort_merge_mode::automatic};
  std::uint64_t window_rows{0};     ///< n; 0 = derive per device
  std::uint64_t output_bytes{0};    ///< output chunk per task; 0 = derive
  double one_shot_max_fraction{0};  ///< auto threshold; 0 = derive from device size
  std::uint64_t max_fan_in{config::DEFAULT_SORT_MERGE_MAX_FAN_IN};

  /// Snapshot the sort_merge_* fields of @p params.
  static merge_settings from(const operator_params& params) noexcept
  {
    return {.mode                  = params.sort_merge_mode_setting,
            .window_rows           = params.sort_merge_window_rows,
            .output_bytes          = params.sort_merge_output_bytes,
            .one_shot_max_fraction = params.sort_merge_one_shot_max_fraction,
            .max_fan_in            = params.sort_merge_max_fan_in};
  }
};

/**
 * @brief GPU memory of the device a merge task runs on.
 */
struct device_budget {
  std::size_t gpu_available_bytes{0};  ///< GPU memory free or reserved by tasks in flight
  std::size_t gpu_total_bytes{0};      ///< total GPU memory of the device
  std::size_t concurrent_tasks{1};     ///< sort-merge tasks that can run at once on the device
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

/**
 * @brief Resolve the settings against a device: derive every value left at 0.
 *
 * window_rows and output_target_bytes derive from the device's available GPU memory shared among
 * its concurrent tasks, one_shot_fraction from its total GPU memory. fan_in is num_runs capped by
 * max_fan_in, by kMaxKwayMergeInputs and by the floor-sized windows that fit the task budget (never
 * below 2 unless num_runs is). An explicit window_rows below kMinExplicitWindowRows is raised to
 * it, and window_rows, explicit or derived, never exceeds the cudf row limit (2^31 - 1).
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
  bool any_non_gpu{false};         ///< some input batch is not on the GPU tier
  bool has_nested{false};          ///< some fetched column has a nested type
  bool has_late_mat{false};        ///< the sort carries late-materialized columns
};

/**
 * @brief Choose the merge path for one partition.
 *
 * Nested or late-materialized columns always take the one-shot path, whatever @p mode says. In
 * automatic mode a partition with a batch off the GPU is windowed; otherwise it is windowed when
 * its input plus output bytes exceed one_shot_fraction of gpu_total_bytes.
 *
 * @param mode The sort_merge_mode setting.
 * @param one_shot_fraction The derived auto threshold (derived_merge_config::one_shot_fraction).
 * @param gpu_total_bytes Total GPU memory of the device.
 * @param inputs Size and shape of the partition.
 */
merge_route choose_route(sort_merge_mode mode,
                         double one_shot_fraction,
                         std::size_t gpu_total_bytes,
                         const route_inputs& inputs);

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
 * merges the fan_in smallest runs by bytes (ties to the lower index, indices returned ascending)
 * into an intermediate run. A fan_in below 2 is treated as 2.
 *
 * @param runs The partition's current runs.
 * @param fan_in Most runs one task reads.
 */
pass_plan plan_pass(std::span<const run_summary> runs, std::size_t fan_in);

/**
 * @brief Estimate the GPU bytes one windowed merge task needs.
 *
 * The sum of:
 * - the window buffers of the runs that are not sliced in place: two slots per run, each of
 *   window_rows rows, with the slack, offsets and validity masks window_buffer allocates;
 * - the output: round outputs accumulate until they reach output_target_bytes, so up to the
 *   target plus one round, and are then concatenated into one table of the same size;
 * - the scratch of one bounded merge: its gather map and the co-ranks at the tile boundaries;
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
 */
std::size_t estimate_task_bytes(std::size_t k,
                                std::size_t gpu_runs,
                                std::size_t window_rows,
                                std::size_t row_width,
                                std::size_t key_row_width,
                                std::size_t output_target_bytes,
                                std::size_t fallback_runs_bytes,
                                std::size_t fetch_columns  = 0,
                                std::size_t string_columns = 0);

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
