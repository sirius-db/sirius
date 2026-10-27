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

#pragma once

#include <cstddef>
#include <vector>

namespace sirius::preload {

/**
 * @brief Configuration of the preload executor (`sirius.executor.preload`).
 *
 * The preload executor moves batches that the downgrade executor spilled back toward the GPU
 * (HOST->GPU, DISK->GPU) or toward HOST (DISK->HOST) before their consumer asks for them. It is
 * advisory: a refused or failed promotion costs performance only.
 *
 * Fractions are of the target memory space's capacity unless stated otherwise. The GPU and HOST
 * ceilings are derived from each space's @c downgrade_stop_fraction minus a headroom, so preload
 * on its own never drives a space to its downgrade trigger. A ceiling at or below zero disables
 * that class of promotion on that space.
 *
 * The defaults leave the executor disabled, so behaviour is unchanged unless @c enabled is set.
 */
struct preload_executor_config {
  /// Create and run the preload executor. When false, no executor, thread or subscription exists.
  bool enabled{false};

  /// Concurrent HOST->GPU copies per GPU. 0 = auto: 5 when the CPU and GPU share a C2C link,
  /// otherwise 1. Valid range [0, 64].
  int host_to_gpu_concurrency{0};

  /// Concurrent copies per DISK space. 0 = auto: 5 when GPUDirect Storage is in use, otherwise 1.
  /// Valid range [0, 64].
  int disk_concurrency{0};

  /// Cores to pin preload threads to (`cpu_affinity` in YAML). Empty = no pinning.
  std::vector<int> cpu_affinity_list{};

  /// Committed promotions (inputs of queued tasks) may fill a GPU up to
  /// `downgrade_stop_fraction - gpu_committed_headroom_fraction`.
  double gpu_committed_headroom_fraction{0.0};

  /// Speculative promotions (repository batches without a queued consumer) may fill a GPU up to
  /// `downgrade_stop_fraction - gpu_speculative_headroom_fraction`. Raised to
  /// @c gpu_committed_headroom_fraction when it is smaller.
  double gpu_speculative_headroom_fraction{0.10};

  /// DISK->HOST staging may fill a HOST space up to
  /// `downgrade_stop_fraction - host_headroom_fraction`.
  double host_headroom_fraction{0.10};

  /// Maximum preloaded-but-not-yet-consumed bytes per GPU.
  double gpu_lookahead_fraction{0.20};

  /// Share of a GPU's capacity that speculative promotions may hold out of
  /// @c gpu_lookahead_fraction. Lowered to @c gpu_lookahead_fraction when it is larger.
  double speculative_fraction{0.05};

  /// Never promote a batch to a GPU whose estimated need exceeds this fraction of its capacity.
  /// Lowered to @c max_inflight_fraction when it is larger.
  double max_batch_fraction{0.10};

  /// Maximum bytes being copied into one GPU at once.
  double max_inflight_fraction{0.10};

  /// Queued tasks at the front of a device bucket whose inputs are left alone, because the
  /// scheduler is about to run them. -1 = auto: the pipeline executor's thread count. The
  /// no-preference bucket skips this many tasks per GPU.
  int head_margin_tasks{-1};

  /// Queued tasks per bucket, past the head margin, whose inputs are considered.
  int lookahead_tasks{8};

  /// Front-of-FIFO positions considered per (repository, partition).
  int repository_window{2};

  /// For consumers that drain partitions in order: how many partitions past the next one may be
  /// promoted speculatively.
  int repository_partition_lookahead{1};

  /// Stage DISK->HOST when the GPU target is unknown, out of room, or banned. When false, preload
  /// never targets a HOST space.
  bool stage_to_host{true};

  /// Maximum promotions of one batch per query.
  int max_promotions_per_batch{2};

  /// Attempts to take a batch lock before the candidate is dropped. Must be at least 1.
  int max_lock_attempts{3};

  /// GPU out-of-memory retries per candidate, each with a doubled estimate, before the candidate
  /// is banned from GPUs.
  int max_oom_retries{1};

  /// Maximum tracked candidates. At the limit, the newest query's most recently downgraded
  /// candidate is evicted first. Must be positive.
  std::size_t max_candidates{65536};
};

/// A ceiling, as a fraction of capacity, at or below this counts as zero: it absorbs rounding in
/// differences and products of configured fractions.
inline constexpr double ceiling_epsilon = 1e-9;

/**
 * @brief A preload ceiling as a fraction of a memory space's capacity.
 *
 * @param downgrade_stop_fraction The space's downgrade stop threshold, as a fraction of capacity.
 * @param headroom_fraction The headroom preload keeps below it.
 * @return `downgrade_stop_fraction - headroom_fraction`, or 0 when that is at or below
 *         @ref ceiling_epsilon. A ceiling of 0 disables that class of promotion on the space.
 */
[[nodiscard]] constexpr double ceiling_fraction(double downgrade_stop_fraction,
                                                double headroom_fraction) noexcept
{
  auto const ceiling = downgrade_stop_fraction - headroom_fraction;
  return ceiling > ceiling_epsilon ? ceiling : 0.0;
}

/// The two GPU ceilings of one GPU, as fractions of its capacity. 0 disables that class.
struct gpu_ceiling_fractions {
  double committed;
  double speculative;
};

/**
 * @brief Both GPU ceilings of one GPU.
 *
 * Config validation and the executor both derive the ceilings here, so a class that the config
 * reports as disabled is also disabled at run time.
 *
 * @param cfg The preload config, after `sirius_config` has clamped it.
 * @param downgrade_stop_fraction The GPU's downgrade stop threshold, as a fraction of capacity.
 */
[[nodiscard]] constexpr gpu_ceiling_fractions gpu_ceilings(preload_executor_config const& cfg,
                                                           double downgrade_stop_fraction) noexcept
{
  return {ceiling_fraction(downgrade_stop_fraction, cfg.gpu_committed_headroom_fraction),
          ceiling_fraction(downgrade_stop_fraction, cfg.gpu_speculative_headroom_fraction)};
}

}  // namespace sirius::preload
