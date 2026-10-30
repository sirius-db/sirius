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

/**
 * @file preload_policy.hpp
 * @brief Pure decision functions of the preload executor: plan order, target choice, ceilings,
 *        budgets, head-of-line assignment, lane caps and the shared job/outcome types.
 *
 * Nothing here touches a memory space, a batch, a queue or a CUDA API. Callers read live state
 * (consumed bytes, trigger state, board counts, epochs) and pass plain numbers in, so every rule
 * can be tested without a GPU.
 */

#include "exec/queue_priority.hpp"
#include "memory/memory_wait_board.hpp"
#include "preload/config.hpp"
#include "query_id.hpp"

#include <cucascade/memory/common.hpp>

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace cucascade {
class data_batch;
}  // namespace cucascade

namespace sirius::preload {

// ---------------------------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------------------------

/// Promotion class. A: an input of a queued task (committed). B: a repository batch with no
/// queued consumer yet (speculative). A sorts before B at equal priority.
enum class klass : std::uint8_t { A = 0, B = 1 };

/// Plan band of an item that targets a GPU.
inline constexpr std::uint8_t band_gpu = 0;
/// Plan band of an item that can only be staged DISK->HOST.
inline constexpr std::uint8_t band_host_staging = 1;

/**
 * @brief Sort key of a planned item; smaller sorts first.
 *
 * Bands keep GPU targets ahead of staging. Within a band, the scheduler's own priority decides,
 * so a repository batch of an older query goes before a newer query's queued tasks, as the
 * scheduler would run them.
 */
struct plan_key {
  std::uint8_t band{band_gpu};       ///< band_gpu or band_host_staging
  exec::queue_priority priority{0};  ///< A: the task's priority; B: the destination pipeline's
  klass k{klass::A};                 ///< A before B at equal priority
  std::uint32_t partition{0};        ///< B only; 0 for A
  std::uint32_t position{0};         ///< A: position in the bucket's pop order; B: FIFO index
  std::uint64_t batch_id{0};         ///< deterministic tiebreak

  auto operator<=>(plan_key const&) const = default;
};

/// A promotion target: a GPU or HOST space, by its index in the executor's space lists.
struct target {
  cucascade::memory::Tier tier{cucascade::memory::Tier::SIZE};
  std::size_t index{0};

  bool operator==(target const&) const = default;
};

/// What one promotion attempt ended with.
enum class promotion_outcome : std::uint8_t {
  promoted,          ///< the batch is now in the target space
  retired,           ///< the query no longer accepts work
  expired,           ///< the batch is gone
  contended,         ///< the batch was locked or busy
  stale,             ///< the batch is not where the job expected it, or a task claimed it
  already_resident,  ///< the batch is already on a GPU (or, for staging, off DISK)
  unsupported,       ///< no conversion exists for this representation and target
  too_big,           ///< the exact need exceeds the per-batch limit
  no_memory,         ///< the target space refused the reservation
  gated,             ///< the reservation would take the space above the preload ceiling
  held_off,          ///< a task on the GPU is stalled on memory or has an OOM retry pending
  oom,               ///< a GPU copy overran its reservation (the estimate was low)
  failed             ///< any other error
};

/// Which copy a successful promotion made.
enum class promotion_kind : std::uint8_t {
  none,
  h2g,          ///< HOST -> GPU
  d2g,          ///< DISK -> GPU
  d2h,          ///< DISK -> HOST
  relocate_d2h  ///< DISK -> HOST of an encoded representation, through relocate
};

/// Result of one promotion attempt.
struct promotion_result {
  promotion_outcome outcome{promotion_outcome::failed};
  /// For @c stale: where the batch actually is.
  cucascade::memory::memory_space_id observed{cucascade::memory::Tier::SIZE, -1};
  /// Bytes reserved for the copy (the exact need, with slack), when promoted.
  std::size_t bytes{0};
  /// Which copy was made, when promoted. Set it explicitly: when it is left at @c none the store
  /// infers it from the tiers, which cannot tell @c relocate_d2h from @c d2h.
  promotion_kind kind{promotion_kind::none};
  /// When promoted: a value of @c g_conversion_seq taken while the batch was exclusively locked
  /// for the copy, so every downgrade record of an earlier move sorts below it and every later
  /// one above it. 0 when none was taken; the store then falls back to the candidate's sequence
  /// at take time.
  std::uint64_t conversion_seq{0};
};

/// One promotion handed to a lane. Built under the executor's lock; holds no strong reference.
struct promotion_job {
  std::weak_ptr<cucascade::data_batch> batch;
  std::uint64_t batch_id{0};
  query_id_t query{};
  /// Where the candidate was last seen; a mismatch under the batch lock gives @c stale.
  cucascade::memory::memory_space_id expected_source{cucascade::memory::Tier::SIZE, -1};
  target t{};
  klass k{klass::A};
  /// Bytes to reserve before taking the batch lock.
  std::size_t est_reserve{0};
  /// The preload ceiling of the target, in consumed bytes.
  std::size_t ceiling{0};
  /// Memory epoch of the target when the job was taken; a memory refusal gates with this value.
  std::uint64_t mem_epoch_at_take{0};
  /// Queue epoch when the job was taken; a lock refusal blocks the candidate until it moves.
  std::uint64_t queue_epoch_at_take{0};
  /// The candidate's newest conversion sequence when the job was taken; a newer one at outcome
  /// time means the batch was downgraded again during the attempt.
  std::uint64_t seq_at_take{0};
  /// Generation stamp of this job's device preloading budget charge (GPU targets only; 0 when
  /// none). The ledger entry records it so a later release acts on the right generation.
  std::uint64_t job_seq{0};
};

// ---------------------------------------------------------------------------------------------
// Size arithmetic
// ---------------------------------------------------------------------------------------------

[[nodiscard]] constexpr std::size_t sat_add(std::size_t a, std::size_t b) noexcept
{
  return a > std::numeric_limits<std::size_t>::max() - b ? std::numeric_limits<std::size_t>::max()
                                                         : a + b;
}

[[nodiscard]] constexpr std::size_t sat_sub(std::size_t a, std::size_t b) noexcept
{
  return a > b ? a - b : 0;
}

[[nodiscard]] constexpr std::size_t sat_mul(std::size_t a, std::size_t b) noexcept
{
  if (a == 0 || b == 0) { return 0; }
  return a > std::numeric_limits<std::size_t>::max() / b ? std::numeric_limits<std::size_t>::max()
                                                         : a * b;
}

/// Slack added to every estimate: the larger of 1 MiB and 1/64 of it.
[[nodiscard]] constexpr std::size_t with_slack(std::size_t bytes) noexcept
{
  constexpr std::size_t min_slack = std::size_t{1} << 20;
  return sat_add(bytes, std::max(min_slack, bytes / 64));
}

/**
 * @brief Bytes to reserve before locking a candidate.
 *
 * @param est_bytes The candidate's size estimate (its largest recorded size).
 * @return `with_slack(max(est_bytes, 1))`. There is no out-of-memory retry: the first GPU OOM
 *         bans the candidate (@ref promotion_outcome::oom), so the estimate is never inflated.
 */
[[nodiscard]] constexpr std::size_t reserve_estimate(std::size_t est_bytes) noexcept
{
  return with_slack(std::max<std::size_t>(est_bytes, 1));
}

/// @p fraction of @p capacity in bytes, rounded to the nearest byte and clamped to
/// [0, capacity]. Rounding keeps a fraction computed back from a byte threshold on that threshold.
[[nodiscard]] constexpr std::size_t fraction_of(double fraction, std::size_t capacity) noexcept
{
  if (!(fraction > 0.0)) { return 0; }
  if (fraction >= 1.0) { return capacity; }
  auto const bytes = fraction * static_cast<double>(capacity) + 0.5;
  return bytes >= static_cast<double>(capacity) ? capacity : static_cast<std::size_t>(bytes);
}

/**
 * @brief Consumed bytes of a space: `max_memory - min(max_memory, available)`.
 *
 * Available memory is `capacity - allocated`, so this undercounts allocated bytes by
 * `capacity - max_memory`. It is deliberately the quantity `should_downgrade_memory` compares
 * with the downgrade thresholds, and every preload ceiling is in these units; budgets and
 * admission checks must both use it.
 */
[[nodiscard]] constexpr std::size_t consumed_bytes(std::size_t max_memory,
                                                   std::size_t available) noexcept
{
  return max_memory - std::min(max_memory, available);
}

// ---------------------------------------------------------------------------------------------
// Ceilings
// ---------------------------------------------------------------------------------------------

/// The size limits of one memory space, as its getters report them.
struct space_limits {
  std::size_t capacity{0};        ///< configured capacity: the base of every fraction
  std::size_t max_memory{0};      ///< get_max_memory(): the reservation limit
  std::size_t stop_threshold{0};  ///< downgrade stop threshold, in consumed bytes (derived)
};

/// The two GPU ceilings of one GPU, in consumed bytes. 0 disables that class.
struct gpu_ceiling_bytes {
  std::size_t committed{0};
  std::size_t speculative{0};
};

/**
 * @brief A ceiling fraction converted to consumed bytes of a space.
 *
 * Clamped to `min(stop_threshold, max_memory)`, so a ceiling is never above the downgrade stop
 * threshold, nor above what the space can ever consume (the stop threshold is a fraction of
 * capacity, which can exceed the reservation limit).
 */
[[nodiscard]] constexpr std::size_t ceiling_bytes(double ceiling_fraction_of_capacity,
                                                  space_limits const& limits) noexcept
{
  auto const bytes = fraction_of(ceiling_fraction_of_capacity, limits.capacity);
  return std::min({bytes, limits.stop_threshold, limits.max_memory});
}

/**
 * @brief The committed and speculative ceilings of a GPU, in consumed bytes.
 *
 * Derived from @ref gpu_ceilings (the same rule config validation applies).
 */
[[nodiscard]] constexpr gpu_ceiling_bytes gpu_ceilings_bytes(preload_executor_config const& cfg,
                                                             space_limits const& limits) noexcept
{
  if (limits.capacity == 0) { return {}; }
  auto const capacity  = static_cast<double>(limits.capacity);
  auto const fractions = gpu_ceilings(cfg, static_cast<double>(limits.stop_threshold) / capacity);
  return {ceiling_bytes(fractions.committed, limits), ceiling_bytes(fractions.speculative, limits)};
}

/// The DISK->HOST staging ceiling of a HOST space, in consumed bytes. 0 disables staging there.
[[nodiscard]] constexpr std::size_t host_ceiling_bytes(preload_executor_config const& cfg,
                                                       space_limits const& limits) noexcept
{
  if (limits.capacity == 0) { return 0; }
  auto const stop =
    static_cast<double>(limits.stop_threshold) / static_cast<double>(limits.capacity);
  return ceiling_bytes(ceiling_fraction(stop, cfg.host_headroom_fraction), limits);
}

// ---------------------------------------------------------------------------------------------
// Lane caps
// ---------------------------------------------------------------------------------------------

/// Concurrent promotions into one GPU: @c concurrent_preloads when set, else 5 with C2C and 1
/// without.
[[nodiscard]] constexpr int h2g_cap(preload_executor_config const& cfg, bool c2c) noexcept
{
  if (cfg.concurrent_preloads) { return static_cast<int>(*cfg.concurrent_preloads); }
  return c2c ? 5 : 1;
}

/// Concurrent promotions out of one DISK space: @c concurrent_preloads when set, else 1. Sirius
/// has no GPUDirect Storage path for DISK spaces yet (every DISK->GPU copy bounces through host
/// memory), so the auto width is unconditionally 1.
[[nodiscard]] constexpr int disk_cap(preload_executor_config const& cfg,
                                     [[maybe_unused]] bool gds) noexcept
{
  if (cfg.concurrent_preloads) { return static_cast<int>(*cfg.concurrent_preloads); }
  return 1;
}

/**
 * @brief Whether a promotion takes one of the target GPU's HOST->GPU tokens.
 *
 * Every copy into a GPU over the host link does: HOST->GPU, and DISK->GPU without GPUDirect
 * Storage (it bounces through host memory). DISK->GPU with GDS and DISK->HOST do not.
 */
[[nodiscard]] constexpr bool takes_h2g_token(cucascade::memory::Tier source,
                                             cucascade::memory::Tier target_tier,
                                             bool gds) noexcept
{
  using cucascade::memory::Tier;
  if (target_tier != Tier::GPU) { return false; }
  return source == Tier::HOST || (source == Tier::DISK && !gds);
}

// ---------------------------------------------------------------------------------------------
// Queue margins and repository windows
// ---------------------------------------------------------------------------------------------

/// A half-open range of positions in a pop order.
struct position_window {
  std::size_t begin{0};
  std::size_t end{0};

  [[nodiscard]] constexpr bool contains(std::size_t position) const noexcept
  {
    return position >= begin && position < end;
  }
};

/// The head margin H: the configured value, or the pipeline executor's thread count for auto.
[[nodiscard]] constexpr std::size_t resolve_head_margin(preload_executor_config const& cfg,
                                                        int pipeline_threads_per_gpu) noexcept
{
  auto const h = cfg.head_margin_tasks >= 0 ? cfg.head_margin_tasks : pipeline_threads_per_gpu;
  return h > 0 ? static_cast<std::size_t>(h) : 0;
}

/// Positions of a device's own bucket whose inputs are considered: `[H, H + W)`.
[[nodiscard]] constexpr position_window exact_bucket_window(std::size_t head_margin,
                                                            std::size_t lookahead) noexcept
{
  return {head_margin, sat_add(head_margin, lookahead)};
}

/**
 * @brief Positions of the no-preference bucket whose inputs are considered.
 *
 * Every ready device pops that bucket after its own, so it drains up to @p n_gpus times faster
 * and its margin scales with the GPU count: `[H x n_gpus, H x n_gpus + W)`. Positions are those
 * of @ref shared_bucket_position.
 */
[[nodiscard]] constexpr position_window shared_bucket_window(std::size_t head_margin,
                                                             std::size_t lookahead,
                                                             std::size_t n_gpus) noexcept
{
  auto const begin = sat_mul(head_margin, std::max<std::size_t>(n_gpus, 1));
  return {begin, sat_add(begin, lookahead)};
}

/**
 * @brief Position of the no-preference bucket's item @p index in the pop order it is judged by.
 *
 * With one GPU that bucket is popped only after the GPU's own bucket, so its positions continue
 * after the @p exact_bucket_size items there. With several GPUs it is judged on its own.
 */
[[nodiscard]] constexpr std::size_t shared_bucket_position(std::size_t index,
                                                           std::size_t exact_bucket_size,
                                                           std::size_t n_gpus) noexcept
{
  return n_gpus == 1 ? sat_add(exact_bucket_size, index) : index;
}

/**
 * @brief Whether a partition is one a ready, partition-draining consumer may get promoted early.
 *
 * @param rank The partition's rank among the repository's non-empty partitions in ascending
 *             order. Rank 0 is the partition the task creator drains next, which is left alone.
 * @param lookahead `repository_partition_lookahead`: ranks `[1, lookahead]` are admitted.
 */
[[nodiscard]] constexpr bool class_b_partition_admitted(std::size_t rank, int lookahead) noexcept
{
  return rank >= 1 && lookahead > 0 && rank <= static_cast<std::size_t>(lookahead);
}

/// Whether FIFO position @p position of a partition is inside `repository_window`.
[[nodiscard]] constexpr bool fifo_admitted(std::size_t position, int window) noexcept
{
  return window > 0 && position < static_cast<std::size_t>(window);
}

// ---------------------------------------------------------------------------------------------
// Target choice
// ---------------------------------------------------------------------------------------------

/// Which queue bucket a class-A item was found in.
enum class bucket_kind : std::uint8_t {
  exact,  ///< the bucket of one device
  shared  ///< the no-preference bucket
};

/// Everything the target choice depends on, for one candidate.
struct target_request {
  klass k{klass::A};
  /// Where the candidate is (HOST or DISK; anything else is never planned).
  cucascade::memory::Tier location{cucascade::memory::Tier::SIZE};
  bool gpu_banned{false};
  bool speculative_banned{false};

  // Class A.
  bucket_kind bucket{bucket_kind::exact};
  /// The exact bucket's device id.
  std::optional<int> bucket_device;

  // Class B.
  /// Every FULL-barrier producer of the consuming operator has finished.
  bool consumer_ready{false};
  /// The consumer drains one partition per task, in ascending order.
  bool partition_consumer{false};
  /// The candidate's partition rank among non-empty partitions (0 = the creator's next).
  std::size_t partition_rank{0};
  /// Device of the candidate's partition under the consumer's placement; empty if there is no
  /// placement or it does not cover the partition.
  std::optional<int> placement_device;

  /// GPU device ids the query runs on.
  std::span<int const> active_gpus{};
};

/// What the planner does with a candidate.
enum class target_decision : std::uint8_t {
  gpu,           ///< promote to @ref target_choice::device
  host_staging,  ///< stage DISK->HOST
  not_planned    ///< leave it alone
};

/// Why a candidate is not planned.
enum class skip_reason : std::uint8_t {
  none,
  creator_next,  ///< it is in the partition the task creator drains next
  no_target      ///< no deterministic GPU and no staging possible
};

/// Result of @ref choose_target.
struct target_choice {
  target_decision decision{target_decision::not_planned};
  /// GPU target, or for staging the GPU whose NUMA-local host is preferred; -1 if none.
  int device{-1};
  std::uint8_t band{band_host_staging};
  /// For a GPU target: whether the speculative ceiling and window apply (class B).
  bool speculative{false};
  skip_reason reason{skip_reason::none};
};

/**
 * @brief Choose the target of one candidate.
 *
 * - Class A in a device's bucket -> that GPU (committed). In the no-preference bucket -> the
 *   single active GPU, or HOST staging with several GPUs, since the consumer's device is unknown.
 * - Class B whose consumer waits on an unfinished FULL barrier -> HOST staging.
 * - Class B of a ready partition-draining consumer, in an admitted partition past the creator's
 *   next one -> the partition's placement device, else the single active GPU (speculative).
 * - Class B of any other ready consumer -> not planned: the creator pops it next.
 * - A candidate banned from GPUs (or, for class B, banned from speculation) -> HOST staging.
 *
 * HOST staging needs a DISK-resident candidate and `stage_to_host`; otherwise not planned.
 * A GPU target is used only when the consumer's device is deterministic, so preload never moves
 * a batch between GPUs.
 */
[[nodiscard]] constexpr target_choice choose_target(target_request const& r,
                                                    preload_executor_config const& cfg) noexcept
{
  using cucascade::memory::Tier;
  auto const not_planned = [](skip_reason reason) {
    return target_choice{target_decision::not_planned, -1, band_host_staging, false, reason};
  };
  if (r.location != Tier::HOST && r.location != Tier::DISK) {
    return not_planned(skip_reason::no_target);
  }
  auto const stage = [&](std::optional<int> hint) {
    if (r.location != Tier::DISK || !cfg.stage_to_host) {
      return not_planned(skip_reason::no_target);
    }
    return target_choice{target_decision::host_staging,
                         hint.value_or(-1),
                         band_host_staging,
                         false,
                         skip_reason::none};
  };
  auto const single_gpu =
    r.active_gpus.size() == 1 ? std::optional<int>{r.active_gpus.front()} : std::nullopt;
  auto const gpu = [](int device, bool speculative) {
    return target_choice{target_decision::gpu, device, band_gpu, speculative, skip_reason::none};
  };

  if (r.k == klass::A) {
    auto const device = r.bucket == bucket_kind::exact ? r.bucket_device : single_gpu;
    if (!device) { return stage(std::nullopt); }
    if (r.gpu_banned) { return stage(device); }
    return gpu(*device, false);
  }

  if (!r.consumer_ready) { return stage(std::nullopt); }
  if (!r.partition_consumer) { return not_planned(skip_reason::no_target); }
  if (r.partition_rank == 0) { return not_planned(skip_reason::creator_next); }
  if (!class_b_partition_admitted(r.partition_rank, cfg.repository_partition_lookahead)) {
    return not_planned(skip_reason::no_target);
  }
  auto const device = r.placement_device ? r.placement_device : single_gpu;
  if (!device) { return not_planned(skip_reason::no_target); }
  if (r.gpu_banned || r.speculative_banned) { return stage(device); }
  return gpu(*device, true);
}

/**
 * @brief The HOST space (by device id) to stage a DISK-resident candidate into.
 *
 * In order: the NUMA node of the GPU the batch is headed for, the HOST it lived on before DISK,
 * the NUMA node of the GPU it was first evicted from, else -1. A NUMA node of -1 means unknown
 * and is skipped.
 */
[[nodiscard]] constexpr int choose_staging_host(int numa_of_target_gpu,
                                                std::optional<int> origin_host,
                                                int numa_of_origin_gpu) noexcept
{
  if (numa_of_target_gpu >= 0) { return numa_of_target_gpu; }
  if (origin_host && *origin_host >= 0) { return *origin_host; }
  if (numa_of_origin_gpu >= 0) { return numa_of_origin_gpu; }
  return -1;
}

/// @ref choose_staging_host, skipping nodes for which @p has_host is false (no HOST space there).
template <typename HasHost>
[[nodiscard]] constexpr int choose_staging_host(int numa_of_target_gpu,
                                                std::optional<int> origin_host,
                                                int numa_of_origin_gpu,
                                                HasHost&& has_host) noexcept
{
  for (int const node : {numa_of_target_gpu, origin_host.value_or(-1), numa_of_origin_gpu}) {
    if (node >= 0 && has_host(node)) { return node; }
  }
  return -1;
}

// ---------------------------------------------------------------------------------------------
// Budgets
// ---------------------------------------------------------------------------------------------

/// Why a target refused an item. Each value maps to one `refused_*` counter.
enum class refusal : std::uint8_t {
  none,
  held_off,            ///< a task on the GPU (or a NUMA-local GPU) is stalled or has an OOM retry
  epoch,               ///< a worker was refused there since the last memory event
  trigger,             ///< the space is at or above its downgrade trigger
  ceiling,             ///< consumed plus unreserved bytes leave no room under the ceiling
  window,              ///< the lookahead window of preloaded-but-unconsumed bytes is full
  speculative_window,  ///< the class-B share of that window is full
  inflight,            ///< the bytes being copied into the GPU are at their cap
  batch_too_big,       ///< the item exceeds the per-batch limit
  no_target            ///< the item has no usable target
};

inline constexpr std::size_t refusal_count = static_cast<std::size_t>(refusal::no_target) + 1;

/// Bytes a target can take now, and what limits it when that is less than asked.
struct budget {
  std::size_t bytes{0};
  refusal reason{refusal::none};
};

/// Whether a task on @p device is stalled in its reserve phase or has an OOM retry pending.
[[nodiscard]] inline bool held_off(memory::memory_wait_board const& board, int device) noexcept
{
  return board.waiting(device) > 0 || board.oom_pending(device) > 0;
}

/// Live state of one GPU for a planner pass.
struct gpu_budget_inputs {
  int device{-1};
  std::size_t consumed{0};    ///< get_max_memory() - available
  bool above_trigger{false};  ///< should_downgrade_memory()
  std::uint64_t mem_epoch{0};
  std::uint64_t gate_epoch{~std::uint64_t{0}};
  /// Bytes preload's running jobs will reserve here but have not yet. A job's reservation is in
  /// @c consumed once made, as in the worker's own ceiling check, so it is not counted twice.
  std::size_t unreserved_bytes{0};
  gpu_ceiling_bytes ceilings{};
  std::size_t lookahead_bytes{0};    ///< device_preloading_budget x capacity
  std::size_t outstanding{0};        ///< preloaded and not yet consumed
  std::size_t speculative_bytes{0};  ///< speculative_fraction x capacity
  std::size_t outstanding_b{0};      ///< class-B part of @c outstanding
  std::size_t max_batch_bytes{0};    ///< max_batch_fraction x capacity
};

/**
 * @brief Remaining room of one GPU during head-of-line assignment.
 *
 * @c closed_a and @c closed_b close a class for the rest of the pass; a class-A refusal closes
 * both, since every class-B limit is at least as tight.
 */
struct gpu_budget_state {
  std::size_t room_committed{0};
  std::size_t room_speculative{0};
  std::size_t window{0};
  std::size_t speculative_window{0};
  std::size_t max_batch_bytes{0};
  refusal closed_a{refusal::none};
  refusal closed_b{refusal::none};

  /// Bytes an item of class @p k may take now, and the limit that binds.
  [[nodiscard]] constexpr budget available(klass k) const noexcept
  {
    auto const closed = k == klass::A ? closed_a : closed_b;
    if (closed != refusal::none) { return {0, closed}; }
    budget b{k == klass::A ? room_committed : room_speculative, refusal::ceiling};
    if (window < b.bytes) { b = {window, refusal::window}; }
    if (k == klass::B && speculative_window < b.bytes) {
      b = {speculative_window, refusal::speculative_window};
    }
    return b;
  }

  /// Account for @p bytes assigned to an item of class @p k.
  constexpr void take(klass k, std::size_t bytes) noexcept
  {
    room_committed   = sat_sub(room_committed, bytes);
    room_speculative = sat_sub(room_speculative, bytes);
    window           = sat_sub(window, bytes);
    if (k == klass::B) { speculative_window = sat_sub(speculative_window, bytes); }
  }

  /// Close class @p k (and class B with it, for A) with @p reason.
  constexpr void close(klass k, refusal reason) noexcept
  {
    if (k == klass::A && closed_a == refusal::none) { closed_a = reason; }
    if (closed_b == refusal::none) { closed_b = reason; }
  }
};

/**
 * @brief The planner's starting budget state of one GPU.
 *
 * The GPU is closed for the pass when a task there is held off, when a worker was refused since
 * the last memory event, or when the GPU is at or above its downgrade trigger. Otherwise each
 * class gets the room under its ceiling (after preload's unreserved bytes) and the windows.
 */
[[nodiscard]] inline gpu_budget_state make_gpu_budget_state(
  gpu_budget_inputs const& in, memory::memory_wait_board const& board) noexcept
{
  gpu_budget_state s;
  s.max_batch_bytes = in.max_batch_bytes;
  auto const closed = held_off(board, in.device)      ? refusal::held_off
                      : in.gate_epoch == in.mem_epoch ? refusal::epoch
                      : in.above_trigger              ? refusal::trigger
                                                      : refusal::none;
  if (closed != refusal::none) {
    s.closed_a = s.closed_b = closed;
    return s;
  }
  auto const planned   = sat_add(in.consumed, in.unreserved_bytes);
  s.room_committed     = sat_sub(in.ceilings.committed, planned);
  s.room_speculative   = sat_sub(in.ceilings.speculative, planned);
  s.window             = sat_sub(in.lookahead_bytes, in.outstanding);
  s.speculative_window = sat_sub(in.speculative_bytes, in.outstanding_b);
  return s;
}

/// Bytes an item of class @p k may promote into one GPU now, and the binding limit.
[[nodiscard]] inline budget gpu_budget(gpu_budget_inputs const& in,
                                       memory::memory_wait_board const& board,
                                       klass k) noexcept
{
  return make_gpu_budget_state(in, board).available(k);
}

/// Pressure state of one GPU, as seen by a HOST space on the same NUMA node.
struct gpu_pressure {
  int device{-1};
  bool above_trigger{false};
};

/// Live state of one HOST space for a planner pass.
struct host_budget_inputs {
  std::size_t consumed{0};
  bool above_trigger{false};
  std::uint64_t mem_epoch{0};
  std::uint64_t gate_epoch{~std::uint64_t{0}};
  /// As @ref gpu_budget_inputs::unreserved_bytes.
  std::size_t unreserved_bytes{0};
  std::size_t ceiling{0};
  /// GPUs on the same NUMA node: their spills need this host's headroom first.
  std::span<gpu_pressure const> local_gpus{};
};

/// Whether any of @p gpus is held off or at its downgrade trigger.
[[nodiscard]] inline bool any_gpu_pressed(std::span<gpu_pressure const> gpus,
                                          memory::memory_wait_board const& board) noexcept
{
  return std::any_of(gpus.begin(), gpus.end(), [&](gpu_pressure const& g) {
    return g.above_trigger || held_off(board, g.device);
  });
}

/**
 * @brief Bytes DISK->HOST staging may put into one HOST space now, and the binding limit.
 *
 * Zero while a NUMA-local GPU is pressed (a hold-off: GPU spills need the host's headroom
 * first), after a worker refusal since the last memory event, or at the host's trigger.
 */
[[nodiscard]] inline budget host_budget(host_budget_inputs const& in,
                                        memory::memory_wait_board const& board) noexcept
{
  if (any_gpu_pressed(in.local_gpus, board)) { return {0, refusal::held_off}; }
  if (in.gate_epoch == in.mem_epoch) { return {0, refusal::epoch}; }
  if (in.above_trigger) { return {0, refusal::trigger}; }
  return {sat_sub(in.ceiling, sat_add(in.consumed, in.unreserved_bytes)), refusal::ceiling};
}

/// Remaining room of one HOST space during head-of-line assignment.
struct host_budget_state {
  std::size_t room{0};
  refusal closed{refusal::none};
};

/// The planner's starting budget state of one HOST space.
[[nodiscard]] inline host_budget_state make_host_budget_state(
  host_budget_inputs const& in, memory::memory_wait_board const& board) noexcept
{
  auto const b = host_budget(in, board);
  if (b.reason != refusal::ceiling) { return {0, b.reason}; }
  return {b.bytes, refusal::none};
}

// ---------------------------------------------------------------------------------------------
// Head-of-line assignment
// ---------------------------------------------------------------------------------------------

/// One planned item, in plan order, as the assignment sees it.
struct assign_item {
  klass k{klass::A};
  /// GPU target by index; empty for a staging-only item.
  std::optional<std::size_t> gpu;
  /// HOST target by index: the target of a staging-only item, or the fallback of a DISK-resident
  /// GPU item. Empty when staging is not possible. The caller leaves it empty whenever
  /// `stage_to_host` is false: @ref assign_head_of_line does not read the configuration.
  std::optional<std::size_t> host;
  /// Bytes the item needs on the GPU and on the HOST.
  std::size_t gpu_bytes{0};
  std::size_t host_bytes{0};
};

/// One accepted item.
struct assignment {
  std::size_t item{0};  ///< index into the items
  target t{};
  bool fallback{false};  ///< a GPU item staged to HOST because its GPU had no room
};

/// Result of @ref assign_head_of_line.
struct assignment_result {
  std::vector<assignment> assigned;
  /// Refusals per @ref refusal value; one per refused target of an item.
  std::array<std::uint64_t, refusal_count> refusals{};
};

/**
 * @brief Assign items to targets in plan order, strictly head-of-line per target.
 *
 * - An item that fits its GPU's class budget takes it.
 * - The first item that does not fit closes its GPU for its class (class A closes both) for the
 *   rest of the pass, so a later, smaller item can never overtake it.
 * - An item above the GPU's per-batch limit never targets that GPU and closes nothing.
 * - An item refused by its GPU for any reason other than a hold-off falls back to its HOST target
 *   when it has one.
 * - HOST targets follow the same head-of-line rule.
 *
 * @param items Items in plan order (sorted by @ref plan_key).
 * @param gpus Starting state of each GPU, by index; updated as items are assigned.
 * @param hosts Starting state of each HOST space, by index; updated likewise.
 */
[[nodiscard]] inline assignment_result assign_head_of_line(std::span<assign_item const> items,
                                                           std::span<gpu_budget_state> gpus,
                                                           std::span<host_budget_state> hosts)
{
  assignment_result result;
  auto const refuse = [&](refusal r) { ++result.refusals[static_cast<std::size_t>(r)]; };

  for (std::size_t i = 0; i < items.size(); ++i) {
    auto const& item    = items[i];
    bool const has_gpu  = item.gpu && *item.gpu < gpus.size();
    bool const has_host = item.host && *item.host < hosts.size();
    if (!has_gpu && !has_host) {
      refuse(refusal::no_target);
      continue;
    }

    bool try_host = has_host;
    if (has_gpu) {
      auto& g = gpus[*item.gpu];
      if (item.gpu_bytes > g.max_batch_bytes) {
        refuse(refusal::batch_too_big);
      } else {
        auto const b = g.available(item.k);
        if (item.gpu_bytes <= b.bytes) {
          g.take(item.k, item.gpu_bytes);
          result.assigned.push_back({i, {cucascade::memory::Tier::GPU, *item.gpu}, false});
          continue;
        }
        refuse(b.reason);
        g.close(item.k, b.reason);
        if (b.reason == refusal::held_off) { try_host = false; }
      }
    }
    if (!try_host) { continue; }

    auto& h = hosts[*item.host];
    if (h.closed != refusal::none) {
      refuse(h.closed);
      continue;
    }
    if (item.host_bytes > h.room) {
      refuse(refusal::ceiling);
      h.closed = refusal::ceiling;
      continue;
    }
    h.room -= item.host_bytes;
    result.assigned.push_back({i, {cucascade::memory::Tier::HOST, *item.host}, has_gpu});
  }
  return result;
}

}  // namespace sirius::preload
