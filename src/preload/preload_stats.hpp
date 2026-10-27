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

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace sirius::preload {

// Every counter of the preload executor, in one list so the enum, the names and the storage
// cannot drift apart. Groups:
// - intake: candidates arriving from batch_downgraded events, and why some are dropped;
// - planning: planner passes and the items they produce;
// - planner refusals (refused_*): why the planner did not assign an item to a target;
// - worker refusals and work: what promotion attempts ended with;
// - accuracy: whether preloaded batches were consumed where they were put (mispredicted_device_b
//   is the class-B part of mispredicted_device);
// - readiness (input_*): how tasks found their downgraded inputs (see consumption_tracker);
// - timing: microseconds spent per promotion kind, and the planner's wake-to-dispatch latency.
// High-water marks (*_max) are raised, never added to.
#define SIRIUS_PRELOAD_COUNTERS(X) \
  X(events_received)               \
  X(merged)                        \
  X(dropped_unknown_query)         \
  X(dropped_expired)               \
  X(dropped_unsupported)           \
  X(dropped_overflow)              \
  X(dropped_lock_attempts)         \
  X(dropped_promotion_cap)         \
  X(dropped_stale_record)          \
  X(passes)                        \
  X(passes_noop)                   \
  X(items_planned)                 \
  X(class_a_items)                 \
  X(class_b_items)                 \
  X(staging_items)                 \
  X(skipped_creator_next)          \
  X(refused_held_off)              \
  X(refused_epoch)                 \
  X(refused_trigger)               \
  X(refused_ceiling)               \
  X(refused_window)                \
  X(refused_speculative_window)    \
  X(refused_inflight)              \
  X(refused_batch_too_big)         \
  X(refused_no_target)             \
  X(no_memory)                     \
  X(gated)                         \
  X(held_off)                      \
  X(promoted_h2g)                  \
  X(promoted_d2g)                  \
  X(promoted_d2h)                  \
  X(promoted_relocate_d2h)         \
  X(bytes_h2g)                     \
  X(bytes_d2g)                     \
  X(bytes_d2h)                     \
  X(bytes_relocate_d2h)            \
  X(contended)                     \
  X(stale)                         \
  X(already_resident)              \
  X(retired)                       \
  X(expired)                       \
  X(unsupported)                   \
  X(oom)                           \
  X(too_big)                       \
  X(gpu_banned)                    \
  X(failed)                        \
  X(hit)                           \
  X(bounce)                        \
  X(mispredicted_device)           \
  X(mispredicted_device_b)         \
  X(wasted_unconsumed)             \
  X(late_outcome_after_retire)     \
  X(input_ready)                   \
  X(input_waited)                  \
  X(input_waited_us)               \
  X(input_upgraded)                \
  X(input_upgraded_us)             \
  X(promote_us_sum_h2g)            \
  X(promote_us_sum_d2g)            \
  X(promote_us_sum_d2h)            \
  X(promote_us_sum_relocate_d2h)   \
  X(promote_us_max_h2g)            \
  X(promote_us_max_d2g)            \
  X(promote_us_max_d2h)            \
  X(promote_us_max_relocate_d2h)   \
  X(wake_to_dispatch_us_max)

/// One preload counter. See @ref SIRIUS_PRELOAD_COUNTERS for the groups.
enum class counter : std::uint8_t {
#define SIRIUS_PRELOAD_COUNTER_ENUM(name) name,
  SIRIUS_PRELOAD_COUNTERS(SIRIUS_PRELOAD_COUNTER_ENUM)
#undef SIRIUS_PRELOAD_COUNTER_ENUM
};

/// Number of counters in @ref counter.
inline constexpr std::size_t counter_count = 0
#define SIRIUS_PRELOAD_COUNTER_ONE(name) +1
  SIRIUS_PRELOAD_COUNTERS(SIRIUS_PRELOAD_COUNTER_ONE)
#undef SIRIUS_PRELOAD_COUNTER_ONE
  ;

/// The counter's name, as used in logs.
[[nodiscard]] constexpr std::string_view counter_name(counter c) noexcept
{
  constexpr std::array<std::string_view, counter_count> names{
#define SIRIUS_PRELOAD_COUNTER_NAME(name) #name,
    SIRIUS_PRELOAD_COUNTERS(SIRIUS_PRELOAD_COUNTER_NAME)
#undef SIRIUS_PRELOAD_COUNTER_NAME
  };
  auto const i = static_cast<std::size_t>(c);
  return i < names.size() ? names[i] : std::string_view{"unknown"};
}

/// Counters of one query. Not atomic: the owner updates them under its own lock.
struct per_query_stats {
  std::array<std::uint64_t, counter_count> values{};

  void add(counter c, std::uint64_t n = 1) noexcept { values[static_cast<std::size_t>(c)] += n; }

  [[nodiscard]] std::uint64_t operator[](counter c) const noexcept
  {
    return values[static_cast<std::size_t>(c)];
  }
};

/**
 * @brief A point-in-time copy of the preload executor's counters.
 *
 * Counters are read one by one with relaxed loads, so a snapshot taken while the executor runs
 * need not be consistent across counters.
 */
struct preload_stats {
  std::array<std::uint64_t, counter_count> values{};
  /// Highest number of concurrent promotions seen on each lane.
  std::vector<std::uint64_t> lane_concurrency_max;
  /// Highest number of bytes being copied into each GPU at once.
  std::vector<std::uint64_t> gpu_inflight_bytes_max;

  [[nodiscard]] std::uint64_t operator[](counter c) const noexcept
  {
    return values[static_cast<std::size_t>(c)];
  }
};

/**
 * @brief The preload executor's live counters: relaxed atomics, safe to update from any thread.
 */
class preload_counters {
 public:
  /// Counters for @p n_lanes lanes and @p n_gpus GPUs (for the per-lane and per-GPU marks).
  explicit preload_counters(std::size_t n_lanes = 0, std::size_t n_gpus = 0)
    : _n_lanes(n_lanes),
      _n_gpus(n_gpus),
      _lane_max(std::make_unique<std::atomic<std::uint64_t>[]>(n_lanes)),
      _gpu_inflight_max(std::make_unique<std::atomic<std::uint64_t>[]>(n_gpus))
  {
  }

  preload_counters(preload_counters const&)            = delete;
  preload_counters& operator=(preload_counters const&) = delete;
  preload_counters(preload_counters&&)                 = delete;
  preload_counters& operator=(preload_counters&&)      = delete;

  void add(counter c, std::uint64_t n = 1) noexcept
  {
    _values[static_cast<std::size_t>(c)].fetch_add(n, std::memory_order_relaxed);
  }

  /// Raise a high-water counter to at least @p v.
  void raise(counter c, std::uint64_t v) noexcept
  {
    raise(_values[static_cast<std::size_t>(c)], v);
  }

  /// Raise lane @p lane's concurrency mark to at least @p v. Out-of-range lanes are ignored.
  void raise_lane_concurrency(std::size_t lane, std::uint64_t v) noexcept
  {
    if (lane < _n_lanes) { raise(_lane_max[lane], v); }
  }

  /// Raise GPU @p gpu's in-flight bytes mark to at least @p v. Out-of-range GPUs are ignored.
  void raise_gpu_inflight_bytes(std::size_t gpu, std::uint64_t v) noexcept
  {
    if (gpu < _n_gpus) { raise(_gpu_inflight_max[gpu], v); }
  }

  [[nodiscard]] std::uint64_t get(counter c) const noexcept
  {
    return _values[static_cast<std::size_t>(c)].load(std::memory_order_relaxed);
  }

  [[nodiscard]] preload_stats snapshot() const
  {
    preload_stats s;
    for (std::size_t i = 0; i < counter_count; ++i) {
      s.values[i] = _values[i].load(std::memory_order_relaxed);
    }
    s.lane_concurrency_max.reserve(_n_lanes);
    for (std::size_t i = 0; i < _n_lanes; ++i) {
      s.lane_concurrency_max.push_back(_lane_max[i].load(std::memory_order_relaxed));
    }
    s.gpu_inflight_bytes_max.reserve(_n_gpus);
    for (std::size_t i = 0; i < _n_gpus; ++i) {
      s.gpu_inflight_bytes_max.push_back(_gpu_inflight_max[i].load(std::memory_order_relaxed));
    }
    return s;
  }

 private:
  static void raise(std::atomic<std::uint64_t>& slot, std::uint64_t v) noexcept
  {
    auto cur = slot.load(std::memory_order_relaxed);
    while (cur < v && !slot.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {}
  }

  std::array<std::atomic<std::uint64_t>, counter_count> _values{};
  std::size_t _n_lanes;
  std::size_t _n_gpus;
  std::unique_ptr<std::atomic<std::uint64_t>[]> _lane_max;
  std::unique_ptr<std::atomic<std::uint64_t>[]> _gpu_inflight_max;
};

}  // namespace sirius::preload
