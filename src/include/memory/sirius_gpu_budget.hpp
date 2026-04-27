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

#include "memory/sirius_memory_reservation_manager.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace sirius::memory {

/**
 * @brief Snapshot of GPU memory budget partitioned across pipeline concerns.
 *
 * The "build" budget is the headroom that a single hash-join can spend on
 * the build table plus its cuco hash table without colliding with concurrent
 * probe traffic, other pipeline operators, or the cucascade-bug headroom.
 *
 * All fields are bytes. `for_build` is the value most decision sites care
 * about; the other fields are exposed for diagnostics and tuning.
 */
struct gpu_budget_snapshot {
  std::size_t total_available;      ///< Sum of GPU memory_space available bytes
  std::size_t reserved_for_probe;   ///< Bound on probe-side in-flight bytes
  std::size_t headroom;             ///< Slack for cucascade drift + transient peaks
  std::size_t for_build;            ///< total_available - reserved_for_probe - headroom
};

/**
 * @brief Inputs that shape the budget partition.
 *
 * `scan_task_batch_size` and `pipeline_threads` come from the live operator_params
 * and executor config respectively; they bound the probe-side queue ceiling.
 */
struct gpu_budget_inputs {
  std::uint64_t scan_task_batch_size;
  int pipeline_threads;
};

/**
 * @brief Register the most-recently-constructed manager for budget queries.
 *
 * Set by `sirius_memory_reservation_manager`'s constructor; cleared in dtor.
 * Reads are lock-free; concurrent setter from a parallel test fixture is a
 * known harmless race (last-writer-wins).
 *
 * Use this only at decision sites where threading the manager through the
 * call chain would be unreasonably invasive (`update_join_exec_mode`,
 * `determine_num_partitions`). Never for allocation paths.
 */
class gpu_budget_registry {
 public:
  static void set_current(const sirius_memory_reservation_manager* mgr) noexcept
  {
    _current.store(mgr, std::memory_order_release);
  }
  static const sirius_memory_reservation_manager* current() noexcept
  {
    return _current.load(std::memory_order_acquire);
  }

 private:
  inline static std::atomic<const sirius_memory_reservation_manager*> _current{nullptr};
};

/**
 * @brief Compute the build-side memory budget.
 *
 * If no manager has been registered, returns a snapshot with zeros — callers
 * must fall back to the existing static `_max_build_hash_table_bytes` config.
 *
 * Probe reservation: `scan_task_batch_size * pipeline_threads * 2` (double-buffer).
 *
 * Headroom: `max(2 GB, 0.20 * total_available)`.  TODO(cucascade-bug-fixed):
 * once cross-stream alloc/dealloc no longer drifts the unsigned pool counter,
 * lower the headroom factor to ~0.05; that adds ~5 GB of build budget on a
 * 34 GB A100 pool.
 *
 * Caller passes the operator_params-derived inputs; this function does not
 * read live config so it can be called from headers.
 */
inline gpu_budget_snapshot snapshot_build_budget(const gpu_budget_inputs& inputs)
{
  gpu_budget_snapshot snap{};
  const auto* mgr = gpu_budget_registry::current();
  if (mgr == nullptr) { return snap; }

  snap.total_available = mgr->get_total_available_memory();
  if (snap.total_available == 0) { return snap; }

  // Probe reservation: bounded queue allows at most 2× scan_task_batch_size per
  // pipeline thread in flight (one batch produced + one being consumed).
  const std::uint64_t probe_per_thread = inputs.scan_task_batch_size * 2;
  snap.reserved_for_probe = probe_per_thread * static_cast<std::uint64_t>(
                                                 inputs.pipeline_threads > 0 ? inputs.pipeline_threads
                                                                             : 1);

  // Conservative headroom for the cucascade cross-stream attribution drift.
  // TODO(cucascade-bug-fixed): drop to ~0.05 once origin tracking lands.
  constexpr std::size_t kMinHeadroomBytes = 2ULL << 30;  // 2 GiB
  constexpr double kHeadroomFraction      = 0.20;
  const std::size_t headroom_floor =
    static_cast<std::size_t>(static_cast<double>(snap.total_available) * kHeadroomFraction);
  snap.headroom = std::max(kMinHeadroomBytes, headroom_floor);

  if (snap.total_available > snap.reserved_for_probe + snap.headroom) {
    snap.for_build = snap.total_available - snap.reserved_for_probe - snap.headroom;
  }
  return snap;
}

/**
 * @brief Conservative default-input snapshot for callers that don't have
 * operator_params + executor config plumbed through.
 *
 * Defaults: 200 MB scan-batch (matches SF=100 YAML), 8 pipeline threads.
 * TODO(plumb-config): replace with a snapshot taken from the live config so
 * the budget tracks user overrides.
 */
inline gpu_budget_snapshot snapshot_build_budget_default()
{
  gpu_budget_inputs inputs{};
  inputs.scan_task_batch_size = 200ULL * 1024 * 1024;
  inputs.pipeline_threads     = 8;
  return snapshot_build_budget(inputs);
}

}  // namespace sirius::memory
