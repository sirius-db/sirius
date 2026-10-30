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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>

namespace sirius::memory {

/**
 * @brief The per-GPU budget of preloaded-but-not-yet-claimed bytes, released as soon as a task
 *        whose inputs were preloaded to a device is claimed by an executor.
 *
 * The preload executor charges a batch's estimate against the budget when a lane worker takes a
 * GPU promotion job, and the planner caps new promotions at `budget - outstanding(device)`. The
 * task scheduler's matcher releases a claimed task's charges the instant it pops the task, before
 * it publishes @c task_deployed, so the planner pass that event triggers already sees the freed
 * budget.
 *
 * Protocol (watch, charge, claim):
 * - @ref watch at preload intake of a @c batch_downgraded record creates or renews an entry.
 * - @ref charge when a lane worker takes a GPU job; it is refused (no charge) once the entry is
 *   @c claimed, so the worker drops the job.
 * - @ref release on any non-promoting outcome and on every ledger erase; idempotent.
 * - @ref reclassify_committed when a speculative (class B) entry becomes committed (class A).
 * - @ref claim from the matcher, releasing whatever device holds the charge.
 * - @ref unwatch when an id leaves both the candidate set and the ledger.
 *
 * Concurrency: one leaf mutex guards the id map; per-device atomics hold the charged totals for
 * lock-free reads; an atomic watched count gives the matcher a lock-free empty fast path. The
 * board calls nothing, so the only lock nesting with its owner is `executor._mu -> board mutex`.
 * Devices outside @c [0, max_device_id] are ignored; their totals read as zero.
 *
 * `job_seq` is a generation stamp: @ref release and @ref reclassify_committed act only on the
 * charge that carries the same stamp, so an old job's cleanup never disturbs a newer charge.
 */
class device_preload_budget {
 public:
  /// Board sized to cover every id in @p gpu_device_ids. A board with no devices charges nothing.
  explicit device_preload_budget(std::span<int const> gpu_device_ids)
    : _n(max_device_id(gpu_device_ids) + 1),
      _outstanding(std::make_unique<std::atomic<std::size_t>[]>(static_cast<std::size_t>(_n))),
      _outstanding_spec(std::make_unique<std::atomic<std::size_t>[]>(static_cast<std::size_t>(_n)))
  {
  }

  device_preload_budget(device_preload_budget const&)            = delete;
  device_preload_budget& operator=(device_preload_budget const&) = delete;
  device_preload_budget(device_preload_budget&&)                 = delete;
  device_preload_budget& operator=(device_preload_budget&&)      = delete;

  /// Preload intake: create @p batch_id's entry, or renew it as a fresh, unclaimed generation.
  /// May throw @c std::bad_alloc; intake's exception handling then drops the candidate.
  void watch(std::uint64_t batch_id)
  {
    std::lock_guard const lk{_mu};
    auto const [it, inserted] = _entries.try_emplace(batch_id);
    if (inserted) {
      _watched.fetch_add(1, std::memory_order_relaxed);
    } else {
      it->second.claimed = false;
    }
  }

  /// Charge @p bytes of @p batch_id's current generation to @p device. Returns false and charges
  /// nothing when the entry is missing or already claimed by a task.
  [[nodiscard]] bool charge(std::uint64_t batch_id,
                            std::uint64_t job_seq,
                            int device,
                            bool speculative,
                            std::size_t bytes) noexcept
  {
    std::lock_guard const lk{_mu};
    auto const it = _entries.find(batch_id);
    if (it == _entries.end() || it->second.claimed) { return false; }
    auto& e = it->second;
    if (e.bytes != 0) { subtract(e); }  // defensive: a prior charge was never released
    e.device      = device;
    e.speculative = speculative;
    e.bytes       = bytes;
    e.job_seq     = job_seq;
    add(e);
    return true;
  }

  /// Release @p batch_id's charge if it still carries @p job_seq. Idempotent.
  void release(std::uint64_t batch_id, std::uint64_t job_seq) noexcept
  {
    std::lock_guard const lk{_mu};
    auto const it = _entries.find(batch_id);
    if (it == _entries.end()) { return; }
    auto& e = it->second;
    if (e.bytes != 0 && e.job_seq == job_seq) { subtract(e); }
  }

  /// Class B -> A for @p batch_id's current generation: drop its speculative part.
  void reclassify_committed(std::uint64_t batch_id, std::uint64_t job_seq) noexcept
  {
    std::lock_guard const lk{_mu};
    auto const it = _entries.find(batch_id);
    if (it == _entries.end()) { return; }
    auto& e = it->second;
    if (e.bytes != 0 && e.job_seq == job_seq && e.speculative) {
      if (in_range(e.device)) {
        _outstanding_spec[static_cast<std::size_t>(e.device)].fetch_sub(e.bytes,
                                                                        std::memory_order_relaxed);
      }
      e.speculative = false;
    }
  }

  /// Matcher: release each claimed id's charge (whatever device holds it) and mark it claimed so a
  /// later charge is refused. noexcept.
  void claim(std::span<std::uint64_t const> batch_ids) noexcept
  {
    if (_watched.load(std::memory_order_relaxed) == 0) { return; }
    std::lock_guard const lk{_mu};
    for (auto const id : batch_ids) {
      claim_one_locked(id);
    }
  }

  /// Matcher: claim the ids of @p range's elements, where @p get_id(element) returns the id (or
  /// @c std::nullopt to skip). Allocation-free: it reads the task's existing batch range in place
  /// under a single board lock. @p GetId must be noexcept-callable.
  template <typename Range, typename GetId>
  void claim_range(Range const& range, GetId&& get_id) noexcept
  {
    if (_watched.load(std::memory_order_relaxed) == 0) { return; }
    std::lock_guard const lk{_mu};
    for (auto const& element : range) {
      if (std::optional<std::uint64_t> const id = get_id(element)) { claim_one_locked(*id); }
    }
  }

  /// Drop @p batch_id's entry and any remaining charge. noexcept.
  void unwatch(std::uint64_t batch_id) noexcept
  {
    std::lock_guard const lk{_mu};
    auto const it = _entries.find(batch_id);
    if (it == _entries.end()) { return; }
    if (it->second.bytes != 0) { subtract(it->second); }
    _entries.erase(it);
    _watched.fetch_sub(1, std::memory_order_relaxed);
  }

  /// Charged bytes on @p device (all classes).
  [[nodiscard]] std::size_t outstanding(int device) const noexcept
  {
    return in_range(device)
             ? _outstanding[static_cast<std::size_t>(device)].load(std::memory_order_relaxed)
             : 0;
  }

  /// Charged speculative (class B) bytes on @p device.
  [[nodiscard]] std::size_t outstanding_speculative(int device) const noexcept
  {
    return in_range(device)
             ? _outstanding_spec[static_cast<std::size_t>(device)].load(std::memory_order_relaxed)
             : 0;
  }

  /// Whether @p batch_id has been claimed by a task (and not since renewed by @ref watch). Lets
  /// the preload executor keep a claimed entry as a tombstone when its candidate is removed, so a
  /// delayed duplicate record cannot recreate an unclaimed watch.
  [[nodiscard]] bool is_claimed(std::uint64_t batch_id) const noexcept
  {
    std::lock_guard const lk{_mu};
    auto const it = _entries.find(batch_id);
    return it != _entries.end() && it->second.claimed;
  }

  /// Whether no id is watched (a lock-free fast path for the matcher).
  [[nodiscard]] bool empty() const noexcept
  {
    return _watched.load(std::memory_order_relaxed) == 0;
  }

  /// Number of watched ids.
  [[nodiscard]] std::size_t watched_count() const noexcept
  {
    return _watched.load(std::memory_order_relaxed);
  }

  /// Number of charges released by @ref claim, for stats and tests.
  [[nodiscard]] std::uint64_t claimed_releases() const noexcept
  {
    return _claimed_releases.load(std::memory_order_relaxed);
  }

  /// Bytes released by @ref claim, for stats and tests.
  [[nodiscard]] std::uint64_t claimed_bytes_released() const noexcept
  {
    return _claimed_bytes.load(std::memory_order_relaxed);
  }

 private:
  struct entry {
    bool claimed{false};
    int device{-1};
    bool speculative{false};
    std::size_t bytes{0};
    std::uint64_t job_seq{0};
  };

  /// Release @p batch_id's charge (whatever device holds it) and mark it claimed. Call under @c
  /// _mu.
  void claim_one_locked(std::uint64_t batch_id) noexcept
  {
    auto const it = _entries.find(batch_id);
    if (it == _entries.end()) { return; }
    auto& e = it->second;
    if (e.bytes != 0) {
      _claimed_bytes.fetch_add(e.bytes, std::memory_order_relaxed);
      _claimed_releases.fetch_add(1, std::memory_order_relaxed);
      subtract(e);
    }
    e.claimed = true;
  }

  [[nodiscard]] bool in_range(int device) const noexcept { return device >= 0 && device < _n; }

  [[nodiscard]] static int max_device_id(std::span<int const> ids) noexcept
  {
    int m = -1;
    for (int const id : ids) {
      if (id > m) { m = id; }
    }
    return m;
  }

  /// Add @p e's charge to its device totals. Call under @c _mu with @p e newly charged.
  void add(entry const& e) noexcept
  {
    if (!in_range(e.device)) { return; }
    auto const d = static_cast<std::size_t>(e.device);
    _outstanding[d].fetch_add(e.bytes, std::memory_order_relaxed);
    if (e.speculative) { _outstanding_spec[d].fetch_add(e.bytes, std::memory_order_relaxed); }
  }

  /// Remove @p e's charge from its device totals and clear it. Call under @c _mu.
  void subtract(entry& e) noexcept
  {
    if (in_range(e.device)) {
      auto const d = static_cast<std::size_t>(e.device);
      _outstanding[d].fetch_sub(e.bytes, std::memory_order_relaxed);
      if (e.speculative) { _outstanding_spec[d].fetch_sub(e.bytes, std::memory_order_relaxed); }
    }
    e.bytes       = 0;
    e.device      = -1;
    e.speculative = false;
  }

  int _n{0};
  std::unique_ptr<std::atomic<std::size_t>[]> _outstanding;
  std::unique_ptr<std::atomic<std::size_t>[]> _outstanding_spec;
  std::atomic<std::size_t> _watched{0};
  std::atomic<std::uint64_t> _claimed_releases{0};
  std::atomic<std::uint64_t> _claimed_bytes{0};

  mutable std::mutex _mu;
  std::unordered_map<std::uint64_t, entry> _entries;
};

}  // namespace sirius::memory
