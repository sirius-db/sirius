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

#include "memory/pinned_reservation_guard.hpp"

#include "compression/compressed_representation.hpp"
#include "log/logging.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <cudf/column/column.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <chrono>
#include <condition_variable>
#include <exception>
#include <map>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace sirius {
namespace memory {

//===----------------------------------------------------------------------===//
// Unevictable-pinned-bytes provider hook
//===----------------------------------------------------------------------===//

namespace {

struct provider_registry {
  std::mutex mutex;
  std::vector<std::pair<const void*, unevictable_bytes_provider>> providers;

  static provider_registry& instance()
  {
    static provider_registry registry;
    return registry;
  }
};

}  // namespace

void register_unevictable_bytes_provider(const void* owner, unevictable_bytes_provider provider)
{
  auto& registry = provider_registry::instance();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto& [existing_owner, fn] : registry.providers) {
    if (existing_owner == owner) {
      fn = std::move(provider);
      return;
    }
  }
  registry.providers.emplace_back(owner, std::move(provider));
}

void unregister_unevictable_bytes_provider(const void* owner) noexcept
{
  auto& registry = provider_registry::instance();
  std::lock_guard<std::mutex> lock(registry.mutex);
  std::erase_if(registry.providers, [owner](auto const& entry) { return entry.first == owner; });
}

std::size_t unevictable_pinned_bytes(const cucascade::memory::memory_space* space) noexcept
{
  auto& registry = provider_registry::instance();
  std::lock_guard<std::mutex> lock(registry.mutex);
  std::size_t total = 0;
  for (auto const& [owner, fn] : registry.providers) {
    if (!fn) { continue; }
    try {
      total += fn(space);
    } catch (const std::exception& e) {
      // A throwing provider must not take down the executor's manager loop —
      // contribute 0 (the conservative direction: the fail-fast never fires
      // spuriously on missing pin accounting).
      SIRIUS_LOG_WARN("[pinned_reservation_guard] unevictable-bytes provider threw: {}", e.what());
    } catch (...) {
      SIRIUS_LOG_WARN("[pinned_reservation_guard] unevictable-bytes provider threw");
    }
  }
  return total;
}

//===----------------------------------------------------------------------===//
// Scan-manager walk
//===----------------------------------------------------------------------===//

std::size_t gpu_tier_pinned_bytes(const sirius::scan_manager::sirius_scan_manager& mgr,
                                  const cucascade::memory::memory_space* space)
{
  std::size_t total = 0;
  mgr.visit_pinned_entries(
    [&](std::string_view /*name*/, const sirius::scan_manager::pinned_entry& entry) {
      if (entry.tier != cucascade::memory::Tier::GPU) { return true; }
      if (!entry.device_chunks.empty()) {
        // Compression-enabled GPU pin: per chunk either a compressed device
        // representation or a set of uncompressed device columns (a single pin
        // may interleave the two). Same dispatch priority as the cached
        // provider: device_chunks wins over data_batches_by_column.
        for (auto const& chunk : entry.device_chunks) {
          if (chunk.memory_space != space) { continue; }
          if (chunk.compressed) {
            total += chunk.compressed->get_size_in_bytes();
          } else {
            for (auto const& col : chunk.columns) {
              if (col) { total += col->alloc_size(); }
            }
          }
        }
      } else {
        // Plain GPU pin: per-column chunk vectors, placement parallel via
        // chunk_memory_spaces (chunk i of every column shares one space).
        for (auto const& [col_name, chunks] : entry.data_batches_by_column) {
          for (std::size_t i = 0; i < chunks.size(); ++i) {
            if (i < entry.chunk_memory_spaces.size() && entry.chunk_memory_spaces[i] == space &&
                chunks[i]) {
              total += chunks[i]->alloc_size();
            }
          }
        }
      }
      return true;
    });
  return total;
}

//===----------------------------------------------------------------------===//
// Reservation-wait watchdog
//===----------------------------------------------------------------------===//

namespace {

constexpr std::chrono::seconds k_report_interval{10};

class wait_watchdog {
 public:
  static wait_watchdog& instance()
  {
    static wait_watchdog watchdog;
    return watchdog;
  }

  std::uint64_t add(const cucascade::memory::memory_space* space,
                    std::size_t requested_bytes,
                    std::uint64_t pipeline_id,
                    std::uint64_t task_id)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_thread.joinable()) {
      // Lazily created on the first would-block wait ever; parks on _cv
      // whenever _entries is empty, so it never polls on the happy path.
      _thread = std::thread(&wait_watchdog::run, this);
    }
    auto const id  = _next_id++;
    auto const now = std::chrono::steady_clock::now();
    _entries.emplace(
      id, entry{space, requested_bytes, pipeline_id, task_id, /*start=*/now, /*last_report=*/now});
    _cv.notify_all();
    return id;
  }

  void remove(std::uint64_t id)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _entries.erase(id);
    _cv.notify_all();
  }

  ~wait_watchdog()
  {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _stop = true;
    }
    _cv.notify_all();
    if (_thread.joinable()) { _thread.join(); }
  }

 private:
  struct entry {
    const cucascade::memory::memory_space* space;
    std::size_t requested_bytes;
    std::uint64_t pipeline_id;
    std::uint64_t task_id;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point last_report;
  };

  void run()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    while (!_stop) {
      if (_entries.empty()) {
        _cv.wait(lock, [this] { return _stop || !_entries.empty(); });
        continue;
      }
      // Wake at the earliest per-entry report deadline (absolute time, so
      // add/remove churn cannot indefinitely postpone a report).
      auto next_deadline = std::chrono::steady_clock::time_point::max();
      for (auto const& [id, e] : _entries) {
        next_deadline = std::min(next_deadline, e.last_report + k_report_interval);
      }
      _cv.wait_until(lock, next_deadline);
      if (_stop) { break; }
      auto const now = std::chrono::steady_clock::now();
      for (auto& [id, e] : _entries) {
        if (now - e.last_report < k_report_interval) { continue; }
        e.last_report = now;
        report(e, now);
      }
    }
  }

  // Called with _mutex held; only reads the space's atomic counters and the
  // provider registry (whose mutex is strictly nested inside ours).
  static void report(const entry& e, std::chrono::steady_clock::time_point now)
  {
    auto const elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(now - e.start).count();
    std::size_t const available = e.space->get_available_memory();
    std::size_t const reserved  = e.space->get_total_reserved_memory();
    std::size_t const limit     = e.space->get_max_memory();
    std::size_t const pinned    = unevictable_pinned_bytes(e.space);
    SIRIUS_LOG_INFO(
      "[reservation-wait] GPU {}: pipeline {} task {} has been waiting {} s for a {} byte "
      "reservation; available {} bytes, active reservations {} bytes, reservation limit {} "
      "bytes, unevictable gpu-tier pinned {} bytes (max satisfiable {} bytes)",
      e.space->get_device_id(),
      e.pipeline_id,
      e.task_id,
      elapsed_s,
      e.requested_bytes,
      available,
      reserved,
      limit,
      pinned,
      max_satisfiable_reservation(limit, pinned));
  }

  std::mutex _mutex;
  std::condition_variable _cv;
  std::map<std::uint64_t, entry> _entries;
  std::uint64_t _next_id{1};
  bool _stop{false};
  std::thread _thread;
};

}  // namespace

reservation_wait_scope::reservation_wait_scope(const cucascade::memory::memory_space* space,
                                               std::size_t requested_bytes,
                                               std::uint64_t pipeline_id,
                                               std::uint64_t task_id)
  : _id(wait_watchdog::instance().add(space, requested_bytes, pipeline_id, task_id))
{
}

reservation_wait_scope::~reservation_wait_scope() { wait_watchdog::instance().remove(_id); }

}  // namespace memory
}  // namespace sirius
