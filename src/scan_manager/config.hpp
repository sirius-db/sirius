
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

#include "creator/config.hpp"
#include "exec/config.hpp"
#include "io/cache/config.hpp"
#include "io/kvikio/config.hpp"
#include "io/object_store_config.hpp"
#include "io/rest/config.hpp"
#include "io/uring/config.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

namespace sirius::scan_manager {

/// Default uring reactor count; counted in the scan-manager sizing budget below.
inline constexpr std::size_t default_uring_n_reactors = 1;

/// Default scan-manager pool size: every core left after the other default pools
/// (downgrade, task_creator, pipeline, uring reactor), never below 4.
[[nodiscard]] inline int default_scan_manager_num_threads()
{
  constexpr int reserved =
    exec::default_downgrade_num_threads + creator::default_task_creator_num_threads +
    exec::default_gpu_pipeline_num_threads + static_cast<int>(default_uring_n_reactors);
  return std::max(4, static_cast<int>(std::thread::hardware_concurrency()) - reserved);
}

/// How the readahead decides *when* to issue the next prefetch.
///
/// The two differ in what they assume the backend costs.  A round trip to an
/// object store is dead time no matter what else is running, so the deeper the
/// queue the better; a local NVMe read competes with the executor's own reads
/// for the same device, so issuing one while a scan is running just reorders
/// the queue rather than adding throughput.
enum class prefetch_strategy {
  /// Keep the backend's scan budget occupied at all times: every wake-up fills
  /// every free slot.  What an object store wants -- latency is hidden only by
  /// having more requests outstanding.
  eager,
  /// Issue one prefetch each time the executor deploys a task that is NOT a
  /// scan.  A non-scan task means a pipeline thread just went to compute rather
  /// than to read, so the device is idle and this prefetch costs nothing that
  /// the executor wanted.  What a local file wants.
  opportunistic,
};

/// Parse a @ref prefetch_strategy from its lowercase YAML spelling.
inline bool string_to_enum(std::string_view sv, prefetch_strategy& out)
{
  static const std::unordered_map<std::string_view, prefetch_strategy> map = {
    {"eager", prefetch_strategy::eager},
    {"opportunistic", prefetch_strategy::opportunistic},
  };
  auto it = map.find(sv);
  if (it == map.end()) { return false; }
  out = it->second;
  return true;
}

/// Render a @ref prefetch_strategy as its canonical lowercase name.
inline bool enum_to_string(prefetch_strategy s, std::string& out)
{
  switch (s) {
    case prefetch_strategy::eager: out = "eager"; return true;
    case prefetch_strategy::opportunistic: out = "opportunistic"; return true;
  }
  return false;
}

/// IO backend that serves managed reads.
enum class io_backend {
  /// Sirius's own IO stack: uring for local paths, REST for @c s3:// URLs.
  sirius,
  /// The kvikIO backend (drives @c kvikio::FileHandle directly).
  kvikio,
};

/// Parse a @ref io_backend from its lowercase YAML spelling.
inline bool string_to_enum(std::string_view sv, io_backend& out)
{
  static const std::unordered_map<std::string_view, io_backend> map = {
    {"sirius", io_backend::sirius},
    {"kvikio", io_backend::kvikio},
  };
  auto it = map.find(sv);
  if (it != map.end()) {
    out = it->second;
    return true;
  }
  return false;
}

/// Render a @ref io_backend as its canonical lowercase name.
inline bool enum_to_string(io_backend b, std::string& s)
{
  switch (b) {
    case io_backend::sirius: s = "sirius"; return true;
    case io_backend::kvikio: s = "kvikio"; return true;
  }
  return false;
}

/// What the readahead scan manager is started with, resolved from the config
/// and the backend serving the scans.  A @c budget of zero means it does not run.
struct readahead_plan {
  std::size_t budget{0};
  prefetch_strategy strategy{prefetch_strategy::eager};
};

/**
 * @brief Configuration for the background host->GPU memory prefetcher
 *        (see scan_manager/memory_prefetcher.hpp).
 *
 * Set via the yaml block sirius.executor.scan_manager.memory_prefetcher.
 * Disabled by default; single-GPU configurations only (the prefetcher logs a
 * warning and disables itself when more than one GPU space is configured).
 */
struct memory_prefetcher_config {
  /// Master switch; when false the prefetcher is never constructed.
  bool enable{false};
  /// Number of prefetch worker threads. Each drives one in-flight batch
  /// conversion on its own stream, so this bounds conversion concurrency.
  std::size_t num_threads{2};
  /// Keep at least this fraction of the GPU space free after each prefetch;
  /// the reservation for a batch is only attempted above this floor, so the
  /// prefetcher backs off well before competing with pipeline reservations.
  double min_free_fraction{0.4};
  /// Worker sweep interval while waiting for headroom / new splits.
  std::size_t poll_interval_ms{2};
  /// A connector is considered actively draining (and skipped) until this
  /// long has passed since its last pop. Must exceed the scan's inter-pop
  /// interval (~10-40ms per 5GB batch) or sweeps race the scan.
  std::size_t drain_quiet_ms{100};
};

/**
 * @brief Configuration for the scan_manager.
 *
 * @c backend selects the IO stack: @ref io_backend::sirius routes local paths
 * to @c uring_ioctx and @c s3:// URLs to the REST backend, @ref io_backend::kvikio
 * routes local paths to @c kvikio_context. Reads go through
 * @c sirius_datasource either way; the kvikio backend drives
 * @c kvikio::FileHandle directly. Multi-GPU forces @ref io_backend::sirius.
 *
 * @c cache is the read path's whole caching configuration, and its only home;
 * @ref apply_cache_mode derives @c uring.use_odirect and
 * @c cache.dispose_on_idle from it, so neither is settable on its own.
 *
 * Sub-configs:
 *  - @c uring   — uring reactor tunables (local-disk IO path).
 *  - @c rest    — REST reactor tunables (S3/object-store IO path).
 *  - @c kvikio  — kvikIO backend tunables (local-file fallback path).
 *  - @c cache   — caching mode, eviction policy and prefetching-cache tunables.
 *  - @c object_store — object-store credentials and endpoint.
 */
struct scan_manager_config {
  exec::thread_pool_config thread_pool{.num_threads        = default_scan_manager_num_threads(),
                                       .thread_name_prefix = "scan_manager"};
  /// IO backend that serves managed reads.
  io_backend backend{io_backend::sirius};

  /// Number of uring reactor worker threads for the local-disk IO path.
  std::size_t uring_n_reactors{default_uring_n_reactors};

  /// Number of REST reactor worker threads for the S3/object-store IO path
  /// (each its own libcurl event loop + connection pool).
  std::size_t rest_n_reactors{2};

  /// Scans the readahead scan manager may keep in flight, and the switch that
  /// runs it at all.  Unset (the default) defers to the caching configuration:
  /// with a cache on, the budget is the backend reactor's own
  /// @c n_max_concurrent_scans (see @ref resolve_readahead); with
  /// @c cache.mode of @c none there is nothing to read ahead into and the
  /// readahead does not run.  Set it to @c 0 to turn the readahead off even
  /// with a cache on, or to a positive count to override the backend's budget.
  std::optional<std::size_t> max_readahead_scans{};

  /// When the readahead issues.  Unset (the default) takes the serving
  /// backend's own preference — @c eager for an object store, @c opportunistic
  /// for a local device (see @ref prefetch_strategy).  Set it to pin one
  /// strategy whatever the backend.
  std::optional<prefetch_strategy> readahead_strategy{};

  /// Scans the executor can have running at once — the pipeline pool's width.
  /// The budget @c opportunistic schedules against, since one prefetch per
  /// non-scan deployment is only useful while the executor could still take
  /// another scan.  Stamped by @c sirius_config from the pipeline config; zero
  /// means "not stamped", and the backend's own budget is used instead.
  std::size_t pipeline_width{0};

  /// Local (uring) reactor configuration. @c use_odirect is derived from
  /// @ref cache; physical operation size is selected by the worker.
  io::uring::config uring{};

  /// REST (S3/object-store) reactor configuration — timeouts, TLS, logical
  /// merge hints, retry policy, and connection limits. Physical GET sizing is
  /// worker-owned.
  io::rest::config rest{};

  /// kvikIO backend tunables for the local-file fallback path.  Every field is
  /// optional; unset leaves kvikIO's own env-seeded default in place.
  io::kvikio_config kvikio{};

  /// The read path's caching configuration: mode, eviction policy and the
  /// prefetching cache's tunables.  One block, rather than a mode here and the
  /// tunables in a sibling one.
  io::cache::config cache{};

  /// Object-store credentials and endpoint consumed by the REST reactor.
  /// Empty fields disable the S3/REST backend.
  io::object_store_config object_store{};

  /// Background host->GPU memory prefetcher for queued pinned-cache scan
  /// splits. Disabled by default.
  memory_prefetcher_config memory_prefetcher{};

  /// Refresh the knobs derived from @ref cache.
  void apply_cache_mode() noexcept
  {
    cache.apply_mode();
    uring.use_odirect = cache.use_odirect();
  }

  /// Resolve what the readahead should run with, or a budget of 0 to not run
  /// it at all.  @p backend_budget and @p backend_strategy come from the
  /// backend serving the scans — its @c n_max_concurrent_scans and the
  /// preference implied by its reactor type — and are what an unset
  /// @ref max_readahead_scans / @ref readahead_strategy defers to.
  [[nodiscard]] readahead_plan resolve_readahead(std::size_t backend_budget,
                                                 prefetch_strategy backend_strategy) const noexcept
  {
    readahead_plan plan{.budget = 0, .strategy = readahead_strategy.value_or(backend_strategy)};
    // An explicit budget is the whole answer, zero (off) included.
    if (max_readahead_scans.has_value()) {
      plan.budget = *max_readahead_scans;
      return plan;
    }
    // Without a cache the readahead has nowhere to read into: a prefetched
    // chunk would be dropped before its scan ever asked for it.
    if (!cache.enabled()) { return plan; }
    // Opportunistic schedules against what the executor can run, not what the
    // device can queue: one prefetch per non-scan deployment is only useful
    // while a pipeline thread could still pick up another scan.
    plan.budget = plan.strategy == prefetch_strategy::opportunistic && pipeline_width > 0
                    ? pipeline_width
                    : backend_budget;
    return plan;
  }
};

}  // namespace sirius::scan_manager
