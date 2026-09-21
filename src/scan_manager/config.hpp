
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
#include "io/object_store_config.hpp"
#include "io/rest/config.hpp"
#include "io/uring/config.hpp"

#include <algorithm>
#include <thread>

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
 * @c use_sirius_datasource selects the backend for local paths: @c uring_ioctx
 * when true, @c kvikio_context when false. Reads go through
 * @c sirius_datasource either way; the kvikio backend delegates to
 * @c cudf::io::datasource::create(). Multi-GPU forces this to true.
 *
 * Sub-configs:
 *  - @c local   — uring reactor tunables (local-disk IO path).
 *  - @c rest    — REST reactor tunables (S3/object-store IO path).
 *  - @c cache   — prefetching cache tunables.
 *  - @c object_store — object-store credentials and endpoint.
 */
struct scan_manager_config {
  exec::thread_pool_config thread_pool{.num_threads        = default_scan_manager_num_threads(),
                                       .thread_name_prefix = "scan_manager"};
  bool use_sirius_datasource{true};

  /// Number of uring reactor worker threads for the local-disk IO path.
  std::size_t uring_n_reactors{default_uring_n_reactors};

  /// Number of REST reactor worker threads for the S3/object-store IO path
  /// (each its own libcurl event loop + connection pool).
  std::size_t rest_n_reactors{2};

  /// Enable the prefetching cache on the ioctx.  When false the cache is
  /// constructed but unarmed (no background IO threads).
  bool enable_prefetch_cache{false};

  /// Local (uring) reactor configuration — bounce-slot size, O_DIRECT,
  /// ring depth, etc.
  io::uring::config local{};

  /// REST (S3/object-store) reactor configuration — timeouts, TLS, chunking,
  /// retry policy, etc.
  io::rest::config rest{};

  /// Prefetching cache configuration — in-flight budget, pool sizing,
  /// dispose-after-use policy.
  io::cache::config cache{};

  /// Object-store credentials and endpoint consumed by the REST reactor.
  /// Empty fields disable the S3/REST backend.
  io::object_store_config object_store{};

  /// Background host->GPU memory prefetcher for queued pinned-cache scan
  /// splits. Disabled by default.
  memory_prefetcher_config memory_prefetcher{};
};

}  // namespace sirius::scan_manager
