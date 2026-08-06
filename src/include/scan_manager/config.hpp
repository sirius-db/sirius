
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

/// Read-path caching strategy. Single knob that derives @c local.use_odirect,
/// @c enable_prefetch_cache and @c prefetch_cache.dispose_on_idle.
enum class cache_mode {
  /// O_DIRECT reads, no cache anywhere.
  none,
  /// Buffered reads through the OS page cache, no prefetching cache.
  os,
  /// O_DIRECT reads into the pinned prefetching cache, chunks retained for reuse.
  persistent,
  /// O_DIRECT reads into the pinned prefetching cache, chunks dropped once idle.
  prefetch,
};

/// Parse a @ref cache_mode from its lowercase YAML spelling.
inline bool string_to_enum(std::string_view sv, cache_mode& out)
{
  static const std::unordered_map<std::string_view, cache_mode> map = {
    {"none", cache_mode::none},
    {"os", cache_mode::os},
    {"persistent", cache_mode::persistent},
    {"prefetch", cache_mode::prefetch},
  };
  auto it = map.find(sv);
  if (it != map.end()) {
    out = it->second;
    return true;
  }
  return false;
}

/// Render a @ref cache_mode as its canonical lowercase name.
inline bool enum_to_string(cache_mode mode, std::string& s)
{
  switch (mode) {
    case cache_mode::none: s = "none"; return true;
    case cache_mode::os: s = "os"; return true;
    case cache_mode::persistent: s = "persistent"; return true;
    case cache_mode::prefetch: s = "prefetch"; return true;
  }
  return false;
}

/**
 * @brief Configuration for the scan_manager.
 *
 * @c use_sirius_datasource selects the backend for local paths: @c uring_ioctx
 * when true, @c kvikio_context when false. Reads go through
 * @c sirius_datasource either way; the kvikio backend drives
 * @c kvikio::FileHandle directly. Multi-GPU forces this to true.
 *
 * @c cache picks the read-path caching strategy; @ref apply_cache_mode derives
 * @c local.use_odirect, @c enable_prefetch_cache and
 * @c prefetch_cache.dispose_on_idle from it, so those three are not settable
 * on their own.
 *
 * Sub-configs:
 *  - @c local   — uring reactor tunables (local-disk IO path).
 *  - @c rest    — REST reactor tunables (S3/object-store IO path).
 *  - @c kvikio  — kvikIO backend tunables (local-file fallback path).
 *  - @c prefetch_cache — prefetching cache tunables.
 *  - @c object_store — object-store credentials and endpoint.
 */
struct scan_manager_config {
  exec::thread_pool_config thread_pool{.num_threads        = default_scan_manager_num_threads(),
                                       .thread_name_prefix = "scan_manager"};
  bool use_sirius_datasource{true};

  /// Read-path caching strategy; the source of truth for the derived knobs.
  cache_mode cache{cache_mode::none};

  /// Number of uring reactor worker threads for the local-disk IO path.
  std::size_t uring_n_reactors{default_uring_n_reactors};

  /// Number of REST reactor worker threads for the S3/object-store IO path
  /// (each its own libcurl event loop + connection pool).
  std::size_t rest_n_reactors{2};

  /// Enable the prefetching cache on the ioctx.  When false the cache is
  /// constructed but unarmed (no background IO threads).  Derived from @ref cache.
  bool enable_prefetch_cache{false};

  /// Local (uring) reactor configuration — bounce-slot size, O_DIRECT,
  /// ring depth, etc.  @c use_odirect is derived from @ref cache.
  io::uring::config local{};

  /// REST (S3/object-store) reactor configuration — timeouts, TLS, chunking,
  /// retry policy, etc.
  io::rest::config rest{};

  /// kvikIO backend tunables for the local-file fallback path.  Every field is
  /// optional; unset leaves kvikIO's own env-seeded default in place.
  io::kvikio_config kvikio{};

  /// Prefetching cache configuration — in-flight budget, pool sizing,
  /// dispose-on-idle policy.  @c dispose_on_idle is derived from @ref cache.
  io::cache::config prefetch_cache{};

  /// Object-store credentials and endpoint consumed by the REST reactor.
  /// Empty fields disable the S3/REST backend.
  io::object_store_config object_store{};

  /// Overwrite the knobs derived from @ref cache.
  void apply_cache_mode() noexcept
  {
    switch (cache) {
      case cache_mode::none:
        local.use_odirect     = true;
        enable_prefetch_cache = false;
        break;
      case cache_mode::os:
        local.use_odirect     = false;
        enable_prefetch_cache = false;
        break;
      case cache_mode::persistent:
        local.use_odirect              = true;
        enable_prefetch_cache          = true;
        prefetch_cache.dispose_on_idle = false;
        break;
      case cache_mode::prefetch:
        local.use_odirect              = true;
        enable_prefetch_cache          = true;
        prefetch_cache.dispose_on_idle = true;
        break;
    }
  }
};

}  // namespace sirius::scan_manager
