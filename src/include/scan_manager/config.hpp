
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

#include "exec/config.hpp"

namespace sirius::scan_manager {

/**
 * @brief Configuration for the scan_manager.
 *
 * @c use_sirius_datasource controls whether the manager builds a
 * @c sirius_ioctx and routes parquet reads through @c sirius_datasource.
 * Set to @c false to fall back to @c cudf::io::datasource::create() at
 * every read site (e.g. when the sirius IO path is misbehaving).
 */
struct scan_manager_config {
  exec::thread_pool_config thread_pool{.num_threads = 8, .thread_name_prefix = "scan_manager"};
  bool use_sirius_datasource{false};
  /// Reserved (not currently consumed). Intended size of the @c uring_reactor
  /// pool, but the production @c uring_ioctx is built by @c SiriusContext, which
  /// scales the reactor count with the number of GPUs the NUMA node serves
  /// (@c clamp(4 * devices, 4, 16)) rather than reading this field. Parsed from
  /// YAML and kept for forward compatibility / tests; setting it has no effect
  /// on the engine today.
  bool use_odirect{true};
  std::size_t uring_n_reactors{4};
  /// io_uring submission/completion queue depth per reactor.  Ignored when
  /// @c use_sirius_datasource is false.
  unsigned uring_ring_entries{64};
  /// Enable the prefetching cache.  Requires @c use_sirius_datasource=true;
  /// when true, SiriusContext (S6) allocates a pinned-host buffer_pool and
  /// initializes the cache on the IO backends it owns (the per-NUMA urings and
  /// the s3_ioctx).  Off by default.
  bool enable_prefetch_cache{false};
  /// Total pinned-host bytes reserved for the prefetch cache.  Rounded
  /// up to the nearest 500 MiB slab.  Ignored when
  /// @c enable_prefetch_cache is false.
  std::size_t prefetch_buffer_pool_bytes{20ULL << 30};
  /// Maximum chunks the cache may have in flight at once (admission
  /// control).  Ignored when @c enable_prefetch_cache is false.
  std::size_t prefetch_inflight_budget_chunks{2048};

  /// When true (default — current behavior), parquet_split_provider prewarms
  /// per-row-group column-chunk byte ranges via @c cache->insert(obj,
  /// metadata, ranges).  When false, prewarm is skipped: insert is called
  /// with empty ranges (metadata-only, as in §24 describe_parquet).  Lets
  /// the B1 micro-bench A/B compare prefetch overlap on SF10.  Ignored when
  /// @c enable_prefetch_cache is false (no cache → no prewarm regardless).
  bool enable_chunk_prewarm{true};
};

}  // namespace sirius::scan_manager
