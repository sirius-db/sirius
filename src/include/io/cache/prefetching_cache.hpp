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

#include "blockingconcurrentqueue.h"
#include "exec/admission_control.hpp"
#include "exec/scoped_dispatcher.hpp"
#include "exec/semi_future.hpp"
#include "exec/thread_pool.hpp"
#include "io/cache/types.hpp"
#include "planner/query.hpp"

#include <concurrentqueue.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::io {
class sirius_ioctx;
class sirius_datasource;
}  // namespace sirius::io

namespace sirius::memory {
class topology_index;
}  // namespace sirius::memory

namespace cucascade::memory {
class memory_reservation_manager;
}  // namespace cucascade::memory

namespace sirius::io::cache {

enum class prefetching_handle_state { idle, active, cancelled };

struct prefetch_request_context {
  explicit prefetch_request_context(const sirius_io_object& file, std::uint32_t ts) noexcept
    : timestamp(ts),
      obj(file.shared_from_this()),
      state(std::make_shared<entry_state>()),
      user_state(
        std::make_shared<std::atomic<prefetching_handle_state>>(prefetching_handle_state::idle))
  {
  }

  [[nodiscard]] bool is_active() const noexcept
  {
    return user_state &&
           user_state->load(std::memory_order_acquire) == prefetching_handle_state::active;
  }

  [[nodiscard]] bool is_cancelled() const noexcept
  {
    return !user_state ||
           user_state->load(std::memory_order_acquire) == prefetching_handle_state::cancelled;
  }

  const std::uint32_t timestamp;
  const std::shared_ptr<const sirius_io_object> obj;
  std::shared_ptr<entry_state> state;
  std::shared_ptr<std::atomic<prefetching_handle_state>> user_state;
  std::vector<cached_chunk*> chunks;
  /// Preferred NUMA node for the staging buffers, derived from the requesting
  /// GPU's topology.  -1 means "no preference" (allocate from any arena).
  int preferred_numa{-1};
};

class prefetching_handle {
 public:
  prefetching_handle() noexcept;
  ~prefetching_handle();
  prefetching_handle(prefetching_handle const&)            = delete;
  prefetching_handle& operator=(prefetching_handle const&) = delete;

  prefetching_handle(prefetching_handle&& o) noexcept;
  prefetching_handle& operator=(prefetching_handle&& o) noexcept;

  void activate() noexcept;

  void cancel() noexcept;

  [[nodiscard]] bool is_active() const noexcept;

  [[nodiscard]] std::shared_ptr<prefetch_request_context> get_context() const noexcept;

  explicit operator bool() const noexcept;

 private:
  friend class prefetching_cache;
  struct prefetch_lifecycle_manager;

  prefetching_handle(std::unique_ptr<prefetch_lifecycle_manager> mgr) noexcept;

  std::unique_ptr<prefetch_lifecycle_manager> _state;
};

// ---------------------------------------------------------------------------
// prefetching_cache
// ---------------------------------------------------------------------------
//
// Locking hierarchy:
//   Level 0: _map_mtx          — protects _file_cache map
//   Level 1: file_entry::mtx   — protects one file's entry vector
//   (independent): cache_entry atomics — lock-free

class prefetching_cache {
  // The cache only accepts new prefetch requests through
  // sirius_datasource::fadvise — that's the single entry point for the
  // fadvise(speculative/immediate/disposable) protocol.  Friending the
  // datasource keeps insert() out of the public API while still letting
  // fadvise dispatch through it.
  friend class sirius::io::sirius_datasource;
  // prefetching_handle calls notify_disposed() on cancel — needs access to
  // the private method.
  friend class prefetching_handle;

 public:
  using byte_range         = cudf::io::text::byte_range_info;
  using prefetch_request   = std::shared_ptr<prefetch_request_context>;
  using request_queue_type = duckdb_moodycamel::BlockingConcurrentQueue<prefetch_request>;

  prefetching_cache(cucascade::memory::memory_reservation_manager& reservation_manager,
                    sirius_ioctx* io_ctx,
                    size_t inflight_budget_chunks,
                    uint32_t buffer_pool_slabs,
                    std::shared_ptr<const sirius::memory::topology_index> topology_index,
                    bool dispose_after_use = false);
  ~prefetching_cache();

  prefetching_cache(prefetching_cache const&)            = delete;
  prefetching_cache& operator=(prefetching_cache const&) = delete;

  [[nodiscard]] bool is_armed() const noexcept { return _armed; }

  [[nodiscard]] std::size_t host_read(const sirius_io_object& obj,
                                      size_t offset,
                                      size_t size,
                                      uint8_t* dst,
                                      prefetching_handle* out_handle = nullptr);

  [[nodiscard]] exec::semi_future<std::size_t> host_read_async(
    const sirius_io_object& obj,
    size_t offset,
    size_t size,
    uint8_t* dst,
    prefetching_handle* out_handle = nullptr);

  [[nodiscard]] exec::semi_future<std::size_t> device_read_async(
    const sirius_io_object& obj,
    size_t offset,
    size_t size,
    uint8_t* device_ptr,
    rmm::cuda_stream_view stream,
    prefetching_handle* out_handle = nullptr);

  [[nodiscard]] std::string summary() const;

  void prepare_for_query(const sirius::planner::query& query) noexcept;

 private:
  [[nodiscard]] prefetching_handle insert(const sirius_io_object& obj,
                                          std::span<const byte_range> ranges,
                                          std::optional<int> gpu_id = {});

  [[nodiscard]] bool host_read_from_cache_only(const sirius_io_object& obj,
                                               size_t offset,
                                               size_t size,
                                               uint8_t* dst,
                                               prefetching_handle* out_handle);

  struct file_entry {
    std::vector<cached_chunk*> update_and_get_chunks(std::span<size_t> incoming, uint32_t ticker);

    std::vector<cached_chunk*> fetch_chunks(std::size_t offset,
                                            std::size_t size,
                                            coverage_policy policy) const;

    mutable std::shared_mutex mtx;
    std::shared_ptr<const sirius_io_object> io_obj;
    std::vector<std::unique_ptr<cached_chunk>> chunks;
    size_t file_size{0};
  };

  void prepare_loop(const std::stop_token& st);
  void prefetch_loop(const std::stop_token& st);
  void evict_loop(const std::stop_token& st);

  file_entry& get_or_create_file_entry(const sirius_io_object& obj);

  std::unique_ptr<buffer_pool> _pool;

  sirius_ioctx* const _io_ctx;

  // Hardware GPU/NUMA topology index, shared from the scan_manager.  Used to
  // place prefetch staging buffers on the NUMA node closest to the target GPU.
  std::shared_ptr<const sirius::memory::topology_index> const _topology_index;

  bool const _armed;

  std::atomic<bool> _shutting_down{false};

  std::atomic<uint32_t> _ticker{0};  // see prefetch_stats::snapshot for layout

  // ---- Telemetry counters --------------------------------------------------
  // Global cumulative counts.  @c _last_reported snapshots them on every cache
  // refresh (prepare_for_query) so summary() can also report per-cycle deltas.
  struct counters {
    std::atomic<uint64_t> n_reads{0};    // device read requests served by the cache
    std::atomic<uint64_t> hits{0};       // chunks served from cache (read pin acquired, in_use)
    std::atomic<uint64_t> h2d{0};        // chunks (re)loaded via host->device IO (mark_loading)
    std::atomic<uint64_t> misses{0};     // chunks read fresh (missing / not yet usable)
    std::atomic<uint64_t> evictions{0};  // chunks evicted back to the pool
  };
  struct counters_snapshot {
    uint64_t n_reads{0};
    uint64_t hits{0};
    uint64_t h2d{0};
    uint64_t misses{0};
    uint64_t evictions{0};
  };

  counters _counters;
  counters_snapshot _last_reported;

  std::jthread _preparation_thread;
  request_queue_type _preparation_queue;
  std::stop_source _preparation_stop_source;

  std::jthread _prefetch_thread;
  exec::admission_control _rate_limiter;
  request_queue_type _prefetch_queue;
  std::stop_source _prefetch_stop_source;

  std::jthread _evictor_thread;
  request_queue_type _eviction_queue;
  std::stop_source _evictor_stop_source;
  bool _dispose_after_use{false};

  mutable std::shared_mutex _map_mtx;
  std::unordered_map<std::string, std::unique_ptr<file_entry>> _file_cache;

  exec::static_thread_pool _io_cb_thread_pool{
    2, "io_cb"};  // single-threaded pool for IO completion callbacks
  exec::scoped_dispatcher _io_cb_dispatcher{_io_cb_thread_pool, 2};
};

}  // namespace sirius::io::cache
