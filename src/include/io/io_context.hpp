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

#include "exec/semi_future.hpp"
#include "io/cache/metadata_store.hpp"
#include "io/cache/types.hpp"
#include "io/types.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::io {

enum class io_context_type { uring, restful, kvikio };

namespace cache {
class prefetching_cache;
}

class sirius_datasource;

}  // namespace sirius::io

namespace sirius::memory {
class topology_index;
}  // namespace sirius::memory

namespace cucascade::memory {
class memory_reservation_manager;
}  // namespace cucascade::memory

namespace sirius::io {

// ---------------------------------------------------------------------------
// prefetching_stage
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// sirius_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief Abstract shared context passed to every datasource.
 *
 * Holds resources that are shared across all datasources (cache, reactor
 * threads, ...). Extend this class to provide a concrete I/O backend.
 */
class sirius_ioctx : public std::enable_shared_from_this<sirius_ioctx> {
  // prefetching_cache's worker_loop is the only caller of the protected
  // host_read_ranges_async_io entry point through a sirius_ioctx* base
  // pointer.  Friending the cache lets that single call site reach in
  // without forcing the vector-read primitive (which most callers never
  // touch) into the public API.
  friend class cache::prefetching_cache;

 public:
  sirius_ioctx();
  virtual ~sirius_ioctx();

  [[nodiscard]] virtual io_context_type type() const noexcept = 0;

  /// Start the backend's reactors: launch their worker threads and allocate
  /// per-reactor staging.  Deferred from construction so an ioctx can be built
  /// and parked (e.g. in a per-query map of contexts) without spending thread
  /// or pinned-memory resources until it is first used.  Must be called before
  /// the read API is exercised.  Idempotent.  Backends with no reactors
  /// (kvikio, blocking) inherit the default no-op.
  virtual void start() {}

  virtual void shutdown() noexcept = 0;

  /// Open a datasource for @p path.  The backend creates the underlying
  /// io_object internally (however is appropriate for the scheme — opening
  /// local fds, issuing a HEAD for object stores, ...) and wraps it in a
  /// @c sirius_datasource bound to this ioctx.  Throws on unsupported /
  /// unreachable paths (callers that want a check-without-open should use
  /// @c supports()).
  [[nodiscard]] std::unique_ptr<sirius_datasource> open_datasource(std::string path);

  /// Whether this backend can serve reads for @p path.  Backends should
  /// validate scheme/protocol support and any backend-specific
  /// preconditions (e.g. file existence for local-disk backends).
  [[nodiscard]] virtual bool supports(std::string_view path) const noexcept = 0;

  // -- Backend capabilities ---------------------------------------------------

  /// Whether the backend can stream data directly into device memory
  /// (e.g. via O_DIRECT + GDS).  Used by @c sirius_datasource to answer the
  /// equivalent cudf::io::datasource queries.
  [[nodiscard]] virtual bool supports_device_read() const noexcept = 0;

  [[nodiscard]] virtual bool supports_host_to_device_read() const noexcept = 0;

  /// Whether the backend can serve a batch of host reads in a single dispatch
  /// (cf. @c host_read_ranges_async_io).  When false, the prefetching layer
  /// cannot amortise per-request overhead and must fall back to
  /// @c prefetching_stage::none.
  [[nodiscard]] virtual bool supports_vector_host_read() const noexcept = 0;

  /// Prefetching strategy the prefetching layer should use against this
  /// backend.  Returns @c prefetching_stage::none whenever
  /// @c supports_vector_host_read() is false; otherwise the backend picks
  /// between eager prefill and on-demand read-ahead based on its IO
  /// characteristics.
  [[nodiscard]] virtual cache::prefetching_stage preferred_prefetching_stage() const noexcept = 0;

  /// Build the prefetching cache.  One-shot — calling twice is a no-op
  /// after the first successful build.  The cache holds a raw
  /// back-pointer to this ioctx and stays alive until @ref
  /// shutdown_cache is called (or this ioctx is destroyed).  The cache
  /// builds and owns its @c buffer_pool from @p reservation_manager's
  /// HOST-tier memory spaces; @p buffer_pool_slabs sizes that pool.
  ///
  /// The cache constructs itself in an "armed" or "unarmed" state
  /// depending on @c supports_vector_host_read(); the ioctx is unaware
  /// of that distinction — it simply forwards lookups through @c cache().
  void initialize_cache(
    cucascade::memory::memory_reservation_manager& reservation_manager,
    size_t inflight_budget_chunks,
    uint32_t buffer_pool_slabs,
    std::shared_ptr<const sirius::memory::topology_index> topology_index) noexcept;

  /// Tear down the cache (drains background workers and any in-flight
  /// IO via @c admission_control).  Idempotent.  The owner (scan
  /// manager) calls this BEFORE releasing the @c buffer_pool the cache
  /// was constructed with — otherwise workers may issue final IO
  /// against a destroyed pool.
  void shutdown_cache() noexcept;

  /// Every concrete derived class MUST call this as the very first
  /// statement in its destructor.  It drains the cache (so its workers
  /// stop issuing IO) while the derived object's reactors / handles
  /// are still alive.  Without this, the cache's defensive shutdown
  /// in @c ~sirius_ioctx would run AFTER the derived part of the
  /// object has been destroyed, and worker callbacks would reach
  /// already-destroyed reactors.
  ///
  /// Idempotent — calling @c shutdown_cache directly before this is
  /// fine.  Cheap when no cache was ever initialised.
  void pre_destroy() noexcept { shutdown_cache(); }

  [[nodiscard]] cache::prefetching_cache* cache() noexcept { return _cache.get(); }

  /// True iff @c host_read / @c device_read should consult the cache
  /// before falling through to the backend.  Computed live so it tracks
  /// @ref initialize_cache / @ref shutdown_cache transitions.
  [[nodiscard]] inline bool uses_prefetching_cache() const noexcept
  {
    return _cache != nullptr && supports_vector_host_read();
  }

  /// Per-file metadata cache that lives independently of the prefetching
  /// cache.  Always available — callers that have parsed file metadata
  /// (e.g. a parquet footer) park it here so a later scan of the same
  /// path can skip the parse without depending on whether the
  /// prefetching machinery has been wired up.
  [[nodiscard]] cache::metadata_store& metadata_store() noexcept { return _metadata_store; }
  [[nodiscard]] cache::metadata_store const& metadata_store() const noexcept
  {
    return _metadata_store;
  }

  // -- Physical range alignment ------------------------------------------------

  /// Align each input range's ends outward to the backend's I/O alignment and
  /// coalesce overlapping/adjacent results into a minimal set of aligned,
  /// non-overlapping ranges (sorted by offset).  @p alignment is a lower bound:
  /// when unset, or smaller than the backend's optimal alignment, the backend
  /// uses its own alignment instead.
  [[nodiscard]] virtual std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<const cudf::io::text::byte_range_info> ranges,
    std::optional<size_t> alignment = std::nullopt) const noexcept = 0;

  virtual size_t host_read_io(const sirius_io_object& obj,
                              size_t offset,
                              size_t size,
                              uint8_t* dst) = 0;

  virtual exec::semi_future<size_t> host_read_async_io(const sirius_io_object& obj,
                                                       size_t offset,
                                                       size_t size,
                                                       uint8_t* dst) noexcept = 0;

  virtual exec::semi_future<size_t> device_read_async_io(const sirius_io_object& obj,
                                                         size_t offset,
                                                         size_t size,
                                                         uint8_t* dst,
                                                         rmm::cuda_stream_view stream) noexcept = 0;

  virtual exec::semi_future<size_t> host_to_device_read_async_io(
    const sirius_io_object& obj,
    std::span<io_object_segment> slices,
    size_t offset,
    size_t size,
    uint8_t* device_dst,
    rmm::cuda_stream_view stream) noexcept = 0;

  virtual exec::semi_future<size_t> host_read_ranges_async_io(
    const sirius_io_object& obj, std::span<io_object_segment> segments) noexcept = 0;

 protected:
  /// Backend hook: open native handles / resolve metadata for @p path and
  /// return a populated io_object.  Invoked by @c open_datasource; not part of
  /// the public surface (callers receive a ready @c sirius_datasource).  Throws
  /// on unsupported / unreachable paths.
  virtual std::shared_ptr<sirius_io_object> create_io_object(std::string path) = 0;

  /// Owned by this ioctx.  Built by @ref initialize_cache, destroyed
  /// by @ref shutdown_cache (or the ioctx destructor as a safety net,
  /// though callers are expected to drive the lifecycle explicitly so
  /// reactors stay alive while workers drain).
  std::unique_ptr<cache::prefetching_cache> _cache;

  /// Independent of the prefetching machinery — exposed via @c metadata().
  cache::metadata_store _metadata_store;
};

}  // namespace sirius::io
