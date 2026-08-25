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
#include "io/cache/config.hpp"
#include "io/cache/metadata_store.hpp"
#include "io/types.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::io {

enum class io_context_type { uring, restful, kvikio };

/// Hint passed to @c open_datasource so a backend can tailor how it resolves an
/// object's metadata.  @c generic resolves the size however is cheapest for the
/// scheme (a HEAD for object stores).  @c parquet_footer_probe asks the backend
/// to resolve the size *and* stash the object's trailing bytes in one
/// round-trip (a suffix-range GET), so the parquet footer reads that follow are
/// served locally instead of costing extra round-trips.
enum class open_hint { generic, parquet_footer_probe };

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
// ioctx
// ---------------------------------------------------------------------------

/**
 * @brief Abstract shared context passed to every datasource.
 *
 * Holds resources that are shared across all datasources (cache, reactor
 * threads, ...). Extend this class to provide a concrete I/O backend.
 */
class ioctx : public std::enable_shared_from_this<ioctx> {
 public:
  ioctx();
  virtual ~ioctx();

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

  /// As above, forwarding @p hint to the backend's io_object resolution so it
  /// can, e.g., prefetch a parquet footer in the same round-trip as the size.
  [[nodiscard]] std::unique_ptr<sirius_datasource> open_datasource(std::string path,
                                                                   open_hint hint);

  /// As above, with the object's size already known (e.g. from an S3
  /// ListObjectsV2 response), so a backend that can act on it skips its size
  /// discovery entirely (no HEAD for object stores).
  [[nodiscard]] std::unique_ptr<sirius_datasource> open_datasource(std::string path,
                                                                   std::uint64_t known_size);

  /// Open the backend's connections to whatever serves @p bucket_url, ahead of
  /// the first read, so a query does not pay connection setup on its hot path.
  ///
  /// @p bucket_url names a container, not an object -- @c "s3://my-bucket".  A
  /// full object URL is accepted and its key ignored, since connections are per
  /// endpoint and the object adds nothing to the identity.  Deliberately not an
  /// @c io_object: opening one is itself a round trip over the connection being
  /// warmed, which would leave the warm-up nothing left to hide, and it would
  /// tie warm-up traffic to individual data files.
  ///
  /// Best-effort by contract: it must not throw and must not block the caller,
  /// and a failed warm-up is never a reason to fail a read.  For a transport
  /// that is what warms, even a rejected request is a success -- a 403 still
  /// completed the DNS lookup, the TCP connect and the TLS handshake, which is
  /// all that was being bought.
  ///
  /// The default is a no-op, which is the right answer for every backend whose
  /// "connection" is a file descriptor it already holds.
  virtual void warmup(std::string_view /*bucket_url*/) noexcept {}

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
  /// (cf. @c host_readv_async_io).  When false, the prefetching layer
  /// cannot amortise per-request overhead and must fall back to
  /// @c scan_stage::none.
  [[nodiscard]] virtual bool supports_vector_host_read() const noexcept = 0;

  /// Whether the backend can efficiently serve a batch of device reads.
  /// Backends may still process mixed slices serially when this is false.
  [[nodiscard]] virtual bool supports_device_range_read() const noexcept = 0;

  /// Whether this backend would rather be handed one batched request covering
  /// everything a reader needs than a stream of small reads as the reader walks
  /// the file.
  ///
  /// Unlike the supports_* flags this is a preference, not a capability: a
  /// backend that says no can still serve a batch, and one that says yes can
  /// still serve small reads.  It reflects what the request itself costs.  For
  /// an object store a read is a round trip, so the shape of the request set
  /// dominates and knowing all of it up front is worth a great deal; for a local
  /// file a read is a syscall against page cache or NVMe, and batching buys
  /// little while forcing the caller to materialise ranges it may not need.
  ///
  /// Conservatively false: a backend opts in.
  [[nodiscard]] virtual bool prefers_bulk_io() const noexcept { return false; }

  /// The smallest unit this backend can address, in bytes.  A read is widened
  /// out to a multiple of it before being issued: a local file opened O_DIRECT
  /// can only transfer whole pages, while an object store addresses single
  /// bytes and pays nothing for an odd offset.
  ///
  /// Conservatively 1 -- a backend that has not opted in is never widened.
  [[nodiscard]] virtual std::size_t min_alignment_requirement() const noexcept { return 1; }

  /// The largest gap between two ranges still worth bridging into one request,
  /// in bytes.  The bridged bytes are fetched and discarded, traded against the
  /// cost of a second request: a page for a local file, a whole round trip for
  /// an object store, which is why the two want very different answers.
  ///
  /// Conservatively 0 -- only adjacent ranges are fused.
  [[nodiscard]] virtual std::size_t merge_gap_size() const noexcept { return 0; }

  /// How many scan tasks the readahead manager may keep in flight against this
  /// backend at once, as configured on its reactors.  Zero means this backend
  /// opts out of readahead scheduling entirely.
  ///
  /// The bound is a property of the backend, not of the query: it reflects the
  /// queue depth the device is worth driving at (see the per-backend defaults
  /// on each reactor config).  The base returns 0 so a backend that has not
  /// opted in is never scheduled against.
  [[nodiscard]] virtual std::size_t n_max_concurrent_scans() const noexcept { return 0; }

  /// Backend-specific perf counters, formatted for a log/stderr dump, with the
  /// counters zeroed on the way out so successive calls report per-window
  /// deltas.  Empty when the backend keeps no counters.  Best-effort
  /// observability: implementations read racy relaxed atomics and must not
  /// throw.
  [[nodiscard]] virtual std::string perf_report_and_reset() noexcept { return {}; }

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
    io::cache::config const& cache_config,
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
  /// in @c ~ioctx would run AFTER the derived part of the
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
    return can_use_prefetching_cache() && _cache;
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

  virtual size_t host_read_io(const io_object& obj, size_t offset, size_t size, uint8_t* dst) = 0;

  virtual exec::semi_future<size_t> host_read_async_io(const io_object& obj,
                                                       size_t offset,
                                                       size_t size,
                                                       uint8_t* dst) noexcept;

  virtual exec::semi_future<size_t> device_read_async_io(const io_object& obj,
                                                         size_t offset,
                                                         size_t size,
                                                         uint8_t* dst,
                                                         rmm::cuda_stream_view stream) noexcept;

  virtual exec::semi_future<size_t> host_readv_async_io(const io_object& obj,
                                                        std::span<const slice> slices) noexcept;

  virtual exec::semi_future<size_t> device_readv_async_io(const io_object& obj,
                                                          std::span<const slice> slices,
                                                          rmm::cuda_stream_view stream) noexcept;

  /// The sole asynchronous backend hook. All scalar/vector host/device APIs
  /// construct prepared slices and forward here; reactors perform physical
  /// chunking only after queue and slot pressure are known.
  virtual exec::semi_future<size_t> mixed_readv_async_io(
    const io_object& obj, std::vector<prepared_io_slice>&& slices) noexcept = 0;

  exec::semi_future<size_t> host_device_readv_async_io(
    const io_object& obj, std::vector<prepared_io_slice>&& slices) noexcept
  {
    return mixed_readv_async_io(obj, std::move(slices));
  }

  bool can_use_prefetching_cache() const noexcept
  {
    return supports_vector_host_read() || supports_host_to_device_read();
  }

 protected:
  /// Backend hook: open native handles / resolve metadata for @p path and
  /// return a populated io_object.  Invoked by @c open_datasource; not part of
  /// the public surface (callers receive a ready @c sirius_datasource).  Throws
  /// on unsupported / unreachable paths.
  virtual std::shared_ptr<io_object> create_io_object(std::string path) = 0;

  /// Hinted variant.  The base implementation ignores @p hint and delegates to
  /// the required @c create_io_object(path); a backend that can act on the hint
  /// (e.g. rest_ioctx's suffix-range footer probe) overrides this.  Kept a
  /// distinct virtual — not a defaulted argument on the pure virtual above — so
  /// the hint dispatches on the dynamic type instead of binding statically.
  virtual std::shared_ptr<io_object> create_io_object(std::string path, open_hint hint);

  /// Known-size variant.  The base implementation ignores @p known_size and
  /// delegates to the required @c create_io_object(path); a backend whose size
  /// discovery would otherwise cost a round-trip overrides this to build the
  /// io_object without one.  Same distinct-virtual rationale as the hint
  /// variant above.
  virtual std::shared_ptr<io_object> create_io_object(std::string path, std::uint64_t known_size);

  /// Owned by this ioctx.  Built by @ref initialize_cache, destroyed
  /// by @ref shutdown_cache (or the ioctx destructor as a safety net,
  /// though callers are expected to drive the lifecycle explicitly so
  /// reactors stay alive while workers drain).
  std::unique_ptr<cache::prefetching_cache> _cache;

  /// Independent of the prefetching machinery — exposed via @c metadata_store().
  cache::metadata_store _metadata_store;
};

/// Resolves the ioctx that serves a given file path (s3:// -> rest, local ->
/// uring/kvikio).  Returns a valid ioctx or throws if no backend supports the path.
using ioctx_resolver = std::function<std::shared_ptr<ioctx>(std::string_view)>;

}  // namespace sirius::io
