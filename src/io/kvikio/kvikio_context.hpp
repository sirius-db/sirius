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
#include "io/io_context.hpp"
#include "io/kvikio/config.hpp"
#include "io/object_store_config.hpp"

#include <kvikio/file_handle.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/threadpool_wrapper.hpp>

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::io {

namespace rest {
class rest_reactor;
}  // namespace rest

// ---------------------------------------------------------------------------
// kvikio_object
// ---------------------------------------------------------------------------

/**
 * @brief Common base for the @c io_object flavours this backend serves.
 *
 * kvikIO reaches local files through a @c kvikio::FileHandle and object-store
 * URIs through a @c kvikio::RemoteHandle; the two handle types share no base,
 * so this class carries the operations @c kvikio_context needs from both.
 */
class kvikio_object : public io_object {
 public:
  /// Read up to @p size bytes at @p offset into @p dst (host OR device memory;
  /// kvikIO dispatches on the pointer type).  Blocking; returns the byte count.
  /// Implementations clamp @p size to the object's size.
  [[nodiscard]] virtual std::size_t read_at(void* dst,
                                            std::size_t size,
                                            std::size_t offset) const = 0;

  /// Start reading exactly @p size bytes at @p offset into @p dst (host OR
  /// device memory) and return without waiting.  The caller clamps @p size to
  /// the object's size and keeps @p dst alive until the future is ready.
  [[nodiscard]] virtual std::future<std::size_t> read_async(void* dst,
                                                            std::size_t size,
                                                            std::size_t offset) const = 0;
};

// ---------------------------------------------------------------------------
// kvikio_io_object
// ---------------------------------------------------------------------------

/**
 * @brief @c kvikio_object that owns a kvikIO file handle (local paths).
 *
 * Owns the handle for the file's lifetime; @c kvikio_context's read primitives
 * forward straight to it.  kvikIO picks GDS or a POSIX/compat path per call
 * based on the pointer type and its own compatibility mode, so this backend
 * serves both host and device destinations from the same handle.
 */
class kvikio_io_object final : public kvikio_object {
 public:
  kvikio_io_object(std::string path, kvikio::FileHandle handle, size_t file_size)
    : _path(std::move(path)), _handle(std::move(handle)), _file_size(file_size)
  {
    _has_identity = capture_local_identity(_handle.fd(), _identity);
    if (_has_identity) _file_size = _identity.size;
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept final { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept final { return _path; }
  [[nodiscard]] size_t size() const noexcept final { return _file_size; }

  [[nodiscard]] object_identity identity() const override
  {
    return _has_identity ? _identity : io_object::identity();
  }

  /// Mutable: kvikIO's read entry points are non-const, and the reads issued
  /// through them do not mutate observable file state.
  [[nodiscard]] kvikio::FileHandle& handle() const noexcept { return _handle; }

  [[nodiscard]] std::size_t read_at(void* dst, std::size_t size, std::size_t offset) const final;

  [[nodiscard]] std::future<std::size_t> read_async(void* dst,
                                                    std::size_t size,
                                                    std::size_t offset) const final;

 private:
  std::string _path;
  mutable kvikio::FileHandle _handle;
  size_t _file_size{0};
  object_identity _identity;
  bool _has_identity = false;
};

// ---------------------------------------------------------------------------
// kvikio_remote_io_object
// ---------------------------------------------------------------------------

/**
 * @brief @c kvikio_object that owns a kvikIO remote handle (@c s3:// URIs).
 *
 * The handle performs a HEAD at construction to learn the object size, then
 * serves ranged GETs over libcurl.  Device destinations are supported, but
 * kvikIO bounces them through a host buffer internally on a private per-thread
 * stream — there is no stream-ordered read on a remote handle.  @ref
 * kvikio_context therefore synchronizes the destination stream before issuing
 * the GET, so the read observes all work previously queued on that stream and
 * the bytes are visible to it once the read returns.
 */
class kvikio_remote_io_object final : public kvikio_object {
 public:
  /// Trailing bytes of the object, fetched with it by a footer probe.
  using stash_type = std::shared_ptr<const std::span<const std::uint8_t>>;

  /// @param uri The original @c s3://bucket/key URI — kept verbatim so cache
  ///            keys stay stable regardless of the signed URL kvikIO builds.
  /// @param etag The object's ETag when its open learned one, quotes preserved.
  ///            A strong tag lets cached metadata outlive this open.
  /// @param stash The object's bytes over [@p window_lo, @p object_size) when
  ///            the open fetched them, so reads inside that window need no GET.
  kvikio_remote_io_object(std::string uri,
                          kvikio::RemoteHandle handle,
                          size_t object_size,
                          std::string etag   = {},
                          size_t window_lo   = 0,
                          stash_type stash   = nullptr)
    : _uri(std::move(uri)),
      _handle(std::move(handle)),
      _object_size(object_size),
      _etag(std::move(etag)),
      _window_lo(window_lo),
      _stash(std::move(stash))
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept final { return _uri; }
  [[nodiscard]] const std::string& object_path() const noexcept final { return _uri; }
  [[nodiscard]] size_t size() const noexcept final { return _object_size; }
  [[nodiscard]] std::string_view validation_tag() const noexcept final { return _etag; }

  /// Like the REST backend's objects: a strong ETag identifies the object
  /// across opens, anything else only this open.
  [[nodiscard]] object_identity identity() const final;

  /// Mutable for the same reason as @ref kvikio_io_object::handle.
  [[nodiscard]] kvikio::RemoteHandle& handle() const noexcept { return _handle; }

  [[nodiscard]] std::size_t read_at(void* dst, std::size_t size, std::size_t offset) const final;

  [[nodiscard]] std::future<std::size_t> read_async(void* dst,
                                                    std::size_t size,
                                                    std::size_t offset) const final;

 private:
  /// Copy [@p offset, @p offset + @p size) from the stash into @p dst (host or
  /// device memory) when the stash covers it.  Returns whether it did.
  [[nodiscard]] bool read_from_stash(void* dst, std::size_t size, std::size_t offset) const;

  std::string _uri;
  mutable kvikio::RemoteHandle _handle;
  size_t _object_size{0};
  std::string _etag;
  size_t _window_lo{0};
  stash_type _stash;
};

// ---------------------------------------------------------------------------
// kvikio_context
// ---------------------------------------------------------------------------

/**
 * @brief Fallback @c ioctx backed directly by kvikIO
 *        (@c kvikio::FileHandle).
 *
 * The universal backend: it claims any path, so the registry uses it only after
 * the explicit backends (uring / rest) decline — or, when @c backend=kvikio is
 * configured, for local files AND @c s3:// URIs.  Unlike those backends it owns
 * no reactors and no bounce staging — every read goes straight to a kvikIO
 * handle (a @c FileHandle locally, a @c RemoteHandle for @c s3://), which
 * internally chooses GDS or a POSIX/compat path.
 *
 * LIST / glob is NOT served here: it stays on the REST backend.
 *
 * It implements synchronous host I/O plus the shared `mixed_readv_async_io`
 * hook; the base scalar and vector wrappers construct prepared slices.  A batch
 * starts every slice's read at once and returns without waiting: kvikIO serves
 * the reads concurrently (its thread pool, or its MULTI_POLL reactors for
 * remote objects), and a small completion pool settles the batch.
 *
 * Capabilities:
 *   - @c supports_device_read: true (kvikIO reads into device memory, via GDS
 *     where the platform allows).
 *   - @c supports_vector_host_read: true — a batch of host reads, including
 *     fills of prefetching-cache chunks, is served concurrently.
 *   - @c supports_host_to_device_read: true — a slice can fill host buffers
 *     (cache chunks) and copy its range on to the device in one request.
 *   - @c supports_device_range_read: true — a batch of device reads is served
 *     concurrently.
 *
 * Together these let the backend arm the prefetching cache, and with it the
 * readahead, whose budget is @ref n_max_concurrent_scans.
 */
class kvikio_context final : public ioctx {
 public:
  /// Construct with kvikIO left at its own (env-var-seeded) defaults.
  kvikio_context() = default;

  /// Construct and apply @p cfg.  Every field except @c compat_mode is pushed
  /// into kvikIO's PROCESS-GLOBAL defaults — see @ref kvikio_config for the
  /// sharing and ordering caveats.  @c compat_mode is retained and applied per
  /// file handle at open time instead.
  ///
  /// @p os supplies the credentials used for @c s3:// objects; every value is
  /// handed to kvikIO explicitly so remote reads never depend on @c AWS_* env
  /// vars.  A default-constructed @p os disables the remote path (opening an
  /// @c s3:// URI then throws).
  ///
  /// @p control_plane, when given, opens @c s3:// objects with the REST
  /// backend's one-shot requests: a HEAD, or for a parquet footer probe a
  /// suffix-range GET that also returns the footer bytes.  Either learns the
  /// object's ETag, which lets its parquet metadata be cached across opens.
  /// It is never started: only its blocking control-plane calls are used, and
  /// every data read still goes through kvikIO.  Without it, opens use
  /// kvikIO's own HEAD, which reports only the size.
  ///
  /// @throw std::invalid_argument on a zero @c nthreads, @c task_size, or
  ///        @c bounce_buffer_size.
  explicit kvikio_context(kvikio_config cfg,
                          object_store_config os                      = {},
                          std::shared_ptr<rest::rest_reactor> control_plane = nullptr);

  ~kvikio_context() override
  {
    // See ioctx::pre_destroy — drains the cache (if any) while
    // this derived part of the object is still alive.  No reactors to
    // tear down for kvikio_context, but the contract still applies.
    this->pre_destroy();
  }

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::kvikio; }

  void shutdown() noexcept override {}

  [[nodiscard]] bool supports(std::string_view path) const noexcept override;
  [[nodiscard]] bool supports_device_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_vector_host_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_device_range_read() const noexcept override { return true; }

  /// True unless @c kvikio_config::prefer_bulk_io turns it off: a batch's reads
  /// run concurrently, so handing over every range a reader needs at once costs
  /// one round trip of latency rather than one per range, which is what an
  /// object store needs.
  [[nodiscard]] bool prefers_bulk_io() const noexcept override
  {
    return _config.prefer_bulk_io.value_or(true);
  }

  /// Remote objects follow @ref prefers_bulk_io(); local files never prefer the
  /// bulk route, like the local uring backend, since a local read is a syscall
  /// rather than a round trip.
  [[nodiscard]] bool prefers_bulk_io(io_object const& obj) const noexcept override;

  /// The largest gap between two ranges worth bridging into one request.  Same
  /// value as the REST backend's default: for an object store a second request
  /// costs a round trip, far more than reading a gap of this size.
  [[nodiscard]] std::size_t merge_gap_size() const noexcept override { return 512UL << 10; }

  /// How many scan tasks the readahead may keep in flight against this backend:
  /// @c kvikio_config::n_max_concurrent_scans, which the ioctx factory defaults
  /// to the REST backend's budget.  Zero opts out of readahead.
  [[nodiscard]] std::size_t n_max_concurrent_scans() const noexcept override
  {
    return _config.n_max_concurrent_scans.value_or(0);
  }

  /// kvikIO applies no physical block alignment of its own, so ranges pass
  /// through unchanged.
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<const cudf::io::text::byte_range_info> ranges,
    std::optional<size_t> /*alignment*/) const noexcept override;

  // -- Backend primitives ---------------------------------------------------

  size_t host_read_io(const io_object& obj, size_t offset, size_t size, uint8_t* dst) final;

  /// Starts every slice's read and returns without waiting for any of them.
  /// This is the backend's only asynchronous hook; the base scalar/vector
  /// wrappers all forward here.
  exec::semi_future<size_t> mixed_readv_async_io(
    const io_object& obj, std::vector<prepared_io_slice>&& slices) noexcept final;

  /// The config this context was built with (default-constructed when none was
  /// supplied).  Only @c compat_mode is still consulted after construction; the
  /// rest already went into kvikIO's globals.
  [[nodiscard]] kvikio_config const& config() const noexcept { return _config; }

  /// Object-store credentials used for @c s3:// objects.
  [[nodiscard]] object_store_config const& object_store() const noexcept { return _object_store; }

 protected:
  /// Backend hook invoked by @c ioctx::open_datasource: open @p path with
  /// kvikIO and record its size.  An @c s3:// path builds a signed
  /// @c RemoteHandle from @ref object_store (throwing when the store is not
  /// configured); anything else opens a @c FileHandle, applying
  /// @c config().compat_mode when set.  Throws when the object cannot be
  /// opened.
  std::shared_ptr<io_object> create_io_object(std::string path) override;

  /// Hinted open: @c open_hint::parquet_footer_probe opens an @c s3:// object
  /// with one suffix-range GET through the control plane, which resolves the
  /// size and ETag and stashes the footer, so its reads need no further GET.
  std::shared_ptr<io_object> create_io_object(std::string path, open_hint hint) override;

  /// Known-size open: an @c s3:// object whose size the caller already learned
  /// (e.g. from a LIST) is opened without the HEAD request.  Other paths fall
  /// back to @ref create_io_object(std::string).
  std::shared_ptr<io_object> create_io_object(std::string path, std::uint64_t known_size) override;

 private:
  /// Open @p path as an @c s3:// object.  A known size skips the HEAD; a footer
  /// probe, when asked for and a control plane is present, replaces it.
  std::shared_ptr<io_object> create_remote_io_object(std::string path,
                                                     std::optional<std::uint64_t> known_size,
                                                     bool footer_probe);

  kvikio_config _config;
  object_store_config _object_store;
  std::shared_ptr<rest::rest_reactor> _control_plane;

  /// Waits on the reads of in-flight batches and settles them.  Its threads
  /// only block on futures, so it is sized for the number of batches in flight,
  /// not for CPU work.  Declared last so it is destroyed first: its destructor
  /// waits for every batch still being settled.
  kvikio::ThreadPool _completion_pool{64};
};

}  // namespace sirius::io
