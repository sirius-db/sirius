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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::io {

// ---------------------------------------------------------------------------
// kvikio_object
// ---------------------------------------------------------------------------

/**
 * @brief Common base for the @c io_object flavours this backend serves.
 *
 * kvikIO reaches local files through a @c kvikio::FileHandle and object-store
 * URIs through a @c kvikio::RemoteHandle; the two handle types share no base,
 * so this class carries the one operation @c kvikio_context needs from both.
 */
class kvikio_object : public io_object {
 public:
  /// Read up to @p size bytes at @p offset into @p dst (host OR device memory;
  /// kvikIO dispatches on the pointer type).  Blocking; returns the byte count.
  /// Implementations clamp @p size to the object's size.
  [[nodiscard]] virtual std::size_t read_at(void* dst,
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
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept final { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept final { return _path; }
  [[nodiscard]] size_t size() const noexcept final { return _file_size; }

  /// Mutable: kvikIO's read entry points are non-const, and the reads issued
  /// through them do not mutate observable file state.
  [[nodiscard]] kvikio::FileHandle& handle() const noexcept { return _handle; }

  [[nodiscard]] std::size_t read_at(void* dst, std::size_t size, std::size_t offset) const final;

 private:
  std::string _path;
  mutable kvikio::FileHandle _handle;
  size_t _file_size{0};
};

// ---------------------------------------------------------------------------
// kvikio_remote_io_object
// ---------------------------------------------------------------------------

/**
 * @brief @c kvikio_object that owns a kvikIO remote handle (@c s3:// URIs).
 *
 * The handle performs a HEAD at construction to learn the object size, then
 * serves ranged GETs over libcurl.  Device destinations are supported, but
 * kvikIO bounces them through a host buffer internally — there is no
 * stream-ordered read on a remote handle.
 */
class kvikio_remote_io_object final : public kvikio_object {
 public:
  /// @param uri The original @c s3://bucket/key URI — kept verbatim so cache
  ///            keys stay stable regardless of the signed URL kvikIO builds.
  kvikio_remote_io_object(std::string uri, kvikio::RemoteHandle handle, size_t object_size)
    : _uri(std::move(uri)), _handle(std::move(handle)), _object_size(object_size)
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept final { return _uri; }
  [[nodiscard]] const std::string& object_path() const noexcept final { return _uri; }
  [[nodiscard]] size_t size() const noexcept final { return _object_size; }

  /// Mutable for the same reason as @ref kvikio_io_object::handle.
  [[nodiscard]] kvikio::RemoteHandle& handle() const noexcept { return _handle; }

  [[nodiscard]] std::size_t read_at(void* dst, std::size_t size, std::size_t offset) const final;

 private:
  std::string _uri;
  mutable kvikio::RemoteHandle _handle;
  size_t _object_size{0};
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
 * hook eagerly; the base scalar and vector wrappers construct prepared slices.
 *
 * Capabilities:
 *   - @c supports_device_read: true (kvikIO reads into device memory, via GDS
 *     where the platform allows).
 *   - @c supports_vector_host_read: false — no batched dispatch path.
 *   - @c supports_host_to_device_read: false — no bounce-staging path.
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
  /// @throw std::invalid_argument on a zero @c nthreads, @c task_size, or
  ///        @c bounce_buffer_size.
  explicit kvikio_context(kvikio_config cfg, object_store_config os = {});

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
  [[nodiscard]] bool supports_vector_host_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_device_range_read() const noexcept override { return false; }

  /// Always false: kvikIO is a local-file path, and its whole point is streaming
  /// straight to device per read rather than assembling a batch first.
  [[nodiscard]] bool prefers_bulk_io() const noexcept override { return false; }

  [[nodiscard]] std::size_t n_max_concurrent_scans() const noexcept override
  {
    return _config.n_max_concurrent_scans;
  }

  /// kvikIO applies no physical block alignment of its own, so ranges pass
  /// through unchanged.
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<const cudf::io::text::byte_range_info> ranges,
    std::optional<size_t> /*alignment*/) const noexcept override;

  // -- Backend primitives ---------------------------------------------------

  size_t host_read_io(const io_object& obj, size_t offset, size_t size, uint8_t* dst) final;

  /// KvikIO has no reactor queue, so it consumes prepared slices eagerly.
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

 private:
  kvikio_config _config;
  object_store_config _object_store;
};

}  // namespace sirius::io
