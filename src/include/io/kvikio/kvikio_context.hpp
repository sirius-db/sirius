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

#include <kvikio/file_handle.hpp>

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
// kvikio_io_object
// ---------------------------------------------------------------------------

/**
 * @brief @c io_object that owns a kvikIO file handle.
 *
 * Owns the handle for the file's lifetime; @c kvikio_context's read primitives
 * forward straight to it.  kvikIO picks GDS or a POSIX/compat path per call
 * based on the pointer type and its own compatibility mode, so this backend
 * serves both host and device destinations from the same handle.
 */
class kvikio_io_object final : public io_object {
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

 private:
  std::string _path;
  mutable kvikio::FileHandle _handle;
  size_t _file_size{0};
};

// ---------------------------------------------------------------------------
// kvikio_context
// ---------------------------------------------------------------------------

/**
 * @brief Fallback @c ioctx backed directly by kvikIO
 *        (@c kvikio::FileHandle).
 *
 * The universal local-file backend: it claims any path, so the registry uses it
 * only after the explicit backends (uring / rest) decline.  Unlike those, it
 * owns no reactors and no bounce staging — every read goes straight to a
 * kvikIO handle, which internally chooses GDS or a POSIX/compat path.
 *
 * It implements the protected @c _io primitives (not the public read API); the
 * base class's public reads route through them.
 *
 * Capabilities:
 *   - @c supports_device_read: true (kvikIO reads into device memory, via GDS
 *     where the platform allows).
 *   - @c supports_vector_host_read: false — no batched dispatch path.
 *   - @c supports_host_to_device_read: false — no bounce-staging path.
 *   - @c preferred_prefetching_stage: @c none.
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
  /// @throw std::invalid_argument on a zero @c nthreads, @c task_size, or
  ///        @c bounce_buffer_size.
  explicit kvikio_context(kvikio_config cfg);

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
  [[nodiscard]] cache::prefetching_stage preferred_prefetching_stage() const noexcept override
  {
    return cache::prefetching_stage::none;
  }

  /// kvikIO applies no physical block alignment of its own, so ranges pass
  /// through unchanged.
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<const cudf::io::text::byte_range_info> ranges,
    std::optional<size_t> /*alignment*/) const noexcept override;

  // -- Backend primitives ---------------------------------------------------

  size_t host_read_io(const io_object& obj, size_t offset, size_t size, uint8_t* dst) final;

  exec::semi_future<size_t> host_read_async_io(const io_object& obj,
                                               size_t offset,
                                               size_t size,
                                               uint8_t* dst) noexcept final;

  exec::semi_future<size_t> device_read_async_io(const io_object& obj,
                                                 size_t offset,
                                                 size_t size,
                                                 uint8_t* dst,
                                                 rmm::cuda_stream_view stream) noexcept final;

  /// Unsupported: kvikIO has no bounce-staged host->device path here.  Returns
  /// a failed future rather than misbehaving silently.
  exec::semi_future<size_t> host_to_device_read_async_io(
    const io_object& obj,
    std::span<io_object_segment> slices,
    size_t offset,
    size_t size,
    uint8_t* device_dst,
    rmm::cuda_stream_view stream) noexcept final;

  /// Unsupported: no batched dispatch (hence @c supports_vector_host_read()
  /// is false and the prefetching cache stays unarmed).
  exec::semi_future<size_t> host_read_ranges_async_io(
    const io_object& obj, std::span<io_object_segment> segments) noexcept final;

  /// The config this context was built with (default-constructed when none was
  /// supplied).  Only @c compat_mode is still consulted after construction; the
  /// rest already went into kvikIO's globals.
  [[nodiscard]] kvikio_config const& config() const noexcept { return _config; }

 protected:
  /// Backend hook invoked by @c ioctx::open_datasource: open @p path
  /// with kvikIO and record its size.  Applies @c config().compat_mode to the
  /// handle when set.  Throws when the file cannot be opened.
  std::shared_ptr<io_object> create_io_object(std::string path) override;

 private:
  kvikio_config _config;
};

}  // namespace sirius::io
