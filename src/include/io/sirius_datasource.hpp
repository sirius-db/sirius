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

#include "io/cache/prefetching_cache.hpp"
#include "io/io_context.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/version_config.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/stream_ref>

#include <span>

namespace sirius::io {

// cudf >= 26.10 (rapidsai/cudf#23669) takes cuda::stream_ref in the
// device_read virtuals; overrides must match the base signature exactly.
#if CUDF_VERSION_MAJOR > 26 || (CUDF_VERSION_MAJOR == 26 && CUDF_VERSION_MINOR >= 10)
using cudf_datasource_stream_t = ::cuda::stream_ref;
#else
using cudf_datasource_stream_t = rmm::cuda_stream_view;
#endif

// ---------------------------------------------------------------------------
// sirius_datasource
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c io_datasource backed by io_uring.
 *
 * Thin delegate: every read method forwards to @c sirius_ioctx, passing the
 * owned @c sirius_io_object by reference.
 *
 * Ownership model: one scan owns one @c sirius_datasource.  The underlying
 * @c sirius_io_object can be shared across multiple datasources (e.g. when
 * the same file is scanned in different pipelines), but the datasource
 * itself stores per-scan state (notably the @c prefetching_handle returned
 * by an @c fadvise call) and is therefore not safe to share.
 */
class sirius_datasource : public cudf::io::datasource {
 public:
  explicit sirius_datasource(std::shared_ptr<sirius_ioctx> io_ctx,
                             std::shared_ptr<sirius_io_object> io_object);

  ~sirius_datasource() override;

  sirius_datasource(sirius_datasource const&)            = delete;
  sirius_datasource& operator=(sirius_datasource const&) = delete;

  // ---- Context accessors ---------------------------------------------------

  [[nodiscard]] std::shared_ptr<sirius_ioctx> io_ctx() const { return _io_ctx; }

  /// The underlying io_object this datasource reads through.  Exposed so
  /// callers that received the datasource from @c sirius_ioctx::open_datasource
  /// can still reach the io_object (e.g. as the metadata-store cache key).
  [[nodiscard]] const sirius_io_object& io_object() const noexcept { return *_io_object; }

  /// Backend-parsed metadata for this datasource's io_object, looked up in the
  /// ioctx's metadata store (null when no cache or no entry). Independent of
  /// the prefetching machinery.
  [[nodiscard]] std::shared_ptr<sirius_io_object_metadata> metadata() const;

  [[nodiscard]] bool store_metadata(std::shared_ptr<sirius_io_object_metadata> metadata);

  // ---- cudf::io::datasource overrides ---------------------------------------

  [[nodiscard]] size_t size() const override;

  [[nodiscard]] bool supports_device_read() const override;

  [[nodiscard]] bool supports_vector_host_read() const;

  [[nodiscard]] bool is_device_read_preferred(size_t) const override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(size_t offset,
                                                                   size_t size) override;

  std::unique_ptr<datasource::buffer> device_read(size_t offset,
                                                  size_t size,
                                                  cudf_datasource_stream_t stream) override;
  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     cudf_datasource_stream_t stream) override;

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        cudf_datasource_stream_t stream) override;

  // ---- Advisory IO ---------------------------------------------------------

  /// \brief Return a fresh datasource that shares this one's @c sirius_ioctx and
  /// @c sirius_io_object (so it points at the same file) but carries an
  /// empty @c prefetching_handle.
  ///
  /// \note Used when a single file is split across multiple scans (e.g. several
  /// row_group_slices from the same parquet file).  Each split owns its
  /// own datasource via @c duplicate so it can call @c fadvise without
  /// stomping on another scan's handle — io_objects are deliberately
  /// shareable across datasources, but handles are not.
  [[nodiscard]] std::unique_ptr<sirius_datasource> duplicate() const;

  /// \brief Hint the IO layer about @p ranges that this scan will (or might) read
  /// soon.
  ///
  /// The behaviour depends on @p site and the io_ctx's
  /// @c preferred_prefetching_stage:
  ///   - @c speculative / @c immediate: only honored when @p site matches
  ///     the ioctx's preferred mode.  Hands @p ranges to the prefetching
  ///     cache and stashes the returned @c prefetching_handle on this
  ///     datasource so a later @c fadvise(disposable) can cancel.
  ///   - @c disposable: always honored.  If a handle is stored (i.e. a
  ///     prior speculative/immediate call enqueued work), cancel it so the
  ///     cache worker drops still-pending entries.
  ///   - @c none: no-op (either the call site asked for nothing, or the
  ///     backend opted out of prefetching).
  ///
  /// Calling @c fadvise with a non-disposable @p site while a handle is
  /// already stored emits a warning: the datasource lifecycle expects one
  /// speculative-or-immediate insert per scan, with a single
  /// @c disposable call at consume time.
  void fadvise(std::span<const cudf::io::text::byte_range_info> ranges, std::optional<int> dev_id);

  void prefetch(cache::prefetching_stage site);

  [[nodiscard]] bool uses_prefetching_cache() const noexcept;

 private:
  std::shared_ptr<sirius_ioctx> _io_ctx;
  std::shared_ptr<sirius_io_object> _io_object;
  /// Handle of the most recent speculative/immediate insert into the
  /// prefetching cache, or empty if none was made.  fadvise(disposable)
  /// uses this to cancel still-pending work.
  cache::prefetching_handle _prefetch_handle;
};

}  // namespace sirius::io
