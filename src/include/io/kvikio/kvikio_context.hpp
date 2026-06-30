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

#include <cudf/io/datasource.hpp>

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::io {

// ---------------------------------------------------------------------------
// kvikio_io_object
// ---------------------------------------------------------------------------

/**
 * @brief @c sirius_io_object that holds a cudf::io::datasource (kvikio-backed
 *        on the default cudf build).
 *
 * Owns the datasource for the file's lifetime; @c kvikio_context's read
 * overrides forward straight to this datasource so we don't have to
 * translate cudf's future-returning API into the push/callback shape that
 * the base class's protected @c _io primitives expect.
 */
class kvikio_io_object final : public sirius_io_object {
 public:
  kvikio_io_object(std::string path, std::shared_ptr<cudf::io::datasource> ds, size_t file_size)
    : _path(std::move(path)), _datasource(std::move(ds)), _file_size(file_size)
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept final { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept final { return _path; }
  [[nodiscard]] size_t size() const noexcept final { return _file_size; }

  [[nodiscard]] cudf::io::datasource& datasource() const noexcept { return *_datasource; }

 private:
  std::string _path;
  std::shared_ptr<cudf::io::datasource> _datasource;
  size_t _file_size{0};
};

// ---------------------------------------------------------------------------
// kvikio_context
// ---------------------------------------------------------------------------

/**
 * @brief Fallback @c sirius_ioctx that defers to cudf's default datasource
 *        (kvikio-backed for file paths on a stock cudf build).
 *
 * Why override the public read API directly instead of the protected
 * @c _io primitives?  cudf's async path returns @c std::future<size_t>,
 * and the protected @c host_read_async_io / @c device_read_async_io
 * contract returns @c exec::semi_future<size_t>.  Bridging the cudf
 * future into a semi_future per call requires a detached waiter
 * thread; instead, kvikio overrides the public read API so the cudf
 * future flows through unchanged.  The protected @c _io primitives
 * become unreachable placeholders.
 *
 * Capabilities:
 *   - @c supports_device_read: true (cudf's datasource supports it where the
 *     platform allows, e.g. GDS).
 *   - @c supports_vector_host_read: false — no batched dispatch path.
 *   - @c preferred_prefetching_stage: @c none.
 */
class kvikio_context final : public sirius_ioctx {
 public:
  kvikio_context() = default;
  ~kvikio_context() override
  {
    // See sirius_ioctx::pre_destroy — drains the cache (if any) while
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

  std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<const cudf::io::text::byte_range_info> ranges,
    std::optional<size_t> /*alignment*/) const noexcept override;

  // -- Protected _io primitives -------------------------------------------
  //
  // The base class's default read implementations route through these on a
  // cache miss, but kvikio_context overrides the public read API and never
  // attaches a cache (supports_vector_host_read == false), so these are
  // unreachable from the documented code paths.  They remain pure-virtual
  // on the base, so we provide throwing placeholders to keep the class
  // instantiable; any future caller that bypasses the public API will see
  // a clear failure rather than silent misbehaviour.

  size_t host_read_io(const sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst) final;

  exec::semi_future<size_t> host_read_async_io(const sirius_io_object& obj,
                                               size_t offset,
                                               size_t size,
                                               uint8_t* dst) noexcept final;

  exec::semi_future<size_t> device_read_async_io(const sirius_io_object& obj,
                                                 size_t offset,
                                                 size_t size,
                                                 uint8_t* dst,
                                                 rmm::cuda_stream_view stream) noexcept final;

  exec::semi_future<size_t> host_to_device_read_async_io(
    const sirius_io_object& obj,
    std::span<io_object_segment> slices,
    size_t offset,
    size_t size,
    uint8_t* device_dst,
    rmm::cuda_stream_view stream) noexcept final;

  exec::semi_future<size_t> host_read_ranges_async_io(
    const sirius_io_object& obj, std::span<io_object_segment> segments) noexcept final;

 protected:
  /// Backend hook invoked by @c sirius_ioctx::open_datasource.
  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override;
};

}  // namespace sirius::io
