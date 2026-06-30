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

#include "io/kvikio/kvikio_context.hpp"

#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace sirius::io {

namespace {
// -- Protected placeholders --------------------------------------------------

const kvikio_io_object& as_kvikio(const sirius_io_object& obj)
{
  // Concrete type is enforced by create_io_object below; a mismatch is a
  // programmer error (e.g. mixing io_objects across backends), not user
  // input, so a static_cast is appropriate.
  return static_cast<const kvikio_io_object&>(obj);
}

}  // namespace

std::shared_ptr<sirius_io_object> kvikio_context::create_io_object(std::string path)
{
  // cudf::io::datasource::create returns a unique_ptr; promote to shared_ptr
  // so the kvikio_io_object can expose access without transferring
  // ownership (the io_object outlives any single sirius_datasource we hand
  // back from make_datasource).
  std::shared_ptr<cudf::io::datasource> ds = cudf::io::datasource::create(path);
  auto const file_size                     = ds->size();
  return std::make_shared<kvikio_io_object>(std::move(path), std::move(ds), file_size);
}

bool kvikio_context::supports(std::string_view /*path*/) const noexcept
{
  // cudf::io::datasource::create handles file paths, URIs, and registered
  // protocol handlers; the actual feasibility check happens at
  // create_io_object time, where opening the file may throw.
  return true;
}

// -- Public read API ---------------------------------------------------------

std::vector<cudf::io::text::byte_range_info> kvikio_context::align_and_coalesce(
  std::span<const cudf::io::text::byte_range_info> ranges,
  std::optional<size_t> /*alignment*/) const noexcept
{
  return {ranges.begin(), ranges.end()};
}

size_t kvikio_context::host_read_io(const sirius_io_object& obj,
                                    size_t offset,
                                    size_t size,
                                    uint8_t* dst)
{
  return as_kvikio(obj).datasource().host_read(offset, size, reinterpret_cast<uint8_t*>(dst));
}

exec::semi_future<size_t> kvikio_context::host_read_async_io(const sirius_io_object& obj,
                                                             size_t offset,
                                                             size_t size,
                                                             uint8_t* dst) noexcept
{
  auto fut =
    as_kvikio(obj).datasource().host_read_async(offset, size, reinterpret_cast<uint8_t*>(dst));
  return exec::make_semi_future_with([fut = std::move(fut)]() mutable { return fut.get(); });
}

exec::semi_future<size_t> kvikio_context::device_read_async_io(
  const sirius_io_object& obj,
  size_t offset,
  size_t size,
  uint8_t* dst,
  rmm::cuda_stream_view stream) noexcept
{
  auto fut = as_kvikio(obj).datasource().device_read_async(
    offset, size, reinterpret_cast<uint8_t*>(dst), stream);
  return exec::make_semi_future_with([fut = std::move(fut)]() mutable { return fut.get(); });
}

exec::semi_future<size_t> kvikio_context::host_to_device_read_async_io(
  const sirius_io_object& obj,
  std::span<io_object_segment> slices,
  size_t offset,
  size_t size,
  uint8_t* device_dst,
  rmm::cuda_stream_view stream) noexcept
{
  return exec::make_semi_future<size_t>(std::make_exception_ptr(
    std::runtime_error("kvikio_context does not support host_to_device_read_async_io; use "
                       "device_read_async instead")));
}

exec::semi_future<size_t> kvikio_context::host_read_ranges_async_io(
  const sirius_io_object& obj, std::span<io_object_segment> segments) noexcept
{
  return exec::make_semi_future<size_t>(std::make_exception_ptr(
    std::runtime_error("kvikio_context does not support host_read_ranges_async_io")));
}

}  // namespace sirius::io
