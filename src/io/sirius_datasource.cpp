/*
 * Copyright 2025, Sirius Contributors.
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

#include "io/sirius_datasource.hpp"

#include "io/types.hpp"

#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>

#include <algorithm>
#include <ranges>
#include <stdexcept>

namespace sirius::io {

// ---------------------------------------------------------------------------
// sirius_datasource
// ---------------------------------------------------------------------------

sirius_datasource::sirius_datasource(std::shared_ptr<sirius_ioctx> io_ctx,
                                     std::unique_ptr<sirius_io_object> io_object)
  : _io_ctx(std::move(io_ctx)), _io_object(std::move(io_object))
{
}

size_t sirius_datasource::size() const { return _io_object->size(); }

bool sirius_datasource::supports_device_read() const { return true; }

bool sirius_datasource::is_device_read_preferred(size_t) const { return true; }

size_t sirius_datasource::host_read(size_t offset, size_t size, uint8_t* dst)
{
  return _io_ctx->host_read(*_io_object, offset, size, dst);
}

std::unique_ptr<cudf::io::datasource::buffer> sirius_datasource::host_read(size_t offset,
                                                                           size_t size)
{
  return _io_ctx->host_read(*_io_object, offset, size);
}

std::future<size_t> sirius_datasource::host_read_async(size_t offset, size_t size, uint8_t* dst)
{
  return _io_ctx->host_read_async(*_io_object, offset, size, dst);
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>> sirius_datasource::host_read_async(
  size_t offset, size_t size)
{
  return _io_ctx->host_read_async(*_io_object, offset, size);
}

std::unique_ptr<cudf::io::datasource::buffer> sirius_datasource::device_read(
  size_t offset, size_t size, rmm::cuda_stream_view stream)
{
  return _io_ctx->device_read(*_io_object, offset, size, stream);
}

size_t sirius_datasource::device_read(size_t offset,
                                      size_t size,
                                      uint8_t* dst,
                                      rmm::cuda_stream_view stream)
{
  return _io_ctx->device_read(*_io_object, offset, size, dst, stream);
}

std::future<size_t> sirius_datasource::device_read_async(size_t offset,
                                                         size_t size,
                                                         uint8_t* dst,
                                                         rmm::cuda_stream_view stream)
{
  return _io_ctx->device_read_async(*_io_object, offset, size, dst, stream);
}

std::future<size_t> sirius_datasource::host_read_ranges_async(
  std::vector<cudf::io::text::byte_range_info> const& ranges,
  std::span<cudf::host_span<std::byte>> dst)
{
  return _io_ctx->host_read_ranges_async(*_io_object, ranges, dst);
}

size_t sirius_datasource::host_read_ranges(
  std::vector<cudf::io::text::byte_range_info> const& ranges,
  std::span<cudf::host_span<std::byte>> dst)
{
  return _io_ctx->host_read_ranges(*_io_object, ranges, dst);
}

}  // namespace sirius::io
