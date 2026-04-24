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

#include "io/sirius_datasource.hpp"

#include "io/prefetching_cache.hpp"
#include "io/types.hpp"

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>

#include <algorithm>
#include <ranges>
#include <stdexcept>

namespace sirius::io {

// ---------------------------------------------------------------------------
// sirius_ioctx
// ---------------------------------------------------------------------------

sirius_ioctx::sirius_ioctx()  = default;
sirius_ioctx::~sirius_ioctx() = default;

void sirius_ioctx::initialize_cache(buffer_pool& pool, size_t inflight_budget_chunks)
{
  _cache = std::make_unique<prefetching_cache>(pool, this, inflight_budget_chunks);
}

namespace {

// Copy each pinned-host slice to the device buffer on @p stream.
// Returns the total bytes issued (== sum of slice sizes).
size_t copy_pinned_slices_to_device(
  std::vector<cudf::io::datasource::non_owning_buffer> const& slices,
  uint8_t* dst,
  rmm::cuda_stream_view stream)
{
  // Skip empty slices without touching CUDA.
  size_t n_nonempty = 0;
  for (auto const& s : slices)
    if (s.size() > 0) ++n_nonempty;

  if (n_nonempty == 0) return 0;

  // Fast path: one slice (common after pinned_view::slice coalescing when
  // chunks are contiguous in slab memory).  Plain cudaMemcpyAsync avoids the
  // batch-API per-call overhead.
  if (n_nonempty == 1) {
    size_t copied = 0;
    for (auto const& s : slices) {
      if (s.size() == 0) continue;
      auto err =
        cudaMemcpyAsync(dst + copied, s.data(), s.size(), cudaMemcpyHostToDevice, stream.value());
      if (err != cudaSuccess)
        throw std::runtime_error(std::string("sirius_ioctx: cudaMemcpyAsync failed: ") +
                                 cudaGetErrorString(err));
      copied += s.size();
    }
    return copied;
  }

  // Batch path: hand all non-contiguous slices to the driver in one call.
  // Copies within a batch are unordered with respect to each other but the
  // whole batch is stream-ordered — fine here, all copies have disjoint
  // destination ranges.
  std::vector<void*> dsts;
  std::vector<void const*> srcs;
  std::vector<size_t> sizes;
  dsts.reserve(n_nonempty);
  srcs.reserve(n_nonempty);
  sizes.reserve(n_nonempty);

  size_t copied = 0;
  for (auto const& s : slices) {
    auto n = s.size();
    if (n == 0) continue;
    dsts.push_back(dst + copied);
    srcs.push_back(s.data());
    sizes.push_back(n);
    copied += n;
  }

  cudaMemcpyAttributes attrs{};
  attrs.srcAccessOrder  = cudaMemcpySrcAccessOrderStream;
  attrs.srcLocHint.type = cudaMemLocationTypeHost;
  attrs.dstLocHint.type = cudaMemLocationTypeDevice;
  attrs.flags           = 0;
  size_t attrs_idx      = 0;
  size_t fail_idx       = 0;

  auto err = cudaMemcpyBatchAsync(
    dsts.data(), srcs.data(), sizes.data(), n_nonempty, &attrs, &attrs_idx, 1, stream.value());
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("sirius_ioctx: cudaMemcpyBatchAsync failed at idx ") +
                             std::to_string(fail_idx) + ": " + cudaGetErrorString(err));
  return copied;
}

}  // namespace

size_t sirius_ioctx::device_read(
  sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream)
{
  if (_cache) {
    if (auto view = _cache->read(obj, offset, size, stream.value()); view) {
      auto slices = view.slice(offset, size);
      return copy_pinned_slices_to_device(slices, dst, stream);
    }
  }
  return device_read_io(obj, offset, size, dst, stream);
}

std::unique_ptr<cudf::io::datasource::buffer> sirius_ioctx::device_read(
  sirius_io_object& obj, size_t offset, size_t size, rmm::cuda_stream_view stream)
{
  if (_cache) {
    if (auto view = _cache->read(obj, offset, size, stream.value()); view) {
      auto slices = view.slice(offset, size);
      rmm::device_buffer dbuf(size, stream);
      copy_pinned_slices_to_device(slices, static_cast<uint8_t*>(dbuf.data()), stream);
      return cudf::io::datasource::buffer::create(std::move(dbuf));
    }
  }
  return device_read_io(obj, offset, size, stream);
}

void sirius_ioctx::device_read_async(sirius_io_object& obj,
                                     size_t offset,
                                     size_t size,
                                     uint8_t* dst,
                                     rmm::cuda_stream_view stream,
                                     io_completion_handler handler)
{
  if (_cache) {
    if (auto view = _cache->read(obj, offset, size, stream.value()); view) {
      auto slices = view.slice(offset, size);
      try {
        auto copied = copy_pinned_slices_to_device(slices, dst, stream);
        handler(copied, nullptr);
      } catch (...) {
        handler(0, std::current_exception());
      }
      return;
    }
  }
  device_read_io_async(obj, offset, size, dst, stream, std::move(handler));
}

// ---------------------------------------------------------------------------
// sirius_datasource
// ---------------------------------------------------------------------------

sirius_datasource::sirius_datasource(std::shared_ptr<sirius_ioctx> io_ctx,
                                     std::shared_ptr<sirius_io_object> io_object)
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
  auto p = std::make_shared<std::promise<size_t>>();
  auto f = p->get_future();
  _io_ctx->host_read_async(*_io_object, offset, size, dst, [p](size_t n, std::exception_ptr ep) {
    if (ep)
      p->set_exception(ep);
    else
      p->set_value(n);
  });
  return f;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>> sirius_datasource::host_read_async(
  size_t offset, size_t size)
{
  size     = std::min(size, _io_object->size() > offset ? _io_object->size() - offset : size_t{0});
  auto buf = std::make_shared<std::vector<uint8_t>>(size);
  auto p   = std::make_shared<std::promise<std::unique_ptr<datasource::buffer>>>();
  auto f   = p->get_future();
  _io_ctx->host_read_async(
    *_io_object, offset, size, buf->data(), [p, buf](size_t n, std::exception_ptr ep) {
      if (ep) {
        p->set_exception(ep);
        return;
      }
      buf->resize(n);
      p->set_value(datasource::buffer::create(std::move(*buf)));
    });
  return f;
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
  auto p = std::make_shared<std::promise<size_t>>();
  auto f = p->get_future();
  _io_ctx->device_read_async(
    *_io_object, offset, size, dst, stream, [p](size_t n, std::exception_ptr ep) {
      if (ep)
        p->set_exception(ep);
      else
        p->set_value(n);
    });
  return f;
}

void sirius_datasource::host_read_ranges_async(
  std::vector<cudf::io::text::byte_range_info> const& ranges,
  std::span<cudf::host_span<std::byte>> dst,
  io_completion_handler handler)
{
  _io_ctx->host_read_ranges_async(*_io_object, ranges, dst, std::move(handler));
}

size_t sirius_datasource::host_read_ranges(
  std::vector<cudf::io::text::byte_range_info> const& ranges,
  std::span<cudf::host_span<std::byte>> dst)
{
  return _io_ctx->host_read_ranges(*_io_object, ranges, dst);
}

}  // namespace sirius::io
