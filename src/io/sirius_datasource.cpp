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

#include "exec/semi_future.hpp"
#include "exec/try.hpp"
#include "io/cache/prefetching_cache.hpp"

#include <rmm/device_buffer.hpp>

#include <fcntl.h>
#include <log/logging.hpp>
#include <sys/stat.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::io {

namespace {

// Bridge a semi_future into a real (promise-backed) std::future.  The result is
// pushed in via install_callback when the IO settles, so the std::future
// reports readiness normally (wait_for never returns `deferred`) and the
// completion bookkeeping runs on the IO callback thread rather than being
// pulled onto the caller's thread the way std::async(deferred) would.
std::future<size_t> bridge_semi_to_std(exec::semi_future<size_t>&& sf)
{
  auto p   = std::make_shared<std::promise<size_t>>();
  auto fut = p->get_future();
  std::move(sf).install_callback([p = std::move(p)](exec::try_t<size_t>&& t) mutable {
    if (t.has_exception()) {
      p->set_exception(std::move(t).exception());
    } else {
      p->set_value(std::move(t).value());
    }
  });
  return fut;
}

}  // namespace

sirius_datasource::sirius_datasource(std::shared_ptr<sirius_ioctx> io_ctx,
                                     std::shared_ptr<sirius_io_object> io_object)
  : _io_ctx(std::move(io_ctx)), _io_object(std::move(io_object))
{
}

sirius_datasource::~sirius_datasource() {}

std::shared_ptr<sirius_io_object_metadata> sirius_datasource::metadata() const
{
  if (!_io_ctx || !_io_object) { return nullptr; }
  auto& cache = _io_ctx->metadata_store();
  return cache.get_metadata(*_io_object);
}

[[nodiscard]] bool sirius_datasource::store_metadata(
  std::shared_ptr<sirius_io_object_metadata> metadata)
{
  if (!_io_ctx || !_io_object) { return false; }
  auto& cache = _io_ctx->metadata_store();
  cache.register_metadata(*_io_object, std::move(metadata));
  return true;
}

size_t sirius_datasource::size() const { return _io_object->size(); }

bool sirius_datasource::supports_device_read() const { return _io_ctx->supports_device_read(); }

bool sirius_datasource::supports_vector_host_read() const
{
  return _io_ctx->supports_vector_host_read();
}

bool sirius_datasource::is_device_read_preferred(size_t) const
{
  return _io_ctx->supports_device_read();
}

size_t sirius_datasource::host_read(size_t offset, size_t size, uint8_t* dst)
{
  if (uses_prefetching_cache()) {
    auto* cache = _io_ctx->cache();
    return cache->host_read(*_io_object, offset, size, dst, &_prefetch_handle);
  }
  return _io_ctx->host_read_io(*_io_object, offset, size, dst);
}

std::unique_ptr<cudf::io::datasource::buffer> sirius_datasource::host_read(size_t offset,
                                                                           size_t size)
{
  std::vector<uint8_t> buf(size);
  auto n = host_read(offset, size, buf.data());
  buf.resize(n);
  return cudf::io::datasource::buffer::create(std::move(buf));
}

std::future<size_t> sirius_datasource::host_read_async(size_t offset, size_t size, uint8_t* dst)
{
  exec::semi_future<size_t> semi;
  if (uses_prefetching_cache()) {
    auto* cache = _io_ctx->cache();
    semi        = cache->host_read_async(*_io_object, offset, size, dst, &_prefetch_handle);
  } else {
    semi = _io_ctx->host_read_async_io(*_io_object, offset, size, dst);
  }
  return bridge_semi_to_std(std::move(semi));
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>> sirius_datasource::host_read_async(
  size_t offset, size_t size)
{
  auto file_size = _io_object->size();
  size           = std::min(size, file_size > offset ? file_size - offset : size_t{0});
  auto buf       = std::vector<uint8_t>(size);
  auto fut       = host_read_async(offset, size, buf.data());
  return std::async(std::launch::deferred, [s = std::move(fut), buf = std::move(buf)]() mutable {
    auto n = s.get();
    buf.resize(n);
    return cudf::io::datasource::buffer::create(std::move(buf));
  });
}

std::unique_ptr<cudf::io::datasource::buffer> sirius_datasource::device_read(
  size_t offset, size_t size, cudf_datasource_stream_t stream_arg)
{
  rmm::cuda_stream_view stream{stream_arg};
  rmm::device_buffer buf(size, stream);
  auto n = device_read(offset, size, reinterpret_cast<uint8_t*>(buf.data()), stream);
  n      = std::min(n, size);
  buf.resize(n, stream);
  return cudf::io::datasource::buffer::create(std::move(buf));
}

size_t sirius_datasource::device_read(size_t offset,
                                      size_t size,
                                      uint8_t* dst,
                                      cudf_datasource_stream_t stream_arg)
{
  rmm::cuda_stream_view stream{stream_arg};
  auto f = device_read_async(offset, size, dst, stream);
  auto n = f.get();
  stream.synchronize();
  return n;
}

std::future<size_t> sirius_datasource::device_read_async(size_t offset,
                                                         size_t size,
                                                         uint8_t* dst,
                                                         cudf_datasource_stream_t stream_arg)
{
  rmm::cuda_stream_view stream{stream_arg};
  exec::semi_future<size_t> semi;
  if (uses_prefetching_cache()) {
    auto* cache = _io_ctx->cache();
    semi = cache->device_read_async(*_io_object, offset, size, dst, stream, &_prefetch_handle);
  } else {
    semi = _io_ctx->device_read_async_io(*_io_object, offset, size, dst, stream);
  }
  return bridge_semi_to_std(std::move(semi));
}

std::unique_ptr<sirius_datasource> sirius_datasource::duplicate() const
{
  // Share the io_ctx and io_object — both are shared_ptr-managed and
  // deliberately reused across splits of the same file.  The new
  // datasource starts with a default-constructed prefetching_handle so
  // its fadvise() calls can't accidentally cancel the original's work.
  return std::make_unique<sirius_datasource>(_io_ctx, _io_object);
}

void sirius_datasource::fadvise(std::span<const cudf::io::text::byte_range_info> ranges,
                                std::optional<int> dev_id)
{
  auto* cache = _io_ctx->cache();
  if (cache == nullptr || !_io_ctx->can_use_prefetching_cache()) { return; }

  // The contract is "one scan, one datasource": a second
  // speculative/immediate fadvise on a datasource that already carries a
  // handle is a caller bug.  Warn loudly; cancel the stale handle so the
  // worker drops the old request and we don't leak both into the cache.
  if (_prefetch_handle) {
    if (_prefetch_handle.is_active()) {
      SIRIUS_LOG_WARN(
        "sirius_datasource::fadvise: a prefetching_handle was already stored on "
        "this datasource (path={}); cancelling the stale request.  Each scan "
        "should own a unique datasource.",
        _io_object->object_path());
      return;
    }
    _prefetch_handle.cancel();
  }

  // Hand the ranges to the cache.  insert() returns an empty handle when
  // it didn't enqueue any new work (dormant cache, every range coalesced
  // with an existing entry); we only stash a real handle.
  auto handle = cache->insert(*_io_object, ranges, dev_id);
  if (handle) { _prefetch_handle = std::move(handle); }
}

void sirius_datasource::prefetch(cache::prefetching_stage site)
{
  auto const preferred = _io_ctx->preferred_prefetching_stage();
  if (preferred == cache::prefetching_stage::none) { return; }
  if (_prefetch_handle) {
    if (site == cache::prefetching_stage::disposable) {
      _prefetch_handle.cancel();
    } else if (site == preferred) {
      _prefetch_handle.activate();
    }
  }
}

bool sirius_datasource::uses_prefetching_cache() const noexcept
{
  return _io_ctx->uses_prefetching_cache();
}

}  // namespace sirius::io
