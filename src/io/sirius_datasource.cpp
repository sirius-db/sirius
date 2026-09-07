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

#include <ctrack.hpp>
#include <fcntl.h>
#include <io/prefetch_census.hpp>
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

// Bridge a semi_future into a real (promise-backed) std::future. Promise,
// std::future, and the exact type-erased terminal are all built before
// producer() may publish IO, so callback installation is a move-only,
// non-allocating handoff.
template <typename Producer>
std::future<size_t> bridge_semi_to_std(Producer&& producer)
{
  auto p              = std::make_shared<std::promise<size_t>>();
  auto fut            = p->get_future();
  using terminal_type = exec::invocable<void(exec::try_t<size_t>&&) &&>;
  terminal_type terminal{[p = std::move(p)](exec::try_t<size_t>&& t) mutable {
    if (t.has_exception()) {
      p->set_exception(std::move(t).exception());
    } else {
      p->set_value(std::move(t).value());
    }
  }};
  auto sf = std::forward<Producer>(producer)();
  std::move(sf).install_callback(std::move(terminal));
  return fut;
}

}  // namespace

sirius_datasource::sirius_datasource(std::shared_ptr<ioctx> io_ctx,
                                     std::shared_ptr<io_object> io_obj)
  : _io_ctx(std::move(io_ctx)), _io_object(std::move(io_obj))
{
}

sirius_datasource::~sirius_datasource() {}

std::shared_ptr<io_object_metadata> sirius_datasource::metadata() const
{
  if (!_io_ctx || !_io_object) { return nullptr; }
  auto& cache = _io_ctx->metadata_store();
  return cache.get_metadata(*_io_object);
}

[[nodiscard]] bool sirius_datasource::store_metadata(std::shared_ptr<io_object_metadata> metadata)
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
  await_inflight_prefetch();
  if (uses_prefetching_cache()) {
    auto* cache = _io_ctx->cache();
    return cache->host_read(*_io_object, offset, size, dst, &_prefetch_handle);
  }
  return std::move(_io_ctx->host_read_async_io(*_io_object, offset, size, dst)).get();
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
  await_inflight_prefetch();
  return bridge_semi_to_std([&] {
    if (uses_prefetching_cache()) {
      auto* cache = _io_ctx->cache();
      return cache->host_read_async(*_io_object, offset, size, dst, &_prefetch_handle);
    }
    return _io_ctx->host_read_async_io(*_io_object, offset, size, dst);
  });
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
  CTRACK_NAME("ds::device_read_async");
  rmm::cuda_stream_view stream{stream_arg};
  await_inflight_prefetch();
  return bridge_semi_to_std([&] {
    if (uses_prefetching_cache()) {
      auto* cache = _io_ctx->cache();
      return cache->device_read_async(*_io_object, offset, size, dst, stream, &_prefetch_handle);
    }
    return _io_ctx->device_read_async_io(*_io_object, offset, size, dst, stream);
  });
}

std::future<size_t> sirius_datasource::device_read_ranges_async(std::span<const slice> ranges,
                                                                rmm::cuda_stream_view stream)
{
  await_inflight_prefetch();
  return bridge_semi_to_std([&] {
    if (uses_prefetching_cache()) {
      auto* cache = _io_ctx->cache();
      return cache->device_read_ranges_async(*_io_object, ranges, stream, &_prefetch_handle);
    }
    return _io_ctx->device_readv_async_io(*_io_object, ranges, stream);
  });
}

std::future<size_t> sirius_datasource::host_read_ranges_async(std::span<const slice> ranges)
{
  await_inflight_prefetch();
  return bridge_semi_to_std([&] {
    if (uses_prefetching_cache()) {
      auto* cache = _io_ctx->cache();
      return cache->host_read_ranges_async(*_io_object, ranges, &_prefetch_handle);
    }
    return _io_ctx->host_readv_async_io(*_io_object, ranges);
  });
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

  // The contract is "one scan, one datasource": a second inserting fadvise on
  // a datasource that already carries an active handle is a caller bug.  Warn
  // loudly and keep the in-flight request.  An inactive stale handle is
  // disposed by the move-assignment below.
  if (_prefetch_handle && _prefetch_handle.is_active()) {
    SIRIUS_LOG_WARN(
      "sirius_datasource::fadvise: a prefetching_handle was already stored on "
      "this datasource (path={}); cancelling the stale request.  Each scan "
      "should own a unique datasource.",
      _io_object->object_path());
    return;
  }

  // Hand the ranges to the cache.  It returns an empty handle when it didn't
  // enqueue any new work (dormant cache, every range coalesced with an existing
  // entry); we only stash a real handle.
  auto handle = cache->initiate_prefetching_request(*_io_object, ranges, dev_id);
  if (handle) { _prefetch_handle = std::move(handle); }
}

void sirius_datasource::update(cache::scan_stage site)
{
  if (!_prefetch_handle) { return; }
  _prefetch_handle.update(site);
}

void sirius_datasource::await_inflight_prefetch() noexcept
{
  if (!_prefetch_handle || !_prefetch_handle.is_prefetch_in_flight()) { return; }
  CTRACK_NAME("ds::await_inflight_prefetch(blocked)");
  // The readahead already has this split's IO in flight.  Reading now would
  // find every chunk `loading`, miss, and re-read the same bytes through a
  // bounce buffer — the one thing the prefetch exists to avoid.  Waiting costs
  // this thread the remainder of an IO that is already running; the read then
  // serves from cache.
  std::ignore = _prefetch_handle.wait_until_ready();
}

prepare_result sirius_datasource::prepare_prefetch(bool wait_for_eviction)
{
  if (!_prefetch_handle || !uses_prefetching_cache()) { return prepare_result::nothing_to_prepare; }
  auto* cache = _io_ctx->cache();
  if (cache == nullptr) { return prepare_result::nothing_to_prepare; }
  return cache->prepare(_prefetch_handle, wait_for_eviction) ? prepare_result::prepared
                                                             : prepare_result::allocation_failed;
}

prefetch_refusal sirius_datasource::prefetch_async(exec::invocable<void(bool) noexcept> on_done)
{
  if (!_prefetch_handle || !uses_prefetching_cache()) {
    on_done(false);
    return prefetch_refusal::no_cache;
  }

  if (_prefetch_handle.has_started_reading()) {
    prefetch_census::instance().declined_reading.fetch_add(1, std::memory_order_relaxed);
    on_done(false);
    return prefetch_refusal::consumer_ahead;
  }

  auto const producer = _prefetch_handle.producer_state();
  if (producer == cache::producer_stage::abandoned) {
    on_done(false);
    return prefetch_refusal::memory_pressure;
  }
  if (producer < cache::producer_stage::prepared) {
    on_done(false);
    return prefetch_refusal::other;
  }
  if (_io_ctx->cache()->prefetch(_prefetch_handle, std::move(on_done))) {
    return prefetch_refusal::issued;
  }

  return _prefetch_handle.has_started_reading() ? prefetch_refusal::consumer_ahead
                                                : prefetch_refusal::other;
}

bool sirius_datasource::uses_prefetching_cache() const noexcept
{
  return _io_ctx->uses_prefetching_cache();
}

bool sirius_datasource::prefers_bulk_io() const noexcept { return _io_ctx->prefers_bulk_io(); }

}  // namespace sirius::io
