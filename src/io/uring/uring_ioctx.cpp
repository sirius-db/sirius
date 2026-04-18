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

#include "io/uring/uring_ioctx.hpp"

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
// ring_pool
// ---------------------------------------------------------------------------

void ring_pool::init(size_t n_rings, unsigned ring_entries)
{
  _rings  = std::make_unique<io_uring[]>(n_rings);
  _in_use = std::make_unique<bool[]>(n_rings);
  _n      = n_rings;

  for (auto i : std::views::iota(size_t{0}, _n)) {
    int ret = io_uring_queue_init(ring_entries, &_rings[i], 0);
    if (ret < 0)
      throw std::runtime_error("ring_pool: io_uring_queue_init failed: " +
                               std::string(strerror(-ret)));
  }
}

ring_pool::~ring_pool()
{
  std::for_each_n(_rings.get(), _n, [](io_uring& r) { io_uring_queue_exit(&r); });
}

ring_pool::guard ring_pool::acquire()
{
  std::unique_lock lk{_mtx};
  size_t found = _n;
  _cv.wait(lk, [&] {
    auto* first = _in_use.get();
    auto* it    = std::find(first, first + _n, false);
    if (it == first + _n) return false;
    *it   = true;
    found = static_cast<size_t>(it - first);
    return true;
  });
  return guard{this, found};
}

void ring_pool::release(size_t idx)
{
  {
    std::lock_guard lk{_mtx};
    _in_use[idx] = false;
  }
  _cv.notify_one();
}

// ---------------------------------------------------------------------------
// uring_io_object
// ---------------------------------------------------------------------------

uring_io_object::uring_io_object(std::string path) : _path(std::move(path))
{
  _fd = file_descriptor(::open(_path.c_str(), O_RDONLY));
  if (!_fd)
    throw std::runtime_error("uring_io_object: open failed: " + _path + ": " + strerror(errno));

  struct stat st{};
  if (::fstat(_fd.get(), &st) < 0)
    throw std::runtime_error("uring_io_object: fstat failed: " + std::string(strerror(errno)));
  _file_size = static_cast<size_t>(st.st_size);

  _fd_direct = file_descriptor(::open(_path.c_str(), O_RDONLY | O_DIRECT));
  if (!_fd_direct)
    throw std::runtime_error("uring_io_object: O_DIRECT open failed: " + _path + ": " +
                             strerror(errno));
}

// ---------------------------------------------------------------------------
// uring_ioctx
// ---------------------------------------------------------------------------

uring_ioctx::uring_ioctx(unsigned host_ring_depth,
                         unsigned ring_entries,
                         size_t n_reactors,
                         size_t bounce_slot_size)
{
  _host_pool.init(host_ring_depth, ring_entries);
  _reactors.reserve(n_reactors);
  std::generate_n(std::back_inserter(_reactors), n_reactors, [&] {
    return std::make_unique<uring_reactor>(ring_entries, bounce_slot_size);
  });
  _ring_entries = ring_entries;
}

void uring_ioctx::shutdown()
{
  std::ranges::for_each(_reactors, [](auto& r) { r->shutdown(); });
}

ring_pool::guard uring_ioctx::acquire_host_ring() { return _host_pool.acquire(); }

uring_reactor& uring_ioctx::assign_reactor()
{
  size_t idx = _next.fetch_add(1, std::memory_order_relaxed) % _reactors.size();
  return *_reactors.at(idx);
}

std::unique_ptr<cudf::io::datasource> uring_ioctx::make_datasource(
  std::unique_ptr<sirius_io_object> io_object)
{
  auto* uobj = dynamic_cast<uring_io_object*>(io_object.get());
  if (uobj) uobj->set_reactor(&assign_reactor());
  return std::make_unique<sirius_datasource>(shared_from_this(), std::move(io_object));
}

uring_io_object& uring_ioctx::as_uring(sirius_io_object& obj)
{
  auto* p = dynamic_cast<uring_io_object*>(&obj);
  if (!p) throw std::runtime_error("uring_ioctx: io_object is not a uring_io_object");
  return *p;
}

// -- uring_ioctx: host reads --------------------------------------------------

size_t uring_ioctx::host_read(sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst)
{
  auto& uobj = as_uring(obj);
  size       = std::min(size, uobj.size() > offset ? uobj.size() - offset : 0UL);
  if (size == 0) return 0;
  ssize_t n = ::pread(uobj.fd(), dst, size, static_cast<off_t>(offset));
  if (n < 0) throw std::runtime_error("host_read pread: " + std::string(strerror(errno)));
  return static_cast<size_t>(n);
}

std::unique_ptr<cudf::io::datasource::buffer> uring_ioctx::host_read(sirius_io_object& obj,
                                                                     size_t offset,
                                                                     size_t size)
{
  auto& uobj = as_uring(obj);
  spdlog::debug("host_read(buf)  file={} offset={} size={:.2f}MB cursor {}",
                uobj.raw_file_cache_id(),
                offset,
                to_mb(size),
                offset + size);
  size = std::min(size, uobj.size() > offset ? uobj.size() - offset : 0UL);
  std::vector<uint8_t> buf(size);
  size_t n = host_read(obj, offset, size, buf.data());
  buf.resize(n);
  return cudf::io::datasource::buffer::create(std::move(buf));
}

std::future<size_t> uring_ioctx::host_read_async(sirius_io_object& obj,
                                                 size_t offset,
                                                 size_t size,
                                                 uint8_t* dst)
{
  spdlog::debug("host_read_async(dst)  file={} offset={} size={:.2f}MB cursor {}",
                obj.raw_file_cache_id(),
                offset,
                to_mb(size),
                offset + size);
  return std::async(std::launch::async, [this, &obj, offset, size, dst]() mutable {
    return host_read(obj, offset, size, dst);
  });
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>> uring_ioctx::host_read_async(
  sirius_io_object& obj, size_t offset, size_t size)
{
  spdlog::debug("host_read_async(buf)  file={} offset={} size={:.2f}MB cursor {}",
                obj.raw_file_cache_id(),
                offset,
                to_mb(size),
                offset + size);
  return std::async(
    std::launch::async,
    [this, &obj, offset, size]() mutable -> std::unique_ptr<cudf::io::datasource::buffer> {
      return host_read(obj, offset, size);
    });
}

// -- uring_ioctx: device reads ------------------------------------------------

std::unique_ptr<cudf::io::datasource::buffer> uring_ioctx::device_read(sirius_io_object& obj,
                                                                       size_t offset,
                                                                       size_t size,
                                                                       rmm::cuda_stream_view stream)
{
  auto& uobj = as_uring(obj);
  spdlog::debug("device_read(buf)  file={} offset={} size={:.2f}MB cursor {}",
                uobj.raw_file_cache_id(),
                offset,
                to_mb(size),
                offset + size);
  rmm::device_buffer dbuf(size, stream);
  enqueue_device_read(uobj, offset, size, static_cast<uint8_t*>(dbuf.data()), stream.value()).get();
  return cudf::io::datasource::buffer::create(std::move(dbuf));
}

size_t uring_ioctx::device_read(
  sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream)
{
  auto& uobj = as_uring(obj);
  spdlog::debug("device_read(dst)  file={} offset={} size={:.2f}MB cursor {}",
                uobj.raw_file_cache_id(),
                offset,
                to_mb(size),
                offset + size);
  return enqueue_device_read(uobj, offset, size, dst, stream.value()).get();
}

std::future<size_t> uring_ioctx::device_read_async(
  sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream)
{
  auto& uobj = as_uring(obj);
  spdlog::debug("device_read_async file={} offset={} size={:.2f}MB cursor {}",
                uobj.raw_file_cache_id(),
                offset,
                to_mb(size),
                offset + size);
  return enqueue_device_read(uobj, offset, size, dst, stream.value());
}

std::future<size_t> uring_ioctx::enqueue_device_read(
  uring_io_object& uobj, size_t offset, size_t size, uint8_t* dst, cudaStream_t stream)
{
  auto file_size = uobj.size();
  if (size == 0 || offset >= file_size) {
    std::promise<size_t> p;
    p.set_value(0);
    return p.get_future();
  }
  size = std::min(size, file_size - offset);

  size_t a_start = offset & ~(IO_BLOCK_SIZE - 1);
  size_t a_end   = std::min((offset + size + IO_BLOCK_SIZE - 1) & ~(IO_BLOCK_SIZE - 1),
                          (file_size + IO_BLOCK_SIZE - 1) & ~(IO_BLOCK_SIZE - 1));
  size_t prefix  = offset - a_start;

  size_t n_chunks = (a_end - a_start + CHUNK_SIZE - 1) / CHUNK_SIZE;

  auto ctx         = std::make_shared<request_context>();
  ctx->total_bytes = size;
  ctx->pending.store(n_chunks, std::memory_order_relaxed);
  std::future<size_t> fut = ctx->promise.get_future();

  auto* reactor   = uobj.reactor();
  size_t produced = 0;
  for (size_t cur = a_start; cur < a_end; cur += CHUNK_SIZE) {
    device_read_req req;
    req.fd_direct = uobj.fd_direct();
    req.file_off  = cur;
    req.io_size   = std::min(CHUNK_SIZE, a_end - cur);
    req.data_off  = (cur == a_start) ? prefix : 0;
    req.data_size = std::min(req.io_size - req.data_off, size - produced);
    req.dst       = dst + produced;
    req.stream    = stream;
    req.ctx       = ctx;
    produced += req.data_size;
    reactor->enqueue(std::move(req));
  }

  return fut;
}

// -- uring_ioctx: batch host reads --------------------------------------------

std::future<size_t> uring_ioctx::host_read_ranges_async(
  sirius_io_object& obj,
  std::vector<cudf::io::text::byte_range_info> const& ranges,
  std::span<cudf::host_span<std::byte>> dst)
{
  auto& uobj     = as_uring(obj);
  auto file_size = uobj.size();

  if (ranges.empty()) {
    std::promise<size_t> p;
    p.set_value(0);
    return p.get_future();
  }

  size_t total    = 0;
  size_t n_active = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    auto off  = static_cast<size_t>(ranges[i].offset());
    size_t sz = std::min(static_cast<size_t>(ranges[i].size()),
                         file_size > off ? file_size - off : size_t{0});
    if (sz > 0 && sz <= dst[i].size()) {
      total += sz;
      ++n_active;
    }
  }

  if (n_active == 0) {
    std::promise<size_t> p;
    p.set_value(0);
    return p.get_future();
  }

  auto ctx         = std::make_shared<request_context>();
  ctx->total_bytes = total;
  ctx->pending.store(n_active, std::memory_order_relaxed);

  auto* reactor = uobj.reactor();
  for (size_t i = 0; i < ranges.size(); ++i) {
    auto off  = static_cast<size_t>(ranges[i].offset());
    size_t sz = std::min(static_cast<size_t>(ranges[i].size()),
                         file_size > off ? file_size - off : size_t{0});
    if (sz == 0 || sz > dst[i].size()) continue;
    host_read_req req;
    req.fd     = uobj.fd();
    req.offset = off;
    req.size   = sz;
    req.dst    = reinterpret_cast<uint8_t*>(dst[i].data());
    req.ctx    = ctx;
    reactor->enqueue_host(std::move(req));
  }

  return ctx->promise.get_future();
}

size_t uring_ioctx::host_read_ranges(sirius_io_object& obj,
                                     std::vector<cudf::io::text::byte_range_info> const& ranges,
                                     std::span<cudf::host_span<std::byte>> dst)
{
  return host_read_ranges_async(obj, ranges, dst).get();
}

}  // namespace sirius::io
