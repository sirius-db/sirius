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

#include "io/s3/s3_ioctx.hpp"

#include "io/s3/credential_provider.hpp"
#include "io/s3/s3_io_object.hpp"
#include "io/sirius_datasource.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <curl/curl.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <future>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::io::s3 {

namespace {

// --- RAII for curl_global_init / curl_global_cleanup -----------------------
// We init once per process on first s3_ioctx construction. curl_global_cleanup
// is intentionally not called: it races with late-destroyed static curl
// handles in third-party libs and AWS SDK has hit this too. The leak is bounded.
struct curl_global_init_once {
  curl_global_init_once()
  {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != 0)
      throw std::runtime_error("s3_ioctx: curl_global_init failed");
  }
};
void ensure_curl_inited()
{
  static curl_global_init_once inst;
  (void)inst;
}

// libcurl write callback appending bytes to a std::string* buffer.
std::size_t curl_write_string(void* ptr, std::size_t size, std::size_t nmemb, void* userdata)
{
  auto* s     = static_cast<std::string*>(userdata);
  auto nbytes = size * nmemb;
  s->append(static_cast<char const*>(ptr), nbytes);
  return nbytes;
}

// libcurl write callback writing into a caller-supplied flat buffer.
struct buf_sink {
  std::uint8_t* dst;
  std::size_t capacity;
  std::size_t written;
};
std::size_t curl_write_buf(void* ptr, std::size_t size, std::size_t nmemb, void* userdata)
{
  auto* sink  = static_cast<buf_sink*>(userdata);
  auto nbytes = size * nmemb;
  if (sink->written + nbytes > sink->capacity) {
    // Server returned more than requested; signal error via short write.
    return 0;
  }
  std::memcpy(sink->dst + sink->written, ptr, nbytes);
  sink->written += nbytes;
  return nbytes;
}

// libcurl header callback that snags the @c Content-Range value (if present).
// We use it to verify a 206 response actually covers the byte range we asked
// for, rather than trusting the server to honor Range silently.
struct header_capture {
  std::string content_range;
};
std::size_t curl_header_capture(char* buffer, std::size_t size, std::size_t nmemb, void* userdata)
{
  auto* hc          = static_cast<header_capture*>(userdata);
  std::size_t total = size * nmemb;
  std::string_view l(buffer, total);
  // Header lines arrive as "Name: value\r\n". Match Content-Range
  // case-insensitively (HTTP header names are case-insensitive).
  static constexpr std::string_view kPrefix = "content-range:";
  if (l.size() >= kPrefix.size()) {
    bool match = true;
    for (std::size_t i = 0; i < kPrefix.size(); ++i) {
      if (std::tolower(static_cast<unsigned char>(l[i])) !=
          static_cast<unsigned char>(kPrefix[i])) {
        match = false;
        break;
      }
    }
    if (match) {
      auto v = l.substr(kPrefix.size());
      while (!v.empty() && (v.front() == ' ' || v.front() == '\t'))
        v.remove_prefix(1);
      while (!v.empty() &&
             (v.back() == '\r' || v.back() == '\n' || v.back() == ' ' || v.back() == '\t'))
        v.remove_suffix(1);
      hc->content_range = std::string(v);
    }
  }
  return total;
}

// Parse "bytes <start>-<end>/<total>" (or "*"). Returns false if malformed.
bool parse_content_range(std::string_view v, std::size_t& start, std::size_t& end)
{
  static constexpr std::string_view kPrefix = "bytes ";
  if (v.size() <= kPrefix.size() || v.substr(0, kPrefix.size()) != kPrefix) return false;
  v.remove_prefix(kPrefix.size());
  auto dash = v.find('-');
  auto sl   = v.find('/');
  if (dash == std::string_view::npos || sl == std::string_view::npos || dash >= sl) return false;
  try {
    start = std::stoull(std::string{v.substr(0, dash)});
    end   = std::stoull(std::string{v.substr(dash + 1, sl - dash - 1)});
  } catch (...) {
    return false;
  }
  return start <= end;
}

}  // namespace

// ===========================================================================
// handle_slot
// ===========================================================================

void s3_ioctx::handle_slot::reset()
{
  if (owner && easy) { owner->release_handle(handle_slot{owner, easy}); }
  owner = nullptr;
  easy  = nullptr;
}

// ===========================================================================
// s3_ioctx lifecycle
// ===========================================================================

s3_ioctx::s3_ioctx(s3_ioctx_config config) : _cfg(std::move(config))
{
  if (!_cfg.creds) throw std::invalid_argument("s3_ioctx: credential_provider is required");
  if (_cfg.max_connections == 0) _cfg.max_connections = 1;

  ensure_curl_inited();
}

s3_ioctx::~s3_ioctx() { shutdown(); }

void s3_ioctx::shutdown()
{
  std::vector<void*> to_free;
  {
    std::lock_guard lk{_pool_mtx};
    if (_shutdown) return;
    _shutdown = true;
    to_free   = std::move(_free_handles);
    _free_handles.clear();
  }
  _pool_cv.notify_all();
  for (auto* h : to_free)
    curl_easy_cleanup(static_cast<CURL*>(h));
}

std::unique_ptr<cudf::io::datasource> s3_ioctx::make_datasource(
  std::shared_ptr<sirius_io_object> io_object)
{
  return std::make_unique<sirius_datasource>(shared_from_this(), std::move(io_object));
}

// ===========================================================================
// handle pool
// ===========================================================================

s3_ioctx::handle_slot s3_ioctx::acquire_handle()
{
  std::unique_lock lk{_pool_mtx};
  _pool_cv.wait(lk, [&] {
    return _shutdown || !_free_handles.empty() || _total_handles < _cfg.max_connections;
  });
  if (_shutdown) throw std::runtime_error("s3_ioctx: acquire_handle after shutdown");

  if (!_free_handles.empty()) {
    void* h = _free_handles.back();
    _free_handles.pop_back();
    return handle_slot{this, h};
  }
  // Grow the pool on demand up to max_connections.
  CURL* h = curl_easy_init();
  if (!h) throw std::runtime_error("s3_ioctx: curl_easy_init failed");
  ++_total_handles;
  return handle_slot{this, h};
}

void s3_ioctx::release_handle(handle_slot slot)
{
  void* h    = slot.easy;
  slot.easy  = nullptr;
  slot.owner = nullptr;
  if (!h) return;
  // Reset reusable options so the next borrower starts clean.
  curl_easy_reset(static_cast<CURL*>(h));
  {
    std::lock_guard lk{_pool_mtx};
    if (_shutdown) {
      curl_easy_cleanup(static_cast<CURL*>(h));
      return;
    }
    _free_handles.push_back(h);
  }
  _pool_cv.notify_one();
}

// ===========================================================================
// HEAD / range GET
// ===========================================================================

std::size_t s3_ioctx::head_object_size(std::string_view bucket, std::string_view key)
{
  auto slot = acquire_handle();
  auto* h   = static_cast<CURL*>(slot.easy);

  // Auth lives in the URL's query string (X-Amz-Signature etc.); no
  // Authorization / x-amz-date headers needed.
  std::string url = _cfg.creds->get_presigned_url(
    s3_object_ref{std::string{bucket}, std::string{key}}, presign_method::HEAD);

  curl_easy_setopt(h, CURLOPT_URL, url.c_str());
  curl_easy_setopt(h, CURLOPT_NOBODY, 1L);
  if (_cfg.request_timeout_s > 0) curl_easy_setopt(h, CURLOPT_TIMEOUT, _cfg.request_timeout_s);

  auto rc = curl_easy_perform(h);
  if (rc != CURLE_OK) {
    throw std::runtime_error(std::string("s3_ioctx: HEAD failed: ") + curl_easy_strerror(rc));
  }
  long http_code = 0;
  curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &http_code);
  if (http_code < 200 || http_code >= 300) {
    std::ostringstream os;
    os << "s3_ioctx: HEAD " << bucket << "/" << key << " returned HTTP " << http_code;
    throw std::runtime_error(os.str());
  }
  curl_off_t content_len = -1;
  curl_easy_getinfo(h, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_len);
  if (content_len < 0) throw std::runtime_error("s3_ioctx: HEAD response missing Content-Length");
  return static_cast<std::size_t>(content_len);
}

std::size_t s3_ioctx::range_get(std::string_view bucket,
                                std::string_view key,
                                std::size_t offset,
                                std::size_t size,
                                std::uint8_t* dst)
{
  if (size == 0) return 0;
  auto slot = acquire_handle();
  auto* h   = static_cast<CURL*>(slot.easy);

  // Auth lives in the URL's query string; we only attach the Range header,
  // which the presigned URL deliberately leaves unsigned (SignedHeaders=host
  // only) so callers may add Range / Accept / etc. without breaking the
  // signature.
  std::string url = _cfg.creds->get_presigned_url(
    s3_object_ref{std::string{bucket}, std::string{key}}, presign_method::GET);

  std::ostringstream range_os;
  range_os << "Range: bytes=" << offset << "-" << (offset + size - 1);
  std::string range_header = range_os.str();
  curl_slist* hdrs         = curl_slist_append(nullptr, range_header.c_str());

  buf_sink sink{dst, size, 0};
  header_capture hc;
  curl_easy_setopt(h, CURLOPT_URL, url.c_str());
  curl_easy_setopt(h, CURLOPT_HTTPGET, 1L);
  curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);
  curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, curl_write_buf);
  curl_easy_setopt(h, CURLOPT_WRITEDATA, &sink);
  curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, curl_header_capture);
  curl_easy_setopt(h, CURLOPT_HEADERDATA, &hc);
  if (_cfg.request_timeout_s > 0) curl_easy_setopt(h, CURLOPT_TIMEOUT, _cfg.request_timeout_s);

  auto rc = curl_easy_perform(h);
  curl_slist_free_all(hdrs);
  if (rc != CURLE_OK) {
    throw std::runtime_error(std::string("s3_ioctx: GET failed: ") + curl_easy_strerror(rc));
  }
  long http_code = 0;
  curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &http_code);

  // Validate the response actually covers the byte range we asked for. A
  // misbehaving / non-Range-aware store could otherwise hand us bytes from
  // the wrong offset under a 200 response and we'd silently feed corrupt
  // data into the parquet reader.
  auto fail = [&](std::string_view why) {
    std::ostringstream os;
    os << "s3_ioctx: GET " << bucket << "/" << key << " offset=" << offset << " size=" << size
       << " returned HTTP " << http_code << "; " << why;
    if (!hc.content_range.empty()) os << " (Content-Range: " << hc.content_range << ")";
    throw std::runtime_error(os.str());
  };

  if (http_code == 206) {
    if (sink.written != size)
      fail("206 body length " + std::to_string(sink.written) + " != requested size");
    if (!hc.content_range.empty()) {
      std::size_t got_start = 0;
      std::size_t got_end   = 0;
      if (!parse_content_range(hc.content_range, got_start, got_end))
        fail("malformed Content-Range");
      if (got_start != offset || got_end - got_start + 1 != size) fail("Content-Range mismatch");
    }
  } else if (http_code == 200) {
    // Some stores collapse a small ranged GET to a 200 with the full body.
    // Only accept that when the request started at offset 0 *and* the body
    // we got happens to match the requested length — both conditions guard
    // against silently grabbing the head of an object when we asked for an
    // interior range.
    if (offset != 0 || sink.written != size)
      fail("server returned 200 (no Range honored) for ranged request");
  } else {
    fail("unexpected status");
  }
  return sink.written;
}

// ===========================================================================
// host read APIs
// ===========================================================================

std::size_t s3_ioctx::host_read(sirius_io_object& obj,
                                std::size_t offset,
                                std::size_t size,
                                std::uint8_t* dst)
{
  auto& so = dynamic_cast<s3_io_object&>(obj);
  // Clip to object size so reads past EOF return short instead of having S3
  // throw 416 (matches the templated_ioctx generic behavior on local files).
  size = std::min(size, so.size() > offset ? so.size() - offset : std::size_t{0});
  if (size == 0) return 0;
  return range_get(so.bucket(), so.key(), offset, size, dst);
}

std::unique_ptr<cudf::io::datasource::buffer> s3_ioctx::host_read(sirius_io_object& obj,
                                                                  std::size_t offset,
                                                                  std::size_t size)
{
  auto& so = dynamic_cast<s3_io_object&>(obj);
  size     = std::min(size, so.size() > offset ? so.size() - offset : 0UL);
  std::vector<std::uint8_t> owned(size);
  std::size_t got = host_read(obj, offset, size, owned.data());
  owned.resize(got);
  return cudf::io::datasource::buffer::create(std::move(owned));
}

namespace {

void dispatch_async(io_completion_handler handler, std::function<std::size_t()> op)
{
  // Fire-and-forget worker thread. Exceptions from @p op are routed through
  // the handler as (0, exception_ptr). The base class contract is that the
  // handler is invoked exactly once on completion.
  std::thread([handler = std::move(handler), op = std::move(op)]() mutable {
    try {
      auto bytes = op();
      handler(bytes, nullptr);
    } catch (...) {
      handler(0, std::current_exception());
    }
  }).detach();
}

}  // namespace

void s3_ioctx::host_read_async(sirius_io_object& obj,
                               std::size_t offset,
                               std::size_t size,
                               std::uint8_t* dst,
                               io_completion_handler handler)
{
  // Capture shared_ptrs to keep both the ioctx and the io_object alive until
  // the detached worker completes; otherwise a caller dropping the datasource
  // (and its shared_ptr<sirius_io_object> / shared_ptr<sirius_ioctx>) before
  // the future resolves would leave us with dangling `this` / `&obj`.
  auto self      = shared_from_this();
  auto obj_owner = obj.shared_from_this();
  dispatch_async(std::move(handler), [self, obj_owner, offset, size, dst]() {
    return self->host_read(*obj_owner, offset, size, dst);
  });
}

void s3_ioctx::host_read_ranges_async(sirius_io_object& obj,
                                      std::vector<cudf::io::text::byte_range_info> const& ranges,
                                      std::span<cudf::host_span<std::byte>> dst,
                                      io_completion_handler handler)
{
  auto self      = shared_from_this();
  auto obj_owner = obj.shared_from_this();
  dispatch_async(std::move(handler), [self, obj_owner, ranges, dst]() {
    return self->host_read_ranges(*obj_owner, ranges, dst);
  });
}

std::size_t s3_ioctx::host_read_ranges(sirius_io_object& obj,
                                       std::vector<cudf::io::text::byte_range_info> const& ranges,
                                       std::span<cudf::host_span<std::byte>> dst)
{
  if (ranges.size() != dst.size())
    throw std::invalid_argument("s3_ioctx::host_read_ranges: ranges/dst size mismatch");
  std::size_t total = 0;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    auto offset = static_cast<std::size_t>(ranges[i].offset());
    auto size   = static_cast<std::size_t>(ranges[i].size());
    if (dst[i].size() < size)
      throw std::invalid_argument("s3_ioctx::host_read_ranges: dst span too small");
    total += host_read(obj, offset, size, reinterpret_cast<std::uint8_t*>(dst[i].data()));
  }
  return total;
}

// ===========================================================================
// device read APIs — host staging + H2D bounce
// ===========================================================================
//
// S3 has no native device path, so device_read_io lands bytes in a host buffer
// and issues cudaMemcpyAsync onto the caller-supplied stream. The base class
// device_read / device_read_async first consults the (optional) prefetching
// cache; these methods only run on cache miss.

std::unique_ptr<cudf::io::datasource::buffer> s3_ioctx::device_read_io(sirius_io_object& obj,
                                                                       std::size_t offset,
                                                                       std::size_t size,
                                                                       rmm::cuda_stream_view stream)
{
  // Round-trip through a host-owned buffer, then copy onto a freshly-allocated
  // device buffer returned as an owned_buffer wrapping a device_buffer.
  std::vector<std::uint8_t> host(size);
  auto got = host_read(obj, offset, size, host.data());
  host.resize(got);

  // Allocate device memory and issue an async H2D copy; sync before returning
  // so the buffer is safe to hand to cudf.
  rmm::device_buffer device_buf(got, stream);
  if (got > 0) {
    auto rc =
      cudaMemcpyAsync(device_buf.data(), host.data(), got, cudaMemcpyHostToDevice, stream.value());
    if (rc != cudaSuccess) {
      throw std::runtime_error(std::string("s3_ioctx::device_read_io cudaMemcpyAsync failed: ") +
                               cudaGetErrorString(rc));
    }
    if (auto sync_rc = cudaStreamSynchronize(stream.value()); sync_rc != cudaSuccess) {
      throw std::runtime_error(std::string("s3_ioctx::device_read_io stream sync failed: ") +
                               cudaGetErrorString(sync_rc));
    }
  }
  return cudf::io::datasource::buffer::create(std::move(device_buf));
}

std::size_t s3_ioctx::device_read_io(sirius_io_object& obj,
                                     std::size_t offset,
                                     std::size_t size,
                                     std::uint8_t* dst,
                                     rmm::cuda_stream_view stream)
{
  std::vector<std::uint8_t> host(size);
  auto got = host_read(obj, offset, size, host.data());
  if (got > 0) {
    auto rc = cudaMemcpyAsync(dst, host.data(), got, cudaMemcpyHostToDevice, stream.value());
    if (rc != cudaSuccess) {
      throw std::runtime_error(std::string("s3_ioctx::device_read_io cudaMemcpyAsync failed: ") +
                               cudaGetErrorString(rc));
    }
    if (auto sync_rc = cudaStreamSynchronize(stream.value()); sync_rc != cudaSuccess) {
      throw std::runtime_error(std::string("s3_ioctx::device_read_io stream sync failed: ") +
                               cudaGetErrorString(sync_rc));
    }
  }
  return got;
}

void s3_ioctx::device_read_io_async(sirius_io_object& obj,
                                    std::size_t offset,
                                    std::size_t size,
                                    std::uint8_t* dst,
                                    rmm::cuda_stream_view stream,
                                    io_completion_handler handler)
{
  // Keep the staging buffer alive until the H2D copy is sequenced on the
  // stream; the synchronous path above is sufficient because the worker
  // thread blocks on the stream before reporting completion. Captured
  // shared_ptrs extend the ioctx and io_object lifetimes through the
  // detached worker (see host_read_async).
  auto self      = shared_from_this();
  auto obj_owner = obj.shared_from_this();
  dispatch_async(std::move(handler), [self, obj_owner, offset, size, dst, stream]() {
    return self->device_read_io(*obj_owner, offset, size, dst, stream);
  });
}

// ===========================================================================
// compute_physical_range
// ===========================================================================

cudf::io::text::byte_range_info s3_ioctx::compute_physical_range(
  cudf::io::text::byte_range_info logical, std::size_t file_size) const
{
  auto const off  = static_cast<std::size_t>(logical.offset());
  auto const req  = static_cast<std::size_t>(logical.size());
  auto const clip = off >= file_size ? 0UL : std::min(req, file_size - off);
  return cudf::io::text::byte_range_info{static_cast<int64_t>(off), static_cast<int64_t>(clip)};
}

}  // namespace sirius::io::s3
