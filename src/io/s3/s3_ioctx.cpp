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

#include "io/s3/s3_io_object.hpp"
#include "io/s3/sigv4.hpp"
#include "io/sirius_datasource.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <future>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

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

// Split "http://host:port" into scheme / host[:port]. Required for SigV4's
// Host header (which is signed separately from the request URL).
void parse_endpoint(std::string_view endpoint, std::string& scheme, std::string& host_port)
{
  auto pos = endpoint.find("://");
  if (pos == std::string_view::npos)
    throw std::invalid_argument("s3_ioctx: endpoint must start with http(s)://");
  scheme                = std::string(endpoint.substr(0, pos));
  std::string_view rest = endpoint.substr(pos + 3);
  // Trim trailing slash; libcurl handles both but canonical host must not.
  while (!rest.empty() && rest.back() == '/')
    rest.remove_suffix(1);
  host_port = std::string(rest);
  if (host_port.empty()) throw std::invalid_argument("s3_ioctx: endpoint has no host");
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
  if (_cfg.endpoint.empty()) throw std::invalid_argument("s3_ioctx: endpoint is required");
  if (_cfg.access_key.empty() || _cfg.secret_key.empty())
    throw std::invalid_argument("s3_ioctx: credentials are required");
  if (_cfg.max_connections == 0) _cfg.max_connections = 1;

  ensure_curl_inited();

  parse_endpoint(_cfg.endpoint, _url_scheme, _host_header);

  _creds.access_key = _cfg.access_key;
  _creds.secret_key = _cfg.secret_key;
  _creds.region     = _cfg.region;
  _creds.service    = "s3";
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
  std::unique_ptr<sirius_io_object> io_object)
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

  // Canonical URI: "/bucket/uri_encode(key, encode_slash=false)" — path-style.
  // Slashes inside key are allowed to pass through so that "a/b/c" works.
  std::string canonical_uri =
    "/" + std::string(bucket) + "/" + uri_encode(key, /*encode_slash=*/false);
  std::string url = _url_scheme + "://" + _host_header + canonical_uri;

  std::string empty_sha = sha256_hex("");
  auto signed_req       = sign_request("HEAD",
                                 _host_header,
                                 canonical_uri,
                                 /*canonical_query=*/"",
                                 empty_sha,
                                 /*extra_headers=*/{},
                                 _creds,
                                 std::time(nullptr));

  struct curl_slist* hdrs = nullptr;
  for (auto const& [k, v] : signed_req.headers) {
    std::string line = k + ": " + v;
    hdrs             = curl_slist_append(hdrs, line.c_str());
  }

  curl_easy_setopt(h, CURLOPT_URL, url.c_str());
  curl_easy_setopt(h, CURLOPT_NOBODY, 1L);
  curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);
  if (_cfg.request_timeout_s > 0) curl_easy_setopt(h, CURLOPT_TIMEOUT, _cfg.request_timeout_s);

  auto rc = curl_easy_perform(h);
  curl_slist_free_all(hdrs);
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

  std::string canonical_uri =
    "/" + std::string(bucket) + "/" + uri_encode(key, /*encode_slash=*/false);
  std::string url = _url_scheme + "://" + _host_header + canonical_uri;

  // Range: bytes=offset-(offset+size-1). HTTP range is inclusive.
  std::ostringstream range_os;
  range_os << "bytes=" << offset << "-" << (offset + size - 1);
  std::string range_value = range_os.str();

  std::vector<std::pair<std::string, std::string>> extras = {{"range", range_value}};
  std::string empty_sha                                   = sha256_hex("");
  auto signed_req                                         = sign_request("GET",
                                 _host_header,
                                 canonical_uri,
                                 /*canonical_query=*/"",
                                 empty_sha,
                                 extras,
                                 _creds,
                                 std::time(nullptr));

  struct curl_slist* hdrs = nullptr;
  for (auto const& [k, v] : signed_req.headers) {
    std::string line = k + ": " + v;
    hdrs             = curl_slist_append(hdrs, line.c_str());
  }

  buf_sink sink{dst, size, 0};
  curl_easy_setopt(h, CURLOPT_URL, url.c_str());
  curl_easy_setopt(h, CURLOPT_HTTPGET, 1L);
  curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);
  curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, curl_write_buf);
  curl_easy_setopt(h, CURLOPT_WRITEDATA, &sink);
  if (_cfg.request_timeout_s > 0) curl_easy_setopt(h, CURLOPT_TIMEOUT, _cfg.request_timeout_s);

  auto rc = curl_easy_perform(h);
  curl_slist_free_all(hdrs);
  if (rc != CURLE_OK) {
    throw std::runtime_error(std::string("s3_ioctx: GET failed: ") + curl_easy_strerror(rc));
  }
  long http_code = 0;
  curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &http_code);
  // 206 Partial Content is the normal response to a Range request; some
  // stores collapse small ranges to 200, which we accept as long as we got
  // the bytes we asked for.
  if (http_code != 200 && http_code != 206) {
    std::ostringstream os;
    os << "s3_ioctx: GET " << bucket << "/" << key << " range=" << range_value << " returned HTTP "
       << http_code;
    throw std::runtime_error(os.str());
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

std::future<std::size_t> s3_ioctx::host_read_async(sirius_io_object& obj,
                                                   std::size_t offset,
                                                   std::size_t size,
                                                   std::uint8_t* dst)
{
  return std::async(std::launch::async, [this, &obj, offset, size, dst]() {
    return host_read(obj, offset, size, dst);
  });
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>> s3_ioctx::host_read_async(
  sirius_io_object& obj, std::size_t offset, std::size_t size)
{
  return std::async(std::launch::async,
                    [this, &obj, offset, size]() { return host_read(obj, offset, size); });
}

std::future<std::size_t> s3_ioctx::host_read_ranges_async(
  sirius_io_object& obj,
  std::vector<cudf::io::text::byte_range_info> const& ranges,
  std::span<cudf::host_span<std::byte>> dst)
{
  return std::async(std::launch::async,
                    [this, &obj, ranges, dst]() { return host_read_ranges(obj, ranges, dst); });
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
// device read APIs — unsupported on S3
// ===========================================================================

std::unique_ptr<cudf::io::datasource::buffer> s3_ioctx::device_read(sirius_io_object&,
                                                                    std::size_t,
                                                                    std::size_t,
                                                                    rmm::cuda_stream_view)
{
  throw std::logic_error("s3_ioctx: device_read not supported (host-only backend)");
}

std::size_t s3_ioctx::device_read(
  sirius_io_object&, std::size_t, std::size_t, std::uint8_t*, rmm::cuda_stream_view)
{
  throw std::logic_error("s3_ioctx: device_read not supported (host-only backend)");
}

std::future<std::size_t> s3_ioctx::device_read_async(
  sirius_io_object&, std::size_t, std::size_t, std::uint8_t*, rmm::cuda_stream_view)
{
  throw std::logic_error("s3_ioctx: device_read_async not supported (host-only backend)");
}

}  // namespace sirius::io::s3
