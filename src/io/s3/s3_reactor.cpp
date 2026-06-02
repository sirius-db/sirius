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

#include "io/s3/s3_reactor.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace sirius::io::s3 {

namespace {

struct curl_global_init_once {
  curl_global_init_once()
  {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != 0)
      throw std::runtime_error("s3_reactor: curl_global_init failed");
  }
};
void ensure_curl_inited()
{
  static curl_global_init_once inst;
  (void)inst;
}

// Write callback into a caller-supplied flat buffer (mirrors the production
// s3_ioctx buf_sink). Always consumes the full incoming chunk so curl can
// complete; overflow is detected after the fact via total_received vs written.
struct buf_sink {
  std::uint8_t* dst{nullptr};
  std::size_t capacity{0};
  std::size_t written{0};
  std::size_t total_received{0};
};
std::size_t curl_write_buf(void* ptr, std::size_t size, std::size_t nmemb, void* userdata)
{
  auto* sink  = static_cast<buf_sink*>(userdata);
  auto nbytes = size * nmemb;
  auto room   = sink->capacity > sink->written ? sink->capacity - sink->written : std::size_t{0};
  auto copy   = std::min(nbytes, room);
  if (copy > 0 && sink->dst != nullptr) {
    std::memcpy(sink->dst + sink->written, ptr, copy);
    sink->written += copy;
  }
  sink->total_received += nbytes;
  return nbytes;
}

std::size_t curl_discard(void*, std::size_t size, std::size_t nmemb, void*) { return size * nmemb; }

curl_slist* build_header_slist(std::vector<std::pair<std::string, std::string>> const& headers)
{
  curl_slist* list = nullptr;
  for (auto const& [name, value] : headers) {
    std::string line = name + ": " + value;
    list             = curl_slist_append(list, line.c_str());
  }
  return list;
}

}  // namespace

// ---------------------------------------------------------------------------
// per-async-request transfer state
// ---------------------------------------------------------------------------

struct s3_reactor::transfer {
  host_read_req_type req;
  void* easy{nullptr};  // CURL*
  curl_slist* hdrs{nullptr};
  std::string url;
  std::string range_header;
  buf_sink sink{};
};

// ---------------------------------------------------------------------------
// ctor / dtor
// ---------------------------------------------------------------------------

s3_reactor::s3_reactor(config cfg) : _cfg(std::move(cfg))
{
  if (!_cfg.creds) throw std::invalid_argument("s3_reactor: s3_request_authorizer is required");
  _cfg.max_connections = std::max<std::size_t>(_cfg.max_connections, 1);
  ensure_curl_inited();
  _multi = curl_multi_init();
  if (_multi == nullptr) throw std::runtime_error("s3_reactor: curl_multi_init failed");
  _worker = std::thread([this] { worker_loop(); });
}

s3_reactor::~s3_reactor() { shutdown(); }

void s3_reactor::interrupt()
{
  if (_multi != nullptr) curl_multi_wakeup(static_cast<CURLM*>(_multi));
}

void s3_reactor::shutdown()
{
  bool expected = false;
  if (!_stop.compare_exchange_strong(expected, true)) {
    if (_worker.joinable()) _worker.join();
    return;
  }
  interrupt();
  if (_worker.joinable()) _worker.join();
  if (_multi != nullptr) {
    curl_multi_cleanup(static_cast<CURLM*>(_multi));
    _multi = nullptr;
  }
}

// ---------------------------------------------------------------------------
// synchronous blocking GET / HEAD (separate easy path)
// ---------------------------------------------------------------------------

std::size_t s3_reactor::blocking_request(std::string_view bucket,
                                         std::string_view key,
                                         s3_request_method method,
                                         std::size_t offset,
                                         std::size_t size,
                                         std::uint8_t* dst,
                                         std::size_t* out_object_size)
{
  ensure_curl_inited();
  CURL* h = curl_easy_init();
  if (h == nullptr) throw std::runtime_error("s3_reactor: curl_easy_init failed");

  s3_authorized_request req = _cfg.creds->authorize(
    s3_object_ref{std::string{bucket}, std::string{key}},
    method,
    std::chrono::seconds{_cfg.request_timeout_s > 0 ? _cfg.request_timeout_s : 20});

  curl_slist* hdrs = build_header_slist(req.headers);
  std::string range_header;
  buf_sink sink{dst, size, 0, 0};

  curl_easy_setopt(h, CURLOPT_URL, req.url.c_str());
  if (method == s3_request_method::HEAD) {
    curl_easy_setopt(h, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, curl_discard);
  } else {
    std::ostringstream range_os;
    range_os << "Range: bytes=" << offset << "-" << (offset + size - 1);
    range_header = range_os.str();
    hdrs         = curl_slist_append(hdrs, range_header.c_str());
    curl_easy_setopt(h, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, curl_write_buf);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &sink);
  }
  curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);
  if (_cfg.request_timeout_s > 0) curl_easy_setopt(h, CURLOPT_TIMEOUT, _cfg.request_timeout_s);
  if (!_cfg.ca_bundle_path.empty())
    curl_easy_setopt(h, CURLOPT_CAINFO, _cfg.ca_bundle_path.c_str());
  if (!_cfg.tls_verify) {
    curl_easy_setopt(h, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(h, CURLOPT_SSL_VERIFYHOST, 0L);
  }

  CURLcode rc = curl_easy_perform(h);
  long http   = 0;
  curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &http);
  curl_off_t content_len = 0;
  curl_easy_getinfo(h, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_len);
  curl_slist_free_all(hdrs);

  auto fail = [&](std::string const& why) {
    std::ostringstream os;
    os << "s3_reactor: " << (method == s3_request_method::HEAD ? "HEAD " : "GET ") << bucket << "/"
       << key << " failed: " << why;
    curl_easy_cleanup(h);
    throw std::runtime_error(os.str());
  };

  if (rc != CURLE_OK) fail(std::string{"libcurl "} + curl_easy_strerror(rc));

  if (method == s3_request_method::HEAD) {
    if (http != 200) fail("HTTP " + std::to_string(http));
    if (out_object_size != nullptr)
      *out_object_size = content_len >= 0 ? static_cast<std::size_t>(content_len) : 0;
    curl_easy_cleanup(h);
    return 0;
  }

  if (http != 200 && http != 206) fail("HTTP " + std::to_string(http));
  curl_easy_cleanup(h);
  _bytes_read_total.fetch_add(sink.written, std::memory_order_relaxed);
  return sink.written;
}

std::size_t s3_reactor::host_read(native_handle_type handle,
                                  std::size_t offset,
                                  std::size_t size,
                                  std::uint8_t* dst)
{
  if (size == 0) return 0;
  return blocking_request(
    handle->bucket, handle->key, s3_request_method::GET, offset, size, dst, nullptr);
}

std::size_t s3_reactor::head_object_size(std::string_view bucket, std::string_view key)
{
  std::size_t sz = 0;
  blocking_request(bucket, key, s3_request_method::HEAD, 0, 0, nullptr, &sz);
  return sz;
}

// ---------------------------------------------------------------------------
// async host reads (curl_multi worker loop, bounded submit)
// ---------------------------------------------------------------------------

void s3_reactor::host_read_async(host_read_req_type req)
{
  {
    std::lock_guard<std::mutex> lk(_mtx);
    _incoming.push_back(std::move(req));
  }
  interrupt();
}

void s3_reactor::host_enqueue_bulk(std::span<host_read_req_type> batch)
{
  {
    std::lock_guard<std::mutex> lk(_mtx);
    for (auto& r : batch)
      _incoming.push_back(std::move(r));
  }
  interrupt();
}

void s3_reactor::enqueue_bulk(std::span<device_read_req_type> batch)
{
  // Device reads land in Phase 2. Drain pending ctx so callers don't hang.
  for (auto& r : batch) {
    if (r.ctx) {
      r.ctx->chunk_failed(
        std::make_exception_ptr(std::logic_error("s3_reactor: device reads land in Phase 2")));
    }
  }
  throw std::logic_error("s3_reactor: device reads land in Phase 2");
}

void s3_reactor::submit_pending()
{
  auto* multi = static_cast<CURLM*>(_multi);
  while (_inflight.size() < _cfg.max_connections) {
    host_read_req_type req;
    {
      std::lock_guard<std::mutex> lk(_mtx);
      if (_pending.empty()) break;
      req = std::move(_pending.front());
      _pending.pop_front();
    }

    auto* t = new transfer{};
    t->req  = std::move(req);
    t->easy = curl_easy_init();
    if (t->easy == nullptr) {
      if (t->req.ctx)
        t->req.ctx->chunk_failed(
          std::make_exception_ptr(std::runtime_error("s3_reactor: curl_easy_init failed")));
      delete t;
      continue;
    }

    auto const& state          = *t->req.handle;
    s3_authorized_request auth = _cfg.creds->authorize(
      s3_object_ref{state.bucket, state.key},
      s3_request_method::GET,
      std::chrono::seconds{_cfg.request_timeout_s > 0 ? _cfg.request_timeout_s : 20});
    t->url  = auth.url;
    t->hdrs = build_header_slist(auth.headers);
    std::ostringstream range_os;
    range_os << "Range: bytes=" << t->req.offset << "-" << (t->req.offset + t->req.size - 1);
    t->range_header = range_os.str();
    t->hdrs         = curl_slist_append(t->hdrs, t->range_header.c_str());
    t->sink         = buf_sink{t->req.dst, t->req.size, 0, 0};

    auto* e = static_cast<CURL*>(t->easy);
    curl_easy_setopt(e, CURLOPT_URL, t->url.c_str());
    curl_easy_setopt(e, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(e, CURLOPT_HTTPHEADER, t->hdrs);
    curl_easy_setopt(e, CURLOPT_WRITEFUNCTION, curl_write_buf);
    curl_easy_setopt(e, CURLOPT_WRITEDATA, &t->sink);
    if (_cfg.request_timeout_s > 0) curl_easy_setopt(e, CURLOPT_TIMEOUT, _cfg.request_timeout_s);
    if (!_cfg.ca_bundle_path.empty())
      curl_easy_setopt(e, CURLOPT_CAINFO, _cfg.ca_bundle_path.c_str());
    if (!_cfg.tls_verify) {
      curl_easy_setopt(e, CURLOPT_SSL_VERIFYPEER, 0L);
      curl_easy_setopt(e, CURLOPT_SSL_VERIFYHOST, 0L);
    }
    curl_easy_setopt(e, CURLOPT_PRIVATE, t);

    curl_multi_add_handle(multi, e);
    _inflight.emplace(t->easy, t);
  }
}

void s3_reactor::finish(transfer* t, std::exception_ptr ep)
{
  auto* multi = static_cast<CURLM*>(_multi);
  curl_multi_remove_handle(multi, static_cast<CURL*>(t->easy));
  curl_easy_cleanup(static_cast<CURL*>(t->easy));
  if (t->hdrs != nullptr) curl_slist_free_all(t->hdrs);
  _inflight.erase(t->easy);

  if (ep) {
    if (t->req.ctx) t->req.ctx->chunk_failed(ep);
  } else {
    _bytes_read_total.fetch_add(t->sink.written, std::memory_order_relaxed);
    if (t->req.ctx) t->req.ctx->chunk_done();
  }
  delete t;
}

void s3_reactor::worker_loop()
{
  auto* multi = static_cast<CURLM*>(_multi);
  while (true) {
    // 1. drain newly-enqueued work into the pending list.
    {
      std::lock_guard<std::mutex> lk(_mtx);
      while (!_incoming.empty()) {
        _pending.push_back(std::move(_incoming.front()));
        _incoming.pop_front();
      }
    }

    // 2. on shutdown: cancel everything (in-flight + pending) via the normal
    //    completion path, then exit. Cancellation runs on the worker thread so
    //    all curl handle ops stay single-threaded.
    if (_stop.load(std::memory_order_acquire)) {
      auto cancelled = []() {
        return std::make_exception_ptr(
          std::runtime_error("s3_reactor: read cancelled by shutdown"));
      };
      std::vector<transfer*> live;
      live.reserve(_inflight.size());
      for (auto& [_, t] : _inflight)
        live.push_back(t);
      for (auto* t : live)
        finish(t, cancelled());
      std::deque<host_read_req_type> leftover;
      {
        std::lock_guard<std::mutex> lk(_mtx);
        leftover.swap(_pending);
      }
      for (auto& r : leftover) {
        if (r.ctx) r.ctx->chunk_failed(cancelled());
      }
      break;
    }

    // 3. submit within the max_connections budget.
    submit_pending();

    // 4. drive the multi.
    int running = 0;
    curl_multi_perform(multi, &running);

    // 5. reap completions.
    int queued = 0;
    while (CURLMsg* m = curl_multi_info_read(multi, &queued)) {
      if (m->msg != CURLMSG_DONE) continue;
      transfer* t = nullptr;
      char* priv  = nullptr;
      curl_easy_getinfo(m->easy_handle, CURLINFO_PRIVATE, &priv);
      t = reinterpret_cast<transfer*>(priv);
      if (t == nullptr) continue;

      std::exception_ptr ep;
      if (m->data.result != CURLE_OK) {
        ep = std::make_exception_ptr(std::runtime_error(std::string{"s3_reactor: libcurl "} +
                                                        curl_easy_strerror(m->data.result)));
      } else {
        long http = 0;
        curl_easy_getinfo(m->easy_handle, CURLINFO_RESPONSE_CODE, &http);
        if (http != 200 && http != 206) {
          ep =
            std::make_exception_ptr(std::runtime_error("s3_reactor: HTTP " + std::to_string(http)));
        } else if (t->sink.written != t->req.size) {
          ep = std::make_exception_ptr(std::runtime_error("s3_reactor: short read (" +
                                                          std::to_string(t->sink.written) + " of " +
                                                          std::to_string(t->req.size) + ")"));
        }
      }
      finish(t, ep);
    }

    // 6. completions freed slots — submit more.
    submit_pending();

    // 7. sleep until socket activity, a wakeup, or the 100 ms timeout.
    curl_multi_poll(multi, nullptr, 0, 100, nullptr);
  }
}

// ---------------------------------------------------------------------------
// statistics
// ---------------------------------------------------------------------------

cudf::io::text::byte_range_info s3_reactor::align_to_physical(
  cudf::io::text::byte_range_info logical, std::size_t file_size)
{
  auto offset = static_cast<std::size_t>(logical.offset());
  auto size   = static_cast<std::size_t>(logical.size());
  if (offset >= file_size) return {static_cast<int64_t>(offset), 0};
  size = std::min(size, file_size - offset);
  return {static_cast<int64_t>(offset), static_cast<int64_t>(size)};
}

bool s3_reactor::supports(std::string_view path)
{
  auto pos = path.find("://");
  if (pos == std::string_view::npos) return false;
  return path.substr(0, pos) == "s3";
}

std::unique_ptr<s3_async_io_object> s3_reactor::create_io_object(std::string)
{
  throw std::logic_error(
    "s3_reactor::create_io_object: use s3_async_experimental_ioctx::create_io_object (needs an "
    "instance HEAD via the authorizer)");
}

}  // namespace sirius::io::s3
