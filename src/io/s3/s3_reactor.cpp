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
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::io::s3 {

namespace {

using namespace std::chrono_literals;

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

// Write callback into a caller-supplied flat buffer. Always consumes the full
// incoming chunk so curl can complete; overflow shows up as total_received >
// written (written is capped at capacity) and is rejected during validation.
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

// Header callback capturing Content-Range (206 validation) and Retry-After
// (backoff override on 429 / 503).
struct header_capture {
  std::string content_range;
  std::string retry_after;
};
std::string_view trim_header_value(std::string_view v)
{
  while (!v.empty() && (v.front() == ' ' || v.front() == '\t'))
    v.remove_prefix(1);
  while (!v.empty() &&
         (v.back() == '\r' || v.back() == '\n' || v.back() == ' ' || v.back() == '\t'))
    v.remove_suffix(1);
  return v;
}
std::optional<std::string_view> match_header(std::string_view line, std::string_view lc_prefix)
{
  if (line.size() < lc_prefix.size()) return std::nullopt;
  for (std::size_t i = 0; i < lc_prefix.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(line[i])) !=
        static_cast<unsigned char>(lc_prefix[i]))
      return std::nullopt;
  }
  return trim_header_value(line.substr(lc_prefix.size()));
}
std::size_t curl_header_capture(char* buffer, std::size_t size, std::size_t nmemb, void* userdata)
{
  auto* hc          = static_cast<header_capture*>(userdata);
  std::size_t total = size * nmemb;
  std::string_view l(buffer, total);
  if (auto v = match_header(l, "content-range:")) hc->content_range = std::string(*v);
  if (auto v = match_header(l, "retry-after:")) hc->retry_after = std::string(*v);
  return total;
}

curl_slist* build_header_slist(std::vector<std::pair<std::string, std::string>> const& headers)
{
  curl_slist* list = nullptr;
  for (auto const& [name, value] : headers) {
    std::string line = name + ": " + value;
    list             = curl_slist_append(list, line.c_str());
  }
  return list;
}

bool is_retriable_status(long s)
{
  if (s == 408 || s == 429) return true;
  return s >= 500 && s < 600;
}
bool is_retriable_curl_code(CURLcode code)
{
  switch (code) {
    case CURLE_COULDNT_CONNECT:
    case CURLE_OPERATION_TIMEDOUT:
    case CURLE_GOT_NOTHING:
    case CURLE_RECV_ERROR:
    case CURLE_SEND_ERROR:
    case CURLE_PARTIAL_FILE:
    case CURLE_HTTP2_STREAM: return true;
    default: return false;
  }
}
std::optional<std::chrono::milliseconds> parse_retry_after_seconds(std::string_view v)
{
  if (v.empty()) return std::nullopt;
  for (char c : v) {
    if (!std::isdigit(static_cast<unsigned char>(c)))
      return std::nullopt;  // HTTP-date form ignored
  }
  try {
    return std::chrono::milliseconds{std::stoull(std::string{v}) * 1000};
  } catch (...) {
    return std::nullopt;
  }
}
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

// Validate a 200/206 body actually covers the requested [offset, offset+size).
// Returns an exception_ptr on anomaly (non-retriable wrong-bytes risk), else
// nullptr. Mirrors the production s3_ioctx guard.
std::exception_ptr validate_range(std::string_view bucket,
                                  std::string_view key,
                                  std::size_t offset,
                                  std::size_t size,
                                  long http,
                                  buf_sink const& sink,
                                  header_capture const& hc)
{
  auto fail = [&](std::string_view why) {
    std::ostringstream os;
    os << "s3_reactor: GET " << bucket << "/" << key << " offset=" << offset << " size=" << size
       << " returned HTTP " << http << "; " << why;
    if (!hc.content_range.empty()) os << " (Content-Range: " << hc.content_range << ")";
    return std::make_exception_ptr(std::runtime_error(os.str()));
  };
  if (http == 206) {
    if (sink.total_received != size || sink.written != size)
      return fail("206 body length mismatch");
    if (!hc.content_range.empty()) {
      std::size_t s = 0;
      std::size_t e = 0;
      if (!parse_content_range(hc.content_range, s, e)) return fail("malformed Content-Range");
      if (s != offset || e - s + 1 != size) return fail("Content-Range mismatch");
    }
    return nullptr;
  }
  if (http == 200) {
    if (offset != 0 || sink.total_received != size) return fail("server returned 200 (no Range)");
    return nullptr;
  }
  return fail("unexpected status");
}

}  // namespace

// ---------------------------------------------------------------------------
// per-async-request transfer state
// ---------------------------------------------------------------------------

struct s3_reactor::transfer {
  host_read_req_type req;
  std::size_t attempt{1};
  void* easy{nullptr};  // CURL*
  curl_slist* hdrs{nullptr};
  std::string url;
  std::string range_header;
  buf_sink sink{};
  header_capture hc{};
};

// ---------------------------------------------------------------------------
// ctor / dtor / shutdown
// ---------------------------------------------------------------------------

s3_reactor::s3_reactor(config cfg) : _cfg(std::move(cfg))
{
  if (!_cfg.creds) throw std::invalid_argument("s3_reactor: s3_request_authorizer is required");
  _cfg.max_connections    = std::max<std::size_t>(_cfg.max_connections, 1);
  _cfg.max_retry_attempts = std::max<std::size_t>(_cfg.max_retry_attempts, 1);
  ensure_curl_inited();
  auto* multi = curl_multi_init();
  if (multi == nullptr) throw std::runtime_error("s3_reactor: curl_multi_init failed");
  curl_multi_setopt(multi, CURLMOPT_MAX_TOTAL_CONNECTIONS, static_cast<long>(_cfg.max_connections));
  curl_multi_setopt(multi, CURLMOPT_MAX_HOST_CONNECTIONS, static_cast<long>(_cfg.max_connections));
  _multi  = multi;
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
// backoff
// ---------------------------------------------------------------------------

std::chrono::steady_clock::duration s3_reactor::backoff_delay(
  std::size_t attempt, std::optional<std::chrono::milliseconds> retry_after) const
{
  constexpr std::chrono::milliseconds kCap{30 * 1000};
  if (retry_after && _cfg.honor_retry_after) return std::min(*retry_after, kCap);
  std::size_t const shift = std::min<std::size_t>(attempt > 0 ? attempt - 1 : 0, 16);
  long wait               = _cfg.retry_backoff_base.count() << shift;
  if (_cfg.retry_jitter.count() > 0) {
    thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<long> d(0, _cfg.retry_jitter.count());
    wait += d(rng);
  }
  return std::chrono::milliseconds{wait};
}

// ---------------------------------------------------------------------------
// synchronous blocking GET / HEAD (separate easy path, with retry)
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
  auto const max_attempts = std::max<std::size_t>(_cfg.max_retry_attempts, 1);
  std::string last_why    = "no attempt made";

  for (std::size_t attempt = 1; attempt <= max_attempts; ++attempt) {
    // Authorize first: if signing throws there is no curl handle to leak yet
    // (the async path relies on submit_pending's try/catch for the same reason).
    s3_authorized_request req = _cfg.creds->authorize(
      s3_object_ref{std::string{bucket}, std::string{key}},
      method,
      std::chrono::seconds{_cfg.request_timeout_s > 0 ? _cfg.request_timeout_s : 20});

    CURL* h = curl_easy_init();
    if (h == nullptr) throw std::runtime_error("s3_reactor: curl_easy_init failed");

    curl_slist* hdrs = build_header_slist(req.headers);
    std::string range_header;
    buf_sink sink{dst, size, 0, 0};
    header_capture hc;

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
    curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, curl_header_capture);
    curl_easy_setopt(h, CURLOPT_HEADERDATA, &hc);
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
    curl_easy_cleanup(h);

    bool retriable = false;
    if (rc != CURLE_OK) {
      last_why  = std::string{"libcurl "} + curl_easy_strerror(rc);
      retriable = is_retriable_curl_code(rc);
    } else if (method == s3_request_method::HEAD) {
      if (http == 200) {
        if (out_object_size != nullptr)
          *out_object_size = content_len >= 0 ? static_cast<std::size_t>(content_len) : 0;
        return 0;
      }
      last_why  = "HTTP " + std::to_string(http);
      retriable = is_retriable_status(http);
    } else if (http == 200 || http == 206) {
      if (auto ep = validate_range(bucket, key, offset, size, http, sink, hc)) {
        std::rethrow_exception(ep);  // non-retriable anomaly
      }
      _bytes_read_total.fetch_add(sink.written, std::memory_order_relaxed);
      return sink.written;
    } else {
      last_why  = "HTTP " + std::to_string(http);
      retriable = is_retriable_status(http);
    }

    if (!retriable || attempt >= max_attempts) break;
    std::optional<std::chrono::milliseconds> ra;
    if (_cfg.honor_retry_after) ra = parse_retry_after_seconds(hc.retry_after);
    std::this_thread::sleep_for(backoff_delay(attempt, ra));
  }

  std::ostringstream os;
  os << "s3_reactor: " << (method == s3_request_method::HEAD ? "HEAD " : "GET ") << bucket << "/"
     << key << " failed after " << max_attempts << " attempt(s); last: " << last_why;
  throw std::runtime_error(os.str());
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
// async enqueue
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
  for (auto& r : batch) {
    if (r.ctx)
      r.ctx->chunk_failed(
        std::make_exception_ptr(std::logic_error("s3_reactor: device reads land in Phase 2")));
  }
  throw std::logic_error("s3_reactor: device reads land in Phase 2");
}

// ---------------------------------------------------------------------------
// worker loop
// ---------------------------------------------------------------------------

void s3_reactor::drain_incoming()
{
  std::deque<host_read_req_type> taken;
  {
    std::lock_guard<std::mutex> lk(_mtx);
    taken.swap(_incoming);
  }
  for (auto& r : taken) {
    auto* t    = new transfer{};
    t->req     = std::move(r);
    t->attempt = 1;
    _pending.push_back(t);
  }
}

void s3_reactor::promote_due_retries()
{
  auto const now = std::chrono::steady_clock::now();
  for (auto it = _retry_queue.begin(); it != _retry_queue.end() && it->first <= now;) {
    _pending.push_front(it->second);
    it = _retry_queue.erase(it);
  }
}

void s3_reactor::build_and_add(transfer* t)
{
  auto const& state          = *t->req.handle;
  s3_authorized_request auth = _cfg.creds->authorize(
    s3_object_ref{state.bucket, state.key},
    s3_request_method::GET,
    std::chrono::seconds{_cfg.request_timeout_s > 0 ? _cfg.request_timeout_s : 20});

  t->easy = curl_easy_init();
  if (t->easy == nullptr) throw std::runtime_error("s3_reactor: curl_easy_init failed");
  t->url  = auth.url;
  t->hdrs = build_header_slist(auth.headers);
  std::ostringstream range_os;
  range_os << "Range: bytes=" << t->req.offset << "-" << (t->req.offset + t->req.size - 1);
  t->range_header = range_os.str();
  t->hdrs         = curl_slist_append(t->hdrs, t->range_header.c_str());
  t->sink         = buf_sink{t->req.dst, t->req.size, 0, 0};
  t->hc           = header_capture{};

  auto* e = static_cast<CURL*>(t->easy);
  curl_easy_setopt(e, CURLOPT_PRIVATE, t);  // recovered via CURLINFO_PRIVATE on completion
  curl_easy_setopt(e, CURLOPT_URL, t->url.c_str());
  curl_easy_setopt(e, CURLOPT_HTTPGET, 1L);
  curl_easy_setopt(e, CURLOPT_HTTPHEADER, t->hdrs);
  curl_easy_setopt(e, CURLOPT_WRITEFUNCTION, curl_write_buf);
  curl_easy_setopt(e, CURLOPT_WRITEDATA, &t->sink);
  curl_easy_setopt(e, CURLOPT_HEADERFUNCTION, curl_header_capture);
  curl_easy_setopt(e, CURLOPT_HEADERDATA, &t->hc);
  if (_cfg.request_timeout_s > 0) curl_easy_setopt(e, CURLOPT_TIMEOUT, _cfg.request_timeout_s);
  if (!_cfg.ca_bundle_path.empty())
    curl_easy_setopt(e, CURLOPT_CAINFO, _cfg.ca_bundle_path.c_str());
  if (!_cfg.tls_verify) {
    curl_easy_setopt(e, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(e, CURLOPT_SSL_VERIFYHOST, 0L);
  }

  CURLMcode mc = curl_multi_add_handle(static_cast<CURLM*>(_multi), e);
  if (mc != CURLM_OK)
    throw std::runtime_error(std::string{"s3_reactor: curl_multi_add_handle: "} +
                             curl_multi_strerror(mc));
  _inflight.emplace(t->easy, t);
}

void s3_reactor::submit_pending()
{
  while (_inflight.size() < _cfg.max_connections && !_pending.empty()) {
    transfer* t = _pending.front();
    _pending.pop_front();
    try {
      build_and_add(t);  // may throw (authorize / init / add_handle) — never escapes the worker
    } catch (...) {
      auto ep = std::current_exception();
      if (t->easy != nullptr) {
        curl_easy_cleanup(static_cast<CURL*>(t->easy));  // not added to the multi
        t->easy = nullptr;
      }
      if (t->hdrs != nullptr) {
        curl_slist_free_all(t->hdrs);
        t->hdrs = nullptr;
      }
      if (t->req.ctx) t->req.ctx->chunk_failed(ep);
      delete t;
    }
  }
}

void s3_reactor::schedule_retry(transfer* t, std::chrono::steady_clock::duration delay)
{
  // _inflight was already erased by the worker loop before this call; the easy
  // handle is still registered in the multi, so remove + free it here.
  curl_multi_remove_handle(static_cast<CURLM*>(_multi), static_cast<CURL*>(t->easy));
  curl_easy_cleanup(static_cast<CURL*>(t->easy));
  if (t->hdrs != nullptr) curl_slist_free_all(t->hdrs);
  t->easy = nullptr;
  t->hdrs = nullptr;
  ++t->attempt;
  _retry_queue.emplace(std::chrono::steady_clock::now() + delay, t);
}

void s3_reactor::finish(transfer* t, std::exception_ptr ep)
{
  if (t->easy != nullptr) {
    curl_multi_remove_handle(static_cast<CURLM*>(_multi), static_cast<CURL*>(t->easy));
    curl_easy_cleanup(static_cast<CURL*>(t->easy));
    t->easy = nullptr;
  }
  if (t->hdrs != nullptr) {
    curl_slist_free_all(t->hdrs);
    t->hdrs = nullptr;
  }
  if (ep) {
    if (t->req.ctx) t->req.ctx->chunk_failed(ep);
  } else {
    _bytes_read_total.fetch_add(t->sink.written, std::memory_order_relaxed);
    if (t->req.ctx) t->req.ctx->chunk_done();
  }
  delete t;
}

void s3_reactor::cancel_all_on_shutdown()
{
  auto cancelled = [] {
    return std::make_exception_ptr(std::runtime_error("s3_reactor: read cancelled by shutdown"));
  };
  std::vector<transfer*> live;
  live.reserve(_inflight.size());
  for (auto& [_, t] : _inflight)
    live.push_back(t);
  _inflight.clear();
  for (auto* t : live) {
    if (t->easy != nullptr) {
      curl_multi_remove_handle(static_cast<CURLM*>(_multi), static_cast<CURL*>(t->easy));
      curl_easy_cleanup(static_cast<CURL*>(t->easy));
    }
    if (t->hdrs != nullptr) curl_slist_free_all(t->hdrs);
    if (t->req.ctx) t->req.ctx->chunk_failed(cancelled());
    delete t;
  }
  for (auto& [_, t] : _retry_queue) {
    if (t->req.ctx) t->req.ctx->chunk_failed(cancelled());
    delete t;
  }
  _retry_queue.clear();
  std::deque<transfer*> leftover;
  leftover.swap(_pending);
  for (auto* t : leftover) {
    if (t->req.ctx) t->req.ctx->chunk_failed(cancelled());
    delete t;
  }
  std::deque<host_read_req_type> incoming;
  {
    std::lock_guard<std::mutex> lk(_mtx);
    incoming.swap(_incoming);
  }
  for (auto& r : incoming) {
    if (r.ctx) r.ctx->chunk_failed(cancelled());
  }
}

void s3_reactor::worker_loop()
{
  auto* multi = static_cast<CURLM*>(_multi);
  while (true) {
    drain_incoming();
    if (_stop.load(std::memory_order_acquire)) {
      cancel_all_on_shutdown();
      break;
    }
    promote_due_retries();
    submit_pending();

    int running = 0;
    curl_multi_perform(multi, &running);

    int queued = 0;
    while (CURLMsg* m = curl_multi_info_read(multi, &queued)) {
      if (m->msg != CURLMSG_DONE) continue;
      char* priv = nullptr;
      curl_easy_getinfo(m->easy_handle, CURLINFO_PRIVATE, &priv);
      auto* t = reinterpret_cast<transfer*>(priv);
      if (t == nullptr) continue;
      _inflight.erase(t->easy);  // remove from map; finish/schedule own the handle now

      CURLcode result = m->data.result;
      bool retriable  = false;
      std::string last_why;
      std::optional<std::chrono::milliseconds> retry_after;
      std::exception_ptr ep;

      if (result != CURLE_OK) {
        last_why  = std::string{"libcurl "} + curl_easy_strerror(result);
        retriable = is_retriable_curl_code(result);
        if (!retriable) ep = std::make_exception_ptr(std::runtime_error("s3_reactor: " + last_why));
      } else {
        long http = 0;
        curl_easy_getinfo(m->easy_handle, CURLINFO_RESPONSE_CODE, &http);
        if (http == 200 || http == 206) {
          ep = validate_range(t->req.handle->bucket,
                              t->req.handle->key,
                              t->req.offset,
                              t->req.size,
                              http,
                              t->sink,
                              t->hc);  // non-retriable anomaly or nullptr
        } else if (is_retriable_status(http)) {
          retriable = true;
          last_why  = "HTTP " + std::to_string(http);
          if (_cfg.honor_retry_after) retry_after = parse_retry_after_seconds(t->hc.retry_after);
        } else {
          ep =
            std::make_exception_ptr(std::runtime_error("s3_reactor: HTTP " + std::to_string(http)));
        }
      }

      if (retriable && t->attempt < _cfg.max_retry_attempts) {
        schedule_retry(t, backoff_delay(t->attempt, retry_after));
      } else if (retriable) {
        std::ostringstream os;
        os << "s3_reactor: GET " << t->req.handle->bucket << "/" << t->req.handle->key
           << " failed after " << _cfg.max_retry_attempts << " attempt(s); last: " << last_why;
        finish(t, std::make_exception_ptr(std::runtime_error(os.str())));
      } else {
        finish(t, ep);  // ep nullptr -> chunk_done; else chunk_failed
      }
    }

    submit_pending();

    // Sleep until socket activity, a wakeup, or the next due retry (cap 100ms).
    int timeout_ms = 100;
    if (!_retry_queue.empty()) {
      auto next = _retry_queue.begin()->first;
      auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                  next - std::chrono::steady_clock::now())
                  .count();
      timeout_ms = static_cast<int>(std::clamp<long>(ms, 0, 100));
    }
    curl_multi_poll(multi, nullptr, 0, timeout_ms, nullptr);
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
