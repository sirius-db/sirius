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

#include "io/s3/s3_blocking_ioctx.hpp"

#include "exec/thread_pool.hpp"
#include "io/prefetching_cache.hpp"
#include "io/s3/s3_blocking_io_object.hpp"
#include "io/s3/s3_request_authorizer.hpp"
#include "io/sirius_datasource.hpp"
#include "io/uri_parser.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime.h>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <future>
#include <optional>
#include <random>
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
// We init once per process on first s3_blocking_ioctx construction. curl_global_cleanup
// is intentionally not called: it races with late-destroyed static curl
// handles in third-party libs and AWS SDK has hit this too. The leak is bounded.
struct curl_global_init_once {
  curl_global_init_once()
  {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != 0)
      throw std::runtime_error("s3_blocking_ioctx: curl_global_init failed");
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
// Always consumes the full incoming chunk so curl can complete the request
// (otherwise libcurl reports CURLE_WRITE_ERROR and retry classification
// loses the underlying HTTP status). Overflow detection happens after the
// fact via @c total_received vs the requested size — the success path
// (@c range_get) checks @c written == requested-size and @c total_received
// == @c written; a discrepancy means the server sent more than requested.
// For error responses (429 with "slow down" body, etc.), we deliberately
// drop the body bytes that exceed @c capacity rather than failing the
// transfer — the retry path doesn't need the error body content.
struct buf_sink {
  std::uint8_t* dst;
  std::size_t capacity;
  std::size_t written;
  std::size_t total_received;
};
std::size_t curl_write_buf(void* ptr, std::size_t size, std::size_t nmemb, void* userdata)
{
  auto* sink  = static_cast<buf_sink*>(userdata);
  auto nbytes = size * nmemb;
  auto room   = sink->capacity > sink->written ? sink->capacity - sink->written : std::size_t{0};
  auto copy   = std::min(nbytes, room);
  if (copy > 0) {
    std::memcpy(sink->dst + sink->written, ptr, copy);
    sink->written += copy;
  }
  sink->total_received += nbytes;
  return nbytes;
}

// libcurl header callback that snags the @c Content-Range (for 206 validation)
// and @c Retry-After (for retry backoff override on 429 / 503) values.
struct header_capture {
  std::string content_range;
  std::string retry_after;
};

// Trim whitespace + trailing CRLF in-place on a string_view.
std::string_view trim_header_value(std::string_view v)
{
  while (!v.empty() && (v.front() == ' ' || v.front() == '\t'))
    v.remove_prefix(1);
  while (!v.empty() &&
         (v.back() == '\r' || v.back() == '\n' || v.back() == ' ' || v.back() == '\t'))
    v.remove_suffix(1);
  return v;
}

// Case-insensitive prefix match of `prefix` on `line`. `prefix` MUST already
// be lowercase. Returns the trimmed value substring on match, std::nullopt
// otherwise.
std::optional<std::string_view> match_header(std::string_view line, std::string_view prefix)
{
  if (line.size() < prefix.size()) return std::nullopt;
  for (std::size_t i = 0; i < prefix.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(line[i])) !=
        static_cast<unsigned char>(prefix[i])) {
      return std::nullopt;
    }
  }
  return trim_header_value(line.substr(prefix.size()));
}

std::size_t curl_header_capture(char* buffer, std::size_t size, std::size_t nmemb, void* userdata)
{
  auto* hc          = static_cast<header_capture*>(userdata);
  std::size_t total = size * nmemb;
  std::string_view l(buffer, total);
  if (auto v = match_header(l, "content-range:")) { hc->content_range = std::string(*v); }
  if (auto v = match_header(l, "retry-after:")) { hc->retry_after = std::string(*v); }
  return total;
}

// ---------------------------------------------------------------------------
// Retry classification + backoff helpers (P2)
// ---------------------------------------------------------------------------

// HTTP statuses Sirius retries: 408 Request Timeout, 429 Too Many Requests,
// 5xx server-side. 4xx are deliberately NOT retried (auth / not-found are
// non-transient; retrying would just delay surfacing the real error).
bool is_retriable_status(long http_status)
{
  if (http_status == 408 || http_status == 429) return true;
  return http_status >= 500 && http_status < 600;
}

// libcurl transport-level failures that are usually transient: connection
// failure, timeout, mid-stream drops. Deliberately narrow — protocol errors
// (CURLE_RECV_ERROR is borderline but included since MinIO under load can
// drop reads mid-body) are kept retriable; logic errors like
// CURLE_URL_MALFORMAT are surfaced immediately.
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

// Parse the seconds form of "Retry-After: <N>". The HTTP-date form is
// allowed by RFC but rare in S3-compatible stores; treat malformed / date
// forms as "absent" so the caller falls back to the computed backoff.
std::optional<std::chrono::milliseconds> parse_retry_after_seconds(std::string_view v)
{
  if (v.empty()) return std::nullopt;
  // Reject obvious non-numeric (HTTP-date form starts with weekday name).
  for (char c : v) {
    if (!std::isdigit(static_cast<unsigned char>(c))) return std::nullopt;
  }
  try {
    auto seconds = std::stoull(std::string{v});
    return std::chrono::milliseconds{seconds * 1000};
  } catch (...) {
    return std::nullopt;
  }
}

// Backoff for retry index N (0 == first retry, after the initial attempt
// failed): base * 2^N. Jitter added by caller. When base == 0, returns
// zero — caller can still add a Retry-After override if present.
std::chrono::milliseconds compute_backoff_base(std::chrono::milliseconds base,
                                               std::size_t retry_idx)
{
  if (base == std::chrono::milliseconds{0}) return std::chrono::milliseconds{0};
  // Cap exponent at 16 (2^16 ≈ base * 65536) to avoid overflow on long runs.
  auto shift = std::min<std::size_t>(retry_idx, 16);
  return base * (1ULL << shift);
}

std::chrono::milliseconds compute_jitter(std::chrono::milliseconds jitter)
{
  if (jitter <= std::chrono::milliseconds{0}) return std::chrono::milliseconds{0};
  thread_local std::mt19937_64 rng{std::random_device{}()};
  std::uniform_int_distribution<long long> dist{0, jitter.count()};
  return std::chrono::milliseconds{dist(rng)};
}

// Wait between retry attempts. Caller passes the retry index (0 == first
// retry). If `retry_after` is present and honored, it replaces the
// computed backoff (capped at 30 seconds). Otherwise base * 2^retry_idx
// + uniform[0, jitter]. No-op when total wait is zero.
void sleep_for_retry(std::size_t retry_idx,
                     std::chrono::milliseconds base,
                     std::chrono::milliseconds jitter,
                     std::optional<std::chrono::milliseconds> retry_after)
{
  constexpr std::chrono::milliseconds kRetryAfterCap{30 * 1000};
  std::chrono::milliseconds wait{0};
  if (retry_after) {
    wait = std::min(*retry_after, kRetryAfterCap);
  } else {
    wait = compute_backoff_base(base, retry_idx) + compute_jitter(jitter);
  }
  if (wait > std::chrono::milliseconds{0}) { std::this_thread::sleep_for(wait); }
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

// Build a libcurl header list ("Name: value") from the authorizer-supplied
// headers. Returns nullptr for an empty header set (the presigned-authorizer
// case), so callers can pass the result straight to CURLOPT_HTTPHEADER or
// append additional headers (e.g. Range) onto it. curl_slist_free_all is a
// no-op on nullptr, so callers always free unconditionally.
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

// ===========================================================================
// handle_slot
// ===========================================================================

void s3_blocking_ioctx::handle_slot::reset()
{
  if (owner && easy) { owner->release_handle(handle_slot{owner, easy}); }
  owner = nullptr;
  easy  = nullptr;
}

// ===========================================================================
// s3_blocking_ioctx lifecycle
// ===========================================================================

s3_blocking_ioctx::s3_blocking_ioctx(s3_ioctx_config config) : _cfg(std::move(config))
{
  if (!_cfg.creds)
    throw std::invalid_argument("s3_blocking_ioctx: s3_request_authorizer is required");
  if (_cfg.max_connections == 0) _cfg.max_connections = 1;

  ensure_curl_inited();
}

s3_blocking_ioctx::~s3_blocking_ioctx()
{
  // P3.9 teardown ordering: destroy the prefetch cache (and its worker /
  // evictor jthreads) BEFORE shutting down the curl handle pool. The cache
  // workers call back into host_read_io / device_read_io on cache miss; if
  // the curl pool tears down first, an in-flight load would dereference
  // freed handles. _cache lives on the sirius_ioctx base, so the default
  // destructor would only reach it AFTER ~s3_blocking_ioctx() returns — too late
  // because shutdown() has already invalidated _free_handles by then.
  _cache.reset();
  shutdown();
}

void s3_blocking_ioctx::shutdown()
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

std::unique_ptr<io::sirius_datasource> s3_blocking_ioctx::make_datasource(
  std::shared_ptr<sirius_io_object> io_object)
{
  return std::make_unique<io::sirius_datasource>(shared_from_this(), std::move(io_object));
}

std::shared_ptr<sirius_io_object> s3_blocking_ioctx::create_io_object(std::string path)
{
  // Use PR1's URI parser. It throws std::invalid_argument on:
  //   - empty URI / empty scheme / relative bare path
  //   - empty bucket (host) / empty key (e.g. "s3://bucket" or "s3://bucket/")
  //   - malformed percent-encoding
  // and normalizes the scheme to lowercase. We let those propagate as-is for
  // C5 (non-S3 / malformed path) — caller sees std::invalid_argument.
  auto parsed = sirius::io::parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("s3_blocking_ioctx::create_io_object: unsupported scheme '" +
                                parsed.scheme + "' (only 's3' is supported)");
  }
  // parsed.host == bucket, parsed.path == key (one separator slash consumed
  // by the parser; any further leading slashes survive into the key per S3
  // REST semantics).
  auto size = head_object_size(parsed.host, parsed.path);
  return std::make_shared<s3_blocking_io_object>(
    std::move(parsed.host), std::move(parsed.path), size, std::move(path));
}

bool s3_blocking_ioctx::supports(std::string_view path) const
{
  // Capability check semantics from the base-class doxygen: "can serve reads
  // for @p path". A prefix-only check is too permissive — it would return
  // true for "s3://", "s3://bucket", "s3://bucket/" (malformed / no key),
  // which would lock multi-ioctx dispatch onto this backend only for
  // create_io_object() to throw later. Tighten by also verifying the URI
  // shape via sirius::io::parse (non-empty bucket + non-empty key, no
  // malformed percent-encoding); fall through to other backends on failure.
  //
  // Quick scheme prefix gate first to avoid parser cost on obvious non-S3
  // paths (the common case in mixed-backend dispatch — file:// / bare
  // absolute paths picked up by uring_ioctx::supports()).
  constexpr std::string_view kPrefix = "s3://";
  if (path.size() < kPrefix.size()) return false;
  for (std::size_t i = 0; i < kPrefix.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(path[i])) !=
        static_cast<unsigned char>(kPrefix[i])) {
      return false;
    }
  }
  // supports() must not throw — multi-ioctx dispatch iterates supports()
  // over a vector via find_if and a throw would prevent fallback to other
  // backends. Catch parser exceptions and return false. Deliberately no
  // HEAD here: that would defeat the cheap-capability-probe contract
  // (supports() is O(parse), not O(network); create_io_object() does the
  // HEAD).
  try {
    auto parsed = sirius::io::parse(path);
    return parsed.scheme == "s3" && !parsed.host.empty() && !parsed.path.empty();
  } catch (std::invalid_argument const&) {
    return false;
  }
}

// ===========================================================================
// handle pool
// ===========================================================================

s3_blocking_ioctx::handle_slot s3_blocking_ioctx::acquire_handle()
{
  std::unique_lock lk{_pool_mtx};
  _pool_cv.wait(lk, [&] {
    return _shutdown || !_free_handles.empty() || _total_handles < _cfg.max_connections;
  });
  if (_shutdown) throw std::runtime_error("s3_blocking_ioctx: acquire_handle after shutdown");

  if (!_free_handles.empty()) {
    void* h = _free_handles.back();
    _free_handles.pop_back();
    return handle_slot{this, h};
  }
  // Grow the pool on demand up to max_connections.
  CURL* h = curl_easy_init();
  if (!h) throw std::runtime_error("s3_blocking_ioctx: curl_easy_init failed");
  ++_total_handles;
  return handle_slot{this, h};
}

void s3_blocking_ioctx::release_handle(handle_slot slot)
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

// Per-attempt presigned-URL TTL: cover one libcurl request + clock skew, NOT the
// whole scan/task lifetime (newplan §30.1 / Codex F4). The URL is freshly minted
// each attempt, so a short TTL limits leaked-URL blast radius. Falls back to
// 5 min when no per-request curl timeout is configured.
static std::chrono::seconds presign_ttl_for_attempt(long request_timeout_s)
{
  constexpr long kSkewSeconds = 60;
  long const base             = request_timeout_s > 0 ? request_timeout_s : 300;
  return std::chrono::seconds{base + kSkewSeconds};
}

std::size_t s3_blocking_ioctx::head_object_size(std::string_view bucket, std::string_view key)
{
  // Auth is delegated to the request authorizer. In presigned mode it lives in
  // the URL's query string (X-Amz-Signature etc.) and the returned header set
  // is empty; in header-signing mode the authorizer also returns Authorization
  // / x-amz-* headers that we attach verbatim. Either way the request is
  // re-authorized each attempt — if a retry crosses the authorizer's TTL
  // window the next attempt still gets a valid URL / signature.
  auto const max_attempts = std::max<std::size_t>(_cfg.max_retry_attempts, 1);
  auto slot               = acquire_handle();
  auto* h                 = static_cast<CURL*>(slot.easy);

  std::string last_diag;
  long last_http_code     = 0;
  CURLcode last_curl_code = CURLE_OK;

  for (std::size_t attempt = 1; attempt <= max_attempts; ++attempt) {
    curl_easy_reset(h);
    s3_authorized_request req =
      _cfg.creds->authorize(s3_object_ref{std::string{bucket}, std::string{key}},
                            s3_request_method::HEAD,
                            presign_ttl_for_attempt(_cfg.request_timeout_s));
    curl_slist* hdrs = build_header_slist(req.headers);
    header_capture hc;
    curl_easy_setopt(h, CURLOPT_URL, req.url.c_str());
    curl_easy_setopt(h, CURLOPT_NOBODY, 1L);
    if (hdrs) curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, curl_header_capture);
    curl_easy_setopt(h, CURLOPT_HEADERDATA, &hc);
    if (_cfg.request_timeout_s > 0) curl_easy_setopt(h, CURLOPT_TIMEOUT, _cfg.request_timeout_s);
    // TLS: verify against a custom CA bundle when configured (on-prem / self-signed /
    // local-HTTPS); empty -> libcurl's system CA bundle (AWS). tls_verify=false disables checks.
    if (!_cfg.ca_bundle_path.empty())
      curl_easy_setopt(h, CURLOPT_CAINFO, _cfg.ca_bundle_path.c_str());
    if (!_cfg.tls_verify) {
      curl_easy_setopt(h, CURLOPT_SSL_VERIFYPEER, 0L);
      curl_easy_setopt(h, CURLOPT_SSL_VERIFYHOST, 0L);
    }

    last_curl_code = curl_easy_perform(h);
    curl_slist_free_all(hdrs);
    last_http_code = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &last_http_code);

    if (last_curl_code == CURLE_OK && last_http_code >= 200 && last_http_code < 300) {
      curl_off_t content_len = -1;
      curl_easy_getinfo(h, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_len);
      if (content_len < 0)
        throw std::runtime_error("s3_blocking_ioctx: HEAD response missing Content-Length");
      return static_cast<std::size_t>(content_len);
    }

    bool retriable = (last_curl_code != CURLE_OK && is_retriable_curl_code(last_curl_code)) ||
                     (last_curl_code == CURLE_OK && is_retriable_status(last_http_code));
    last_diag = last_curl_code != CURLE_OK
                  ? std::string{"libcurl "} + curl_easy_strerror(last_curl_code)
                  : "HTTP " + std::to_string(last_http_code);

    if (!retriable || attempt >= max_attempts) break;

    std::optional<std::chrono::milliseconds> retry_after;
    if (_cfg.honor_retry_after) { retry_after = parse_retry_after_seconds(hc.retry_after); }
    sleep_for_retry(attempt - 1, _cfg.retry_backoff_base, _cfg.retry_jitter, retry_after);
  }

  std::ostringstream os;
  os << "s3_blocking_ioctx: HEAD " << bucket << "/" << key << " failed after " << max_attempts
     << " attempt(s); last: " << last_diag;
  throw std::runtime_error(os.str());
}

std::size_t s3_blocking_ioctx::range_get(std::string_view bucket,
                                         std::string_view key,
                                         std::size_t offset,
                                         std::size_t size,
                                         std::uint8_t* dst)
{
  if (size == 0) return 0;
  auto const max_attempts = std::max<std::size_t>(_cfg.max_retry_attempts, 1);
  auto slot               = acquire_handle();
  auto* h                 = static_cast<CURL*>(slot.easy);

  std::string last_diag;
  long last_http_code     = 0;
  CURLcode last_curl_code = CURLE_OK;

  for (std::size_t attempt = 1; attempt <= max_attempts; ++attempt) {
    curl_easy_reset(h);

    // Auth is delegated to the request authorizer. In header-signing mode it
    // returns Authorization / x-amz-* headers (attached first); in presigned
    // mode the header set is empty and auth lives in the URL query, which
    // deliberately leaves the Range header unsigned (SignedHeaders=host only)
    // so callers may add Range / Accept / etc. without breaking the signature.
    // The Range header is appended last in both modes.
    s3_authorized_request req =
      _cfg.creds->authorize(s3_object_ref{std::string{bucket}, std::string{key}},
                            s3_request_method::GET,
                            presign_ttl_for_attempt(_cfg.request_timeout_s));

    std::ostringstream range_os;
    range_os << "Range: bytes=" << offset << "-" << (offset + size - 1);
    std::string range_header = range_os.str();
    curl_slist* hdrs         = build_header_slist(req.headers);
    hdrs                     = curl_slist_append(hdrs, range_header.c_str());

    buf_sink sink{dst, size, 0, 0};
    header_capture hc;
    curl_easy_setopt(h, CURLOPT_URL, req.url.c_str());
    curl_easy_setopt(h, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, curl_write_buf);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &sink);
    curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, curl_header_capture);
    curl_easy_setopt(h, CURLOPT_HEADERDATA, &hc);
    if (_cfg.request_timeout_s > 0) curl_easy_setopt(h, CURLOPT_TIMEOUT, _cfg.request_timeout_s);
    // TLS: verify against a custom CA bundle when configured (on-prem / self-signed /
    // local-HTTPS); empty -> libcurl's system CA bundle (AWS). tls_verify=false disables checks.
    if (!_cfg.ca_bundle_path.empty())
      curl_easy_setopt(h, CURLOPT_CAINFO, _cfg.ca_bundle_path.c_str());
    if (!_cfg.tls_verify) {
      curl_easy_setopt(h, CURLOPT_SSL_VERIFYPEER, 0L);
      curl_easy_setopt(h, CURLOPT_SSL_VERIFYHOST, 0L);
    }

    last_curl_code = curl_easy_perform(h);
    curl_slist_free_all(hdrs);
    last_http_code = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &last_http_code);

    if (last_curl_code == CURLE_OK) {
      // Validate the response actually covers the byte range we asked for.
      // A misbehaving / non-Range-aware store could otherwise hand us bytes
      // from the wrong offset under a 200 response and we'd silently feed
      // corrupt data into the parquet reader.
      auto fail = [&](std::string_view why) {
        std::ostringstream os;
        os << "s3_blocking_ioctx: GET " << bucket << "/" << key << " offset=" << offset
           << " size=" << size << " returned HTTP " << last_http_code << "; " << why;
        if (!hc.content_range.empty()) os << " (Content-Range: " << hc.content_range << ")";
        throw std::runtime_error(os.str());
      };

      if (last_http_code == 206) {
        // sink.written == bytes copied into dst (capped at requested size);
        // sink.total_received == bytes server actually delivered. Mismatch
        // between total_received and the requested size flags either short
        // body (server hung up early) or overflow (server sent more than
        // asked). Both are anomalies on a 206.
        if (sink.total_received != size || sink.written != size)
          fail("206 body length " + std::to_string(sink.total_received) + " != requested size");
        if (!hc.content_range.empty()) {
          std::size_t got_start = 0;
          std::size_t got_end   = 0;
          if (!parse_content_range(hc.content_range, got_start, got_end))
            fail("malformed Content-Range");
          if (got_start != offset || got_end - got_start + 1 != size)
            fail("Content-Range mismatch");
        }
        _bytes_read_total.fetch_add(sink.written, std::memory_order_relaxed);
        return sink.written;
      }
      if (last_http_code == 200) {
        // Some stores collapse a small ranged GET to a 200 with the full body.
        // Only accept that when the request started at offset 0 *and* the body
        // we got happens to match the requested length — both conditions guard
        // against silently grabbing the head of an object when we asked for an
        // interior range. Otherwise treat as a non-retriable response anomaly.
        if (offset != 0 || sink.total_received != size)
          fail("server returned 200 (no Range honored)");
        _bytes_read_total.fetch_add(sink.written, std::memory_order_relaxed);
        return sink.written;
      }
    }

    bool retriable = (last_curl_code != CURLE_OK && is_retriable_curl_code(last_curl_code)) ||
                     (last_curl_code == CURLE_OK && is_retriable_status(last_http_code));
    last_diag = last_curl_code != CURLE_OK
                  ? std::string{"libcurl "} + curl_easy_strerror(last_curl_code)
                  : "HTTP " + std::to_string(last_http_code);

    if (!retriable || attempt >= max_attempts) break;

    std::optional<std::chrono::milliseconds> retry_after;
    if (_cfg.honor_retry_after) { retry_after = parse_retry_after_seconds(hc.retry_after); }
    sleep_for_retry(attempt - 1, _cfg.retry_backoff_base, _cfg.retry_jitter, retry_after);
  }

  std::ostringstream os;
  os << "s3_blocking_ioctx: GET " << bucket << "/" << key << " offset=" << offset
     << " size=" << size << " failed after " << max_attempts << " attempt(s); last: " << last_diag;
  throw std::runtime_error(os.str());
}

// ===========================================================================
// host read APIs
// ===========================================================================

std::size_t s3_blocking_ioctx::host_read_io(sirius_io_object& obj,
                                            std::size_t offset,
                                            std::size_t size,
                                            std::uint8_t* dst)
{
  auto& so = dynamic_cast<s3_blocking_io_object&>(obj);
  // Clip to object size so reads past EOF return short instead of having S3
  // throw 416 (matches the templated_ioctx generic behavior on local files).
  size = std::min(size, so.size() > offset ? so.size() - offset : std::size_t{0});
  if (size == 0) return 0;
  return range_get(so.bucket(), so.key(), offset, size, dst);
}

namespace {

// Dispatch an async I/O completion onto either an injected static_thread_pool
// (preferred — scan_manager / SiriusContext wires the pool's lifetime to the
// engine) or a detached worker (fallback for standalone-test scenarios where
// no scan_manager owns a pool). Either path invokes the handler exactly once
// with either (bytes_written, nullptr) on success or (0, exception_ptr) on
// throw. The s3_blocking_ioctx never owns the pool — it's caller-controlled and
// outlives the ioctx.
void dispatch_async(exec::static_thread_pool* pool,
                    io_completion_handler handler,
                    std::function<std::size_t()> op)
{
  auto task = [handler = std::move(handler), op = std::move(op)]() mutable {
    try {
      auto bytes = op();
      handler(bytes, nullptr);
    } catch (...) {
      handler(0, std::current_exception());
    }
  };
  if (pool != nullptr) {
    pool->schedule(std::move(task));
  } else {
    std::thread(std::move(task)).detach();
  }
}

}  // namespace

void s3_blocking_ioctx::host_read_async_io(sirius_io_object& obj,
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
  dispatch_async(
    _cfg.async_thread_pool, std::move(handler), [self, obj_owner, offset, size, dst]() {
      return self->host_read_io(*obj_owner, offset, size, dst);
    });
}

namespace {

// Shared completion state for the per-range fan-out path. One ranges_ctx is
// created per host_read_ranges_async_io call when a thread pool is present;
// each pool task references it via shared_ptr. The last task to finish
// (pending decrements to 0) invokes the user handler — either with the
// summed byte count or with the first-captured exception_ptr.
struct ranges_ctx {
  std::shared_ptr<sirius::io::s3::s3_blocking_ioctx> self;
  std::shared_ptr<sirius::io::sirius_io_object> obj_owner;
  std::atomic<std::size_t> pending;
  std::atomic<std::size_t> total_bytes{0};
  std::mutex error_mtx;
  std::exception_ptr first_error;
  sirius::io::io_completion_handler handler;
};

}  // namespace

void s3_blocking_ioctx::host_read_ranges_async_io(
  sirius_io_object& obj,
  std::vector<cudf::io::text::byte_range_info> const& ranges,
  std::span<cudf::host_span<std::byte>> dst,
  io_completion_handler handler)
{
  // Async-only contract: every exit path delivers exactly one handler
  // invocation, never throws synchronously. The cache worker
  // (prefetching_cache::worker_loop) calls this without an outer try/catch
  // and holds resources (allocated chunks, budget slot) that would leak
  // if a sync throw escaped the call. Validation errors and lookup throws
  // (shared_from_this / dynamic_cast / dst-size) are captured in Phase 1
  // and dispatched as an exception_ptr through the handler in Phase 2.
  auto* pool = _cfg.async_thread_pool;

  // -- Phase 1: validation + setup. Any throw here is caught and surfaces
  //    through the handler below; no resource leaks because nothing has
  //    been scheduled yet.
  std::shared_ptr<s3_blocking_ioctx> self_kept;
  std::shared_ptr<sirius_io_object> obj_kept;
  std::vector<std::size_t> clipped_sizes;
  std::vector<std::size_t> offsets;
  std::vector<std::uint8_t*> dst_ptrs;
  std::exception_ptr setup_error;

  try {
    if (ranges.size() != dst.size())
      throw std::invalid_argument("s3_blocking_ioctx::host_read_ranges: ranges/dst size mismatch");
    self_kept = std::static_pointer_cast<s3_blocking_ioctx>(shared_from_this());
    obj_kept  = obj.shared_from_this();
    if (!ranges.empty()) {
      auto& so                 = dynamic_cast<s3_blocking_io_object&>(obj);
      std::size_t const obj_sz = so.size();
      clipped_sizes.reserve(ranges.size());
      offsets.reserve(ranges.size());
      dst_ptrs.reserve(ranges.size());
      for (std::size_t i = 0; i < ranges.size(); ++i) {
        auto offset  = static_cast<std::size_t>(ranges[i].offset());
        auto size    = static_cast<std::size_t>(ranges[i].size());
        auto clipped = std::min(size, obj_sz > offset ? obj_sz - offset : std::size_t{0});
        if (dst[i].size() < clipped)
          throw std::invalid_argument("s3_blocking_ioctx::host_read_ranges: dst span too small");
        clipped_sizes.push_back(clipped);
        offsets.push_back(offset);
        dst_ptrs.push_back(reinterpret_cast<std::uint8_t*>(dst[i].data()));
      }
    }
  } catch (...) {
    setup_error = std::current_exception();
  }

  // -- Phase 2a: no-pool fallback. dispatch_async already wraps the inner
  //    op in try/catch and delivers exceptions through the handler, so a
  //    setup_error here just flows through the same path.
  if (pool == nullptr) {
    if (setup_error) {
      std::thread([h = std::move(handler), ep = setup_error]() mutable { h(0, ep); }).detach();
      return;
    }
    std::vector<cudf::host_span<std::byte>> dst_owned(dst.begin(), dst.end());
    dispatch_async(nullptr,
                   std::move(handler),
                   [self      = std::move(self_kept),
                    obj_owner = std::move(obj_kept),
                    ranges,
                    dst_owned = std::move(dst_owned)]() mutable {
                     return self->host_read_ranges_serial_impl(*obj_owner, ranges, dst_owned);
                   });
    return;
  }

  // -- Phase 2b: pool path. Setup errors and the empty-ranges shortcut
  //    both deliver via a single pool task to preserve the "handler runs
  //    on a pool thread" contract that the pre-P1.6 dispatch_async path
  //    established.
  if (setup_error) {
    pool->schedule([h = std::move(handler), ep = setup_error]() mutable { h(0, ep); });
    return;
  }
  if (ranges.empty()) {
    pool->schedule([h = std::move(handler)]() mutable { h(0, nullptr); });
    return;
  }

  // -- Phase 2c: pool path fan-out. One task per range, shared ctx tracks
  //    completion; last task to decrement pending fires the handler.
  auto ctx = std::make_shared<ranges_ctx>();
  ctx->self.swap(self_kept);
  ctx->obj_owner.swap(obj_kept);
  ctx->pending.store(ranges.size(), std::memory_order_relaxed);
  ctx->handler = std::move(handler);

  for (std::size_t i = 0; i < ranges.size(); ++i) {
    auto offset = offsets[i];
    auto size   = clipped_sizes[i];
    auto* dst_i = dst_ptrs[i];
    pool->schedule([ctx, offset, size, dst_i]() {
      try {
        std::size_t bytes = 0;
        if (size > 0) { bytes = ctx->self->host_read_io(*ctx->obj_owner, offset, size, dst_i); }
        ctx->total_bytes.fetch_add(bytes, std::memory_order_relaxed);
      } catch (...) {
        std::scoped_lock lk(ctx->error_mtx);
        if (!ctx->first_error) ctx->first_error = std::current_exception();
      }
      if (ctx->pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (ctx->first_error) {
          ctx->handler(0, ctx->first_error);
        } else {
          ctx->handler(ctx->total_bytes.load(std::memory_order_relaxed), nullptr);
        }
      }
    });
  }
}

std::size_t s3_blocking_ioctx::host_read_ranges_serial_impl(
  sirius_io_object& obj,
  std::vector<cudf::io::text::byte_range_info> const& ranges,
  std::span<cudf::host_span<std::byte>> dst)
{
  if (ranges.size() != dst.size())
    throw std::invalid_argument("s3_blocking_ioctx::host_read_ranges: ranges/dst size mismatch");
  auto& so                 = dynamic_cast<s3_blocking_io_object&>(obj);
  std::size_t const obj_sz = so.size();
  std::size_t total        = 0;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    auto offset  = static_cast<std::size_t>(ranges[i].offset());
    auto size    = static_cast<std::size_t>(ranges[i].size());
    auto clipped = std::min(size, obj_sz > offset ? obj_sz - offset : std::size_t{0});
    if (dst[i].size() < clipped)
      throw std::invalid_argument("s3_blocking_ioctx::host_read_ranges: dst span too small");
    total += host_read_io(obj, offset, clipped, reinterpret_cast<std::uint8_t*>(dst[i].data()));
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

std::size_t s3_blocking_ioctx::device_read_io(sirius_io_object& obj,
                                              std::size_t offset,
                                              std::size_t size,
                                              std::uint8_t* dst,
                                              rmm::cuda_stream_view stream)
{
  if (size == 0) { return 0; }

  // Fallback: no caller-supplied host FSMR (standalone / unit-test
  // construction). One std::vector staging buffer sized to the whole
  // request — unbounded, but kept so direct s3_blocking_ioctx construction works.
  if (_cfg.host_memory_resource == nullptr) {
    std::vector<std::uint8_t> host(size);
    auto got = host_read_io(obj, offset, size, host.data());
    if (got > 0) {
      auto rc = cudaMemcpyAsync(dst, host.data(), got, cudaMemcpyHostToDevice, stream.value());
      if (rc != cudaSuccess) {
        throw std::runtime_error(
          std::string("s3_blocking_ioctx::device_read_io cudaMemcpyAsync failed: ") +
          cudaGetErrorString(rc));
      }
      if (auto sync_rc = cudaStreamSynchronize(stream.value()); sync_rc != cudaSuccess) {
        throw std::runtime_error(
          std::string("s3_blocking_ioctx::device_read_io stream sync failed: ") +
          cudaGetErrorString(sync_rc));
      }
    }
    return got;
  }

  // FSMR path: stage the H2D bounce through a single bounded host block,
  // reused across chunks, so transient host memory is capped at one block
  // regardless of the request size.
  auto* host_mr           = _cfg.host_memory_resource;
  std::size_t const chunk = host_mr->get_block_size();
  if (chunk == 0) {
    throw std::runtime_error("s3_blocking_ioctx::device_read_io: FSMR block size is zero");
  }

  cucascade::memory::fixed_size_host_memory_resource::fixed_multiple_blocks_allocation staging;
  try {
    staging = host_mr->allocate_multiple_blocks(chunk);
  } catch (rmm::out_of_memory const& e) {
    // Rethrow as rmm::out_of_memory (not std::runtime_error) so the pipeline's
    // OOM classification (gpu_pipeline_task) still recognizes this and can run
    // its reschedule / downgrade path; only enrich the message with S3 context.
    throw rmm::out_of_memory(
      std::string("s3_blocking_ioctx::device_read_io: out of memory allocating a "
                  "fixed_size_host_memory_resource staging block (object='") +
      obj.object_path() + "' offset=" + std::to_string(offset) + " size=" + std::to_string(size) +
      " chunk=" + std::to_string(chunk) + "): " + e.what());
  } catch (std::exception const& e) {
    throw std::runtime_error(
      std::string("s3_blocking_ioctx::device_read_io: failed to allocate a staging block from the "
                  "fixed_size_host_memory_resource (object='") +
      obj.object_path() + "' offset=" + std::to_string(offset) + " size=" + std::to_string(size) +
      " chunk=" + std::to_string(chunk) + "): " + e.what());
  }
  _fsmr_borrows_total.fetch_add(1, std::memory_order_relaxed);
  auto* host = reinterpret_cast<std::uint8_t*>(staging->at(0).data());

  std::size_t total = 0;
  while (total < size) {
    std::size_t const todo = std::min(chunk, size - total);
    std::size_t const got  = host_read_io(obj, offset + total, todo, host);
    if (got == 0) { break; }  // EOF
    auto rc = cudaMemcpyAsync(dst + total, host, got, cudaMemcpyHostToDevice, stream.value());
    if (rc != cudaSuccess) {
      throw std::runtime_error(
        std::string("s3_blocking_ioctx::device_read_io cudaMemcpyAsync failed: ") +
        cudaGetErrorString(rc));
    }
    // Sync before the next iteration overwrites the shared staging block.
    if (auto sync_rc = cudaStreamSynchronize(stream.value()); sync_rc != cudaSuccess) {
      throw std::runtime_error(
        std::string("s3_blocking_ioctx::device_read_io stream sync failed: ") +
        cudaGetErrorString(sync_rc));
    }
    total += got;
    if (got < todo) { break; }  // short read = EOF
  }
  return total;
}

void s3_blocking_ioctx::device_read_async_io(sirius_io_object& obj,
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
  // detached worker (see host_read_async_io).
  auto self      = shared_from_this();
  auto obj_owner = obj.shared_from_this();
  dispatch_async(
    _cfg.async_thread_pool, std::move(handler), [self, obj_owner, offset, size, dst, stream]() {
      return self->device_read_io(*obj_owner, offset, size, dst, stream);
    });
}

// ===========================================================================
// compute_physical_range
// ===========================================================================

cudf::io::text::byte_range_info s3_blocking_ioctx::compute_physical_range(
  cudf::io::text::byte_range_info logical, std::size_t file_size) const
{
  auto const off  = static_cast<std::size_t>(logical.offset());
  auto const req  = static_cast<std::size_t>(logical.size());
  auto const clip = off >= file_size ? 0UL : std::min(req, file_size - off);
  return cudf::io::text::byte_range_info{static_cast<int64_t>(off), static_cast<int64_t>(clip)};
}

}  // namespace sirius::io::s3
