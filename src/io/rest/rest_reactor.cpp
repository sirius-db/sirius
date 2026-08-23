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

#include "io/rest/rest_reactor.hpp"

#include "cucascade/cuda/event.hpp"
#include "io/details/slot_pool.hpp"
#include "io/rest/curl_handle.hpp"
#include "io/uri_parser.hpp"
#include "log/logging.hpp"

#include <rmm/cuda_device.hpp>

#include <sys/epoll.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <format>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::io::rest {

namespace {

// ---- libcurl callbacks -----------------------------------------------------

/// Write callback: copy curl's bytes into the sink's destination buffer at the
/// running cursor, and ALWAYS report the full incoming size so curl never
/// aborts the transfer with CURLE_WRITE_ERROR.  Overflow past capacity is
/// counted (total_received) but not stored, so a server that ignored the Range
/// header can be detected after the fact.
size_t write_to_sink(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* sink         = static_cast<buf_sink*>(userdata);
  size_t const bytes = size * nmemb;
  sink->total_received += bytes;
  size_t remaining = bytes;
  auto const* src  = reinterpret_cast<uint8_t const*>(ptr);
  // Scatter across the destination buffers in file order; a fused contiguous
  // GET spills from one buffer into the next as each fills.
  //
  // A null buffer inside a multi-buffer sink is a hole — the bytes bridging two
  // fused segments — so its span is stepped over rather than stored, and counted
  // in `written` because the read covered it.  A single-buffer sink is a
  // different animal: null there means a bounce-staged device read whose slot
  // submit() should already have bound (set_data), so a null that survives to
  // here is a bug.  Stop rather than silently discard the body — the short read
  // that follows fails the request instead of reporting success on bytes that
  // went nowhere.
  bool const holes_expected = sink->buffers.size() > 1;
  while (remaining > 0 && sink->active < sink->buffers.size()) {
    iovec& b = sink->buffers[sink->active];
    if (b.iov_base == nullptr && !holes_expected) { break; }
    if (sink->cursor >= b.iov_len) {
      ++sink->active;
      sink->cursor = 0;
      continue;
    }
    size_t const n = std::min(b.iov_len - sink->cursor, remaining);
    if (b.iov_base != nullptr) {
      std::memcpy(static_cast<uint8_t*>(b.iov_base) + sink->cursor, src, n);
    }
    sink->cursor += n;
    sink->written += n;
    src += n;
    remaining -= n;
  }
  return bytes;
}

/// Discard callback for HEAD requests (no body expected, but be defensive).
size_t write_discard(char* /*ptr*/, size_t size, size_t nmemb, void* /*userdata*/)
{
  return size * nmemb;
}

/// Accumulate the whole response body into a std::string (small control-plane
/// responses only — e.g. one ListObjectsV2 XML page).
size_t write_string(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* out = static_cast<std::string*>(userdata);
  out->append(ptr, size * nmemb);
  return size * nmemb;
}

/// Lowercase a byte.
char ascii_lower(char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }

/// Case-insensitively match @p line against "<name>:" and, on a hit, return the
/// trimmed value; otherwise return empty.
std::string match_header(std::string_view line, std::string_view name)
{
  if (line.size() < name.size() + 1) { return {}; }
  for (size_t i = 0; i < name.size(); ++i) {
    if (ascii_lower(line[i]) != ascii_lower(name[i])) { return {}; }
  }
  if (line[name.size()] != ':') { return {}; }
  std::string_view val = line.substr(name.size() + 1);
  // Trim surrounding whitespace and trailing CRLF.
  while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) {
    val.remove_prefix(1);
  }
  while (!val.empty() &&
         (val.back() == '\r' || val.back() == '\n' || val.back() == ' ' || val.back() == '\t')) {
    val.remove_suffix(1);
  }
  return std::string(val);
}

/// Header callback: capture Content-Range and Retry-After.
size_t capture_header(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* hc           = static_cast<header_capture*>(userdata);
  size_t const bytes = size * nitems;
  std::string_view line(buffer, bytes);
  if (auto v = match_header(line, "content-range"); !v.empty()) {
    hc->content_range = std::move(v);
  }
  if (auto v = match_header(line, "retry-after"); !v.empty()) { hc->retry_after = std::move(v); }
  return bytes;
}

/// True iff @p line is an HTTP status line ("HTTP/..."), i.e. the start of a
/// (possibly interim) response's header block within one transfer.
bool is_http_status_line(std::string_view line) noexcept
{
  return line.size() >= 5 && ascii_lower(line[0]) == 'h' && ascii_lower(line[1]) == 't' &&
         ascii_lower(line[2]) == 't' && ascii_lower(line[3]) == 'p' && line[4] == '/';
}

/// Per-attempt capture for the blocking HEAD: Retry-After for backoff plus the
/// object's ETag.  Separate from @c header_capture so the async data-GET path
/// parses nothing it does not consume.  The ETag resets on every status line,
/// so interim responses (proxy CONNECT) within one transfer leave no residue.
struct head_capture {
  std::string retry_after;
  std::string etag;
};

size_t head_header_cb(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* hc           = static_cast<head_capture*>(userdata);
  size_t const bytes = size * nitems;
  std::string_view const line(buffer, bytes);
  if (is_http_status_line(line)) {
    hc->etag.clear();
    hc->retry_after.clear();
  }
  if (auto v = match_header(line, "etag"); !v.empty()) { hc->etag = std::move(v); }
  if (auto v = match_header(line, "retry-after"); !v.empty()) { hc->retry_after = std::move(v); }
  return bytes;
}

/// Shared sink for a suffix-range footer probe: the header callback records the
/// HTTP status (from the status line) plus Content-Range / Retry-After / ETag;
/// the body callback consults @c status to abort a non-206 response before it
/// streams a whole object into us.  @c HEADERDATA and @c WRITEDATA point at the
/// same one.
struct suffix_sink {
  std::vector<std::uint8_t> data;
  std::size_t cap{0};
  std::size_t total_received{0};  // wire bytes, incl. those dropped by cap/abort
  long status{0};
  std::string content_range;
  std::string retry_after;
  std::string etag;
};

/// Header callback for a suffix probe: parse the status code out of the status
/// line so the body callback can abort a non-206 early, and capture the headers
/// the caller needs (Content-Range to verify the 206, Retry-After for backoff,
/// ETag for the probe result).
size_t suffix_header_cb(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* s            = static_cast<suffix_sink*>(userdata);
  size_t const bytes = size * nitems;
  std::string_view const line(buffer, bytes);
  if (is_http_status_line(line)) {
    s->etag.clear();
    s->content_range.clear();
    s->retry_after.clear();
    if (auto const sp = line.find(' '); sp != std::string_view::npos) {
      long code = 0;
      for (size_t i = sp + 1; i < line.size() && line[i] >= '0' && line[i] <= '9'; ++i) {
        code = code * 10 + (line[i] - '0');
      }
      if (code != 0) { s->status = code; }
    }
  }
  if (auto v = match_header(line, "content-range"); !v.empty()) { s->content_range = std::move(v); }
  if (auto v = match_header(line, "retry-after"); !v.empty()) { s->retry_after = std::move(v); }
  if (auto v = match_header(line, "etag"); !v.empty()) { s->etag = std::move(v); }
  return bytes;
}

/// Body callback for a suffix probe: abort a non-206 response (a deliberate
/// short write, surfacing as CURLE_WRITE_ERROR) so a server that ignores the
/// Range or answers 416/4xx never streams a whole object into us; otherwise
/// append up to @c cap bytes and report the full incoming size to curl.
size_t suffix_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* s            = static_cast<suffix_sink*>(userdata);
  size_t const bytes = size * nmemb;
  s->total_received += bytes;
  if (s->status != 206) { return 0; }
  if (s->data.size() < s->cap) {
    size_t const take = std::min(s->cap - s->data.size(), bytes);
    auto const* src   = reinterpret_cast<std::uint8_t const*>(ptr);
    s->data.insert(s->data.end(), src, src + take);
  }
  return bytes;
}

// ---- retry classification --------------------------------------------------

/// HTTP status codes worth retrying (transient server / throttling).  Only the
/// transient 5xx are included: 500 Internal Error, 502 Bad Gateway, 503 Slow
/// Down / Service Unavailable, 504 Gateway Timeout.  Permanent 5xx (501 Not
/// Implemented, 505 HTTP Version Not Supported, ...) are NOT retried — they
/// would only burn the full retry budget on an error that cannot succeed.
bool is_retriable_status(long status) noexcept
{
  return status == 408 || status == 429 || status == 500 || status == 502 || status == 503 ||
         status == 504;
}

/// libcurl error codes worth retrying (transient transport failures).
bool is_retriable_curl(CURLcode rc) noexcept
{
  switch (rc) {
    case CURLE_COULDNT_CONNECT:
    case CURLE_COULDNT_RESOLVE_HOST:
    case CURLE_OPERATION_TIMEDOUT:
    case CURLE_GOT_NOTHING:
    case CURLE_RECV_ERROR:
    case CURLE_SEND_ERROR:
    case CURLE_PARTIAL_FILE:
    case CURLE_SSL_CONNECT_ERROR:
    case CURLE_HTTP2_STREAM: return true;
    default: return false;
  }
}

// ---- per-request helpers ---------------------------------------------------

/// Presigned-URL TTL for a request: the whole-request timeout plus clock-skew
/// headroom, with a sane floor so very short timeouts still leave a usable
/// window.
std::chrono::seconds presign_ttl(const config& cfg) noexcept
{
  long const base = cfg.request_timeout_s > 0 ? cfg.request_timeout_s + 60 : 300;
  return std::chrono::seconds{base};
}

/// Apply per-request TLS + timeout options on top of configure_easy_handle.
void apply_request_opts(CURL* h, const config& cfg)
{
  if (!cfg.ca_bundle_path.empty()) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_CAINFO, cfg.ca_bundle_path.c_str()));
  }
  if (!cfg.tls_verify) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYPEER, 0L));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYHOST, 0L));
  }
  if (cfg.request_timeout_s > 0) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_TIMEOUT, cfg.request_timeout_s));
  }
}

/// Build a header list from the authorizer's headers (empty in presigned mode)
/// plus an optional Range header.
curl_slist_ptr build_header_list(std::vector<std::pair<std::string, std::string>> const& headers,
                                 std::string const* range)
{
  curl_slist* list = nullptr;
  for (auto const& [k, v] : headers) {
    std::string const h = k + ": " + v;
    list                = curl_slist_append(list, h.c_str());
  }
  if (range != nullptr) { list = curl_slist_append(list, range->c_str()); }
  return curl_slist_ptr{list};
}

/// "Range: bytes=<lo>-<hi>" (inclusive end) for [offset, offset+size).
std::string range_header(size_t offset, size_t size)
{
  return std::format("Range: bytes={}-{}", offset, offset + size - 1);
}

/// "Range: bytes=-<n>" — the last @p n bytes of an object (a suffix range).
/// Unlike range_header this needs no prior knowledge of the object's size.
std::string suffix_range_header(size_t n) { return std::format("Range: bytes=-{}", n); }

/// Parse the first-byte position out of a Content-Range value of the form
/// "bytes <first>-<last>/<total>" (the trimmed value captured by the header
/// callback).  Returns nullopt for any value that does not start with a
/// well-formed "bytes <first>-" so the caller can reject an unverifiable 206.
std::optional<size_t> content_range_start(std::string_view cr)
{
  constexpr std::string_view kUnit = "bytes";
  std::string_view sv{cr};
  if (sv.size() < kUnit.size()) { return std::nullopt; }
  for (size_t i = 0; i < kUnit.size(); ++i) {
    if (ascii_lower(sv[i]) != kUnit[i]) { return std::nullopt; }
  }
  sv.remove_prefix(kUnit.size());
  while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
    sv.remove_prefix(1);
  }
  if (sv.empty() || sv.front() < '0' || sv.front() > '9') { return std::nullopt; }
  size_t value = 0;
  size_t i     = 0;
  for (; i < sv.size() && sv[i] >= '0' && sv[i] <= '9'; ++i) {
    value = value * 10 + static_cast<size_t>(sv[i] - '0');
  }
  // A valid first-byte position is immediately followed by '-' (the range
  // separator); anything else ("*", end of string, ...) is not parseable.
  if (i >= sv.size() || sv[i] != '-') { return std::nullopt; }
  return value;
}

/// Backoff before the next attempt: honor a numeric Retry-After (seconds,
/// capped at 30 s) when present and enabled, else exponential base<<attempt
/// plus uniform jitter.
std::chrono::milliseconds compute_backoff(std::size_t attempt,
                                          std::string const& retry_after,
                                          const config& cfg)
{
  if (cfg.honor_retry_after && !retry_after.empty()) {
    try {
      long const secs = std::stol(retry_after);
      if (secs >= 0) {
        return std::min(std::chrono::milliseconds{secs * 1000}, std::chrono::milliseconds{30'000});
      }
    } catch (...) {
      // Non-numeric (HTTP-date) Retry-After: fall through to exponential.
    }
  }
  std::size_t const shift = std::min<std::size_t>(attempt, 16);
  auto const base         = cfg.retry_backoff_base * (std::size_t{1} << shift);
  std::chrono::milliseconds jitter{0};
  if (cfg.retry_jitter.count() > 0) {
    thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<long> dist(0, cfg.retry_jitter.count());
    jitter = std::chrono::milliseconds{dist(rng)};
  }
  return base + jitter;
}

constexpr std::size_t rest_min_segment_bytes = 4UL << 20;
constexpr std::size_t rest_max_segment_bytes = 16UL << 20;

[[nodiscard]] std::size_t dynamic_segment_target(std::size_t backlog,
                                                 std::size_t free_connections) noexcept
{
  free_connections                = std::max<std::size_t>(free_connections, 1);
  auto const bytes_per_connection = backlog / free_connections;
  if (bytes_per_connection >= rest_max_segment_bytes) return rest_max_segment_bytes;
  if (bytes_per_connection > rest_min_segment_bytes) {
    return std::min(bytes_per_connection, rest_max_segment_bytes);
  }
  return rest_min_segment_bytes;
}

[[nodiscard]] std::vector<range> physical_ranges(prepared_io_slice const& slice,
                                                 std::size_t target,
                                                 std::size_t cache_block_size)
{
  if (slice.rng.empty()) return {};
  target = std::clamp(target, rest_min_segment_bytes, rest_max_segment_bytes);

  if (!slice.is_fragmented()) {
    auto const preferred_count = std::max<std::size_t>(1, slice.rng.size / target);
    auto const max_bound_count = 1 + (slice.rng.size - 1) / rest_max_segment_bytes;
    auto const count           = std::max(preferred_count, max_bound_count);
    auto const base            = slice.rng.size / count;
    auto const rem             = slice.rng.size % count;
    std::vector<range> result;
    result.reserve(count);
    auto offset = slice.rng.offset;
    for (std::size_t i = 0; i < count; ++i) {
      auto const bytes = base + (i < rem ? 1 : 0);
      result.push_back(range{offset, bytes});
      offset += bytes;
    }
    return result;
  }

  if (cache_block_size == 0) {
    throw std::runtime_error("rest_reactor: fragmented read requires a cache block size");
  }

  std::vector<range> result;
  range current{};
  for (auto* chunk : slice.h_buffer.fragments()) {
    if (chunk == nullptr || chunk->data == nullptr) {
      throw std::runtime_error("rest_reactor: fragmented read has no cache buffer");
    }
    auto const [fill_lo, fill_hi] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, cache_block_size);
    auto const fill = range{fill_lo, fill_hi - fill_lo};
    if (fill.empty()) continue;
    if (fill.size > rest_max_segment_bytes) {
      throw std::runtime_error("rest_reactor: one cache fill exceeds the REST segment maximum");
    }

    bool const contiguous = !current.empty() && current.end() == fill.offset;
    bool const fits =
      current.size <= rest_max_segment_bytes - std::min(fill.size, rest_max_segment_bytes);
    if (current.empty()) {
      current = fill;
    } else if (contiguous && current.size < target && fits &&
               current.size + fill.size <= rest_max_segment_bytes) {
      current.size += fill.size;
    } else {
      result.push_back(current);
      current = fill;
    }
  }
  if (!current.empty()) result.push_back(current);
  if (result.empty()) {
    throw std::runtime_error("rest_reactor: fragmented read has no physical fill ranges");
  }
  return result;
}

[[nodiscard]] std::vector<iovec> operation_iovecs(prepared_io_slice const& slice,
                                                  range io_rng,
                                                  std::size_t cache_block_size)
{
  std::vector<iovec> result;
  if (slice.is_contiguous()) {
    auto* base = std::get<std::uint8_t*>(slice.h_buffer.buffer);
    result.push_back(iovec{base + (io_rng.offset - slice.rng.offset), io_rng.size});
    return result;
  }
  if (slice.is_staged()) return result;

  std::size_t covered = 0;
  for (auto* chunk : slice.h_buffer.fragments()) {
    auto const [fill_lo, fill_hi] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, cache_block_size);
    auto const overlap = intersect(io_rng, range{fill_lo, fill_hi - fill_lo});
    if (overlap.empty()) continue;
    result.push_back(iovec{chunk->data + (overlap.offset - chunk->offset), overlap.size});
    covered += overlap.size;
  }
  if (covered != io_rng.size) {
    throw std::runtime_error("rest_reactor: cache fragments do not cover the physical range");
  }
  return result;
}

[[nodiscard]] std::vector<cache::cached_chunk*> operation_chunks(prepared_io_slice const& slice,
                                                                 range io_rng,
                                                                 std::size_t cache_block_size)
{
  std::vector<cache::cached_chunk*> result;
  if (!slice.is_fragmented()) return result;
  for (auto* chunk : slice.h_buffer.fragments()) {
    auto const [fill_lo, fill_hi] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, cache_block_size);
    if (!intersect(io_rng, range{fill_lo, fill_hi - fill_lo}).empty()) { result.push_back(chunk); }
  }
  return result;
}

}  // namespace

shared_byte_span make_shared_byte_span(std::vector<std::uint8_t> bytes)
{
  auto owner = std::make_shared<detail::byte_storage>(std::move(bytes));
  // Aliasing constructor: shares `owner`'s control block (keeping the buffer
  // alive) while the pointer itself refers to the span member inside it.
  return shared_byte_span{owner, &owner->view};
}

std::optional<size_t> content_range_total(std::string_view cr)
{
  constexpr std::string_view kUnit = "bytes";
  std::string_view sv{cr};
  if (sv.size() < kUnit.size()) { return std::nullopt; }
  for (size_t i = 0; i < kUnit.size(); ++i) {
    if (ascii_lower(sv[i]) != kUnit[i]) { return std::nullopt; }
  }
  sv.remove_prefix(kUnit.size());
  while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
    sv.remove_prefix(1);
  }
  // The range part must be a satisfied "<first>-<last>", never "*": a leading
  // digit both rejects "bytes */..." and confirms a total follows the '/'.
  if (sv.empty() || sv.front() < '0' || sv.front() > '9') { return std::nullopt; }
  auto const slash = sv.find('/');
  if (slash == std::string_view::npos) { return std::nullopt; }
  std::string_view const total = sv.substr(slash + 1);
  if (total.empty() || total.front() < '0' || total.front() > '9') { return std::nullopt; }
  size_t value = 0;
  for (char const c : total) {
    if (c < '0' || c > '9') { break; }
    value = value * 10 + static_cast<size_t>(c - '0');
  }
  return value;
}

// ---------------------------------------------------------------------------
// construction / lifecycle
// ---------------------------------------------------------------------------

rest_reactor::rest_reactor(std::shared_ptr<reactor_context> ctx, std::string_view tname)
  : _ctx(std::move(ctx)), _tname(tname)
{
  if (!_ctx) { throw std::invalid_argument("rest_reactor: reactor_context must be non-null"); }
  _config = _ctx->cfg();
  if (!_ctx->authorizer()) {
    throw std::invalid_argument("rest_reactor: context authorizer must be non-null");
  }
  if (_config.max_connections == 0) {
    throw std::invalid_argument("rest_reactor: max_connections must be > 0");
  }
  if (_config.max_retry_attempts == 0) { _config.max_retry_attempts = 1; }

  // Touch the process-wide curl context so global init + the shared cache are
  // ready before any handle is created — including the blocking HEAD that
  // rest_ioctx::create_io_object issues before start() is ever called.
  (void)global_curl_context::instance();

  // The wakeup fd is cheap and the worker (started in start()) registers it with
  // epoll, so create it up front. No worker thread or device staging allocation
  // is created until it is actually needed.
  _wakeup_fd = make_event_fd();
}

void rest_reactor::start()
{
  std::lock_guard lock(_enqueue_mutex);
  if (_running || _stopped) return;
  _running   = true;
  _accepting = true;
  try {
    _worker = std::jthread([this](std::stop_token const& st) { worker_loop(st); },
                           _stop_source.get_token());
  } catch (...) {
    _running   = false;
    _accepting = false;
    throw;
  }
  if (!_tname.empty()) {
    auto const full_name = _tname + "_worker";
    pthread_setname_np(_worker.native_handle(), full_name.c_str());
  }
}

rest_reactor::~rest_reactor() { shutdown(); }

void rest_reactor::interrupt()
{
  std::uint64_t one = 1;
  std::ignore       = ::write(_wakeup_fd.get(), &one, sizeof(one));
}

void rest_reactor::shutdown() noexcept
{
  {
    std::lock_guard lock(_enqueue_mutex);
    _accepting = false;
    _stopped   = true;
  }
  if (!_worker.joinable()) return;
  _stop_source.request_stop();
  interrupt();
  try {
    _worker.join();
  } catch (std::exception const& error) {
    SIRIUS_LOG_ERROR("rest_reactor: worker join failed: {}", error.what());
  } catch (...) {
    SIRIUS_LOG_ERROR("rest_reactor: worker join failed");
  }
}

void rest_reactor::enqueue(std::unique_ptr<grouped_io_request> req) noexcept
{
  if (req == nullptr) return;
  auto const bytes = req->remaining_bytes();
  auto const error = std::make_error_code(std::errc::operation_canceled);

  bool enqueued = false;
  {
    std::lock_guard lock(_enqueue_mutex);
    if (_accepting) {
      _queued_bytes.fetch_add(bytes, std::memory_order_relaxed);
      try {
        enqueued = _requests.enqueue(std::move(req));
      } catch (...) {
        enqueued = false;
      }
      if (!enqueued) { _queued_bytes.fetch_sub(bytes, std::memory_order_relaxed); }
    }
  }
  if (!enqueued) {
    if (req != nullptr) req->cancel_remaining(error);
    return;
  }
  interrupt();
}

std::size_t rest_reactor::host_read(io_object_type const& file,
                                    std::size_t offset,
                                    std::size_t size,
                                    std::uint8_t* dst)
{
  if (size == 0) return 0;
  size = std::min(size, file.size() > offset ? file.size() - offset : std::size_t{0});
  if (size == 0) return 0;
  if (dst == nullptr) throw std::invalid_argument("rest_reactor::host_read: null destination");

  if (auto const& stash = file.stash(); stash) {
    auto const lo = file.stash_window_lo();
    auto const hi = lo + stash->size();
    if (offset >= lo && offset + size <= hi) {
      std::memcpy(dst, stash->data() + (offset - lo), size);
      return size;
    }
  }

  std::shared_ptr<const io_object> owner;
  try {
    owner = file.shared_from_this();
  } catch (std::bad_weak_ptr const&) {
    owner = std::shared_ptr<const io_object>(&file, [](io_object const*) {});
  }

  auto coordinator = std::make_shared<grouped_coordinator>(size, 1);
  auto future      = coordinator->get_future();
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{offset, size}, host_buffer{dst});
  enqueue(grouped_io_request::create(std::move(owner), std::move(slices), coordinator));
  return std::move(future).get();
}

void rest_reactor::warmup(std::string bucket)
{
  {
    std::lock_guard lk{_warm_mtx};
    _warm_bucket = std::move(bucket);
  }
  _warm_requested.store(true, std::memory_order_release);
  interrupt();
}

head_object_result rest_reactor::head_object(std::string_view bucket, std::string_view key)
{
  object_ref const obj{std::string(bucket), std::string(key)};
  std::string last_error;
  for (std::size_t attempt = 0; attempt < _config.max_retry_attempts; ++attempt) {
    head_capture hc;
    auto const authd =
      _ctx->authorizer()->authorize(obj, request_method::HEAD, presign_ttl(_config));

    curl_easy_ptr h{curl_easy_init()};
    if (!h) { throw std::runtime_error("rest_reactor::head_object: curl_easy_init failed"); }
    configure_easy_handle(h.get(), global_curl_context::instance().share_handle());
    apply_request_opts(h.get(), _config);

    curl_slist_ptr hdrs = build_header_list(authd.headers, nullptr);
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_NOBODY, 1L));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &write_discard));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERFUNCTION, &head_header_cb));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERDATA, &hc));

    CURLcode const rc = curl_easy_perform(h.get());
    long status       = 0;
    curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

    if (rc == CURLE_OK && status == 200) {
      curl_off_t cl = -1;
      curl_easy_getinfo(h.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);
      if (cl < 0) {
        throw std::runtime_error("rest_reactor::head_object: missing Content-Length for " +
                                 obj.bucket + "/" + obj.key);
      }
      return head_object_result{static_cast<size_t>(cl), std::move(hc.etag)};
    }

    last_error =
      rc != CURLE_OK ? std::string(curl_easy_strerror(rc)) : ("HTTP " + std::to_string(status));
    bool const retriable =
      (rc != CURLE_OK && is_retriable_curl(rc)) || (rc == CURLE_OK && is_retriable_status(status));
    if (!retriable) {
      throw std::runtime_error("rest_reactor::head_object: " + last_error + " for " + obj.bucket +
                               "/" + obj.key);
    }
    if (attempt + 1 < _config.max_retry_attempts) {
      SIRIUS_LOG_WARN("rest_reactor::head_object: retrying {}/{} after {} (attempt {}/{})",
                      obj.bucket,
                      obj.key,
                      last_error,
                      attempt + 1,
                      _config.max_retry_attempts);
      std::this_thread::sleep_for(compute_backoff(attempt, hc.retry_after, _config));
    }
  }
  throw std::runtime_error("rest_reactor::head_object: exhausted retries (" + last_error +
                           ") for " + obj.bucket + "/" + obj.key);
}

size_t rest_reactor::head_object_size(std::string_view bucket, std::string_view key)
{
  return head_object(bucket, key).object_size;
}

std::string rest_reactor::list_page(std::string_view bucket,
                                    std::string_view prefix,
                                    std::string_view canonical_query)
{
  std::string const bucket_s{bucket};
  std::string const prefix_s{prefix};
  std::string last_error;
  for (std::size_t attempt = 0; attempt < _config.max_retry_attempts; ++attempt) {
    header_capture hc;
    auto const authd = _ctx->authorizer()->authorize_list(
      bucket_s, std::string{canonical_query}, presign_ttl(_config));

    curl_easy_ptr h{curl_easy_init()};
    if (!h) { throw std::runtime_error("rest_reactor::list_page: curl_easy_init failed"); }
    configure_easy_handle(h.get(), global_curl_context::instance().share_handle());
    apply_request_opts(h.get(), _config);

    std::string body;
    curl_slist_ptr hdrs = build_header_list(authd.headers, nullptr);
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPGET, 1L));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &write_string));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEDATA, &body));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERFUNCTION, &capture_header));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERDATA, &hc));

    CURLcode const rc = curl_easy_perform(h.get());
    long status       = 0;
    curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

    if (rc == CURLE_OK && status == 200) { return body; }

    last_error =
      rc != CURLE_OK ? std::string(curl_easy_strerror(rc)) : ("HTTP " + std::to_string(status));
    bool const retriable =
      (rc != CURLE_OK && is_retriable_curl(rc)) || (rc == CURLE_OK && is_retriable_status(status));
    if (!retriable) {
      throw std::runtime_error("rest_reactor::list_page: " + last_error + " for " + bucket_s + "/" +
                               prefix_s);
    }
    if (attempt + 1 < _config.max_retry_attempts) {
      SIRIUS_LOG_WARN("rest_reactor::list_page: retrying {}/{} after {} (attempt {}/{})",
                      bucket_s,
                      prefix_s,
                      last_error,
                      attempt + 1,
                      _config.max_retry_attempts);
      std::this_thread::sleep_for(compute_backoff(attempt, hc.retry_after, _config));
    }
  }
  throw std::runtime_error("rest_reactor::list_page: exhausted retries (" + last_error + ") for " +
                           bucket_s + "/" + prefix_s);
}

footer_probe rest_reactor::fetch_footer_suffix(std::string_view bucket,
                                               std::string_view key,
                                               std::size_t n)
{
  // Bind-time, blocking, and off the reactor's pooled connections: each call
  // opens a fresh TCP+TLS connection and every file's probe runs on one reactor.
  footer_probe probe;
  if (n == 0) { return probe; }

  object_ref const obj{std::string(bucket), std::string(key)};
  std::string last_error;
  for (std::size_t attempt = 0; attempt < _config.max_retry_attempts; ++attempt) {
    suffix_sink sink;
    sink.cap = n;
    auto const authd =
      _ctx->authorizer()->authorize(obj, request_method::GET, presign_ttl(_config));

    curl_easy_ptr h{curl_easy_init()};
    if (!h) {
      throw std::runtime_error("rest_reactor::fetch_footer_suffix: curl_easy_init failed");
    }
    configure_easy_handle(h.get(), global_curl_context::instance().share_handle());
    apply_request_opts(h.get(), _config);

    std::string const range = suffix_range_header(n);
    curl_slist_ptr hdrs     = build_header_list(authd.headers, &range);
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &suffix_write_cb));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEDATA, &sink));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERFUNCTION, &suffix_header_cb));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERDATA, &sink));

    CURLcode const rc = curl_easy_perform(h.get());
    long status       = 0;
    curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

    // suffix_write_cb aborts any non-206 body, so a CURLE_WRITE_ERROR here is our
    // own doing and the HTTP status is still valid; only a different curl error
    // (no HTTP status) is a genuine transport failure.
    if (rc != CURLE_OK && rc != CURLE_WRITE_ERROR) {
      last_error = std::string(curl_easy_strerror(rc));
      if (!is_retriable_curl(rc)) {
        throw std::runtime_error("rest_reactor::fetch_footer_suffix: " + last_error + " for " +
                                 obj.bucket + "/" + obj.key);
      }
    } else if (status == 206) {
      // Trust the 206 only when the window origin and total both parse and the
      // delivered byte count matches exactly; an unverifiable 206 (missing /
      // "*" Content-Range) reports an empty probe so the caller HEADs instead.
      auto const total = content_range_total(sink.content_range);
      auto const start = content_range_start(sink.content_range);
      if (total && start && *start <= *total && sink.data.size() == *total - *start) {
        probe.object_size = *total;
        probe.window_lo   = *start;
        probe.bytes       = make_shared_byte_span(std::move(sink.data));
        probe.etag        = std::move(sink.etag);
      }
      return probe;
    } else if (status == 200 || status == 416) {
      // Range ignored (full body) or unsatisfiable (416): probe unusable but the
      // object exists — report empty so the caller falls back to a HEAD.
      return probe;
    } else if (is_retriable_status(status)) {
      last_error = "HTTP " + std::to_string(status);
    } else {
      // 404 / 403 / 401 / ... — an error a HEAD would not recover from either.
      throw std::runtime_error("rest_reactor::fetch_footer_suffix: HTTP " + std::to_string(status) +
                               " for " + obj.bucket + "/" + obj.key);
    }
    if (attempt + 1 < _config.max_retry_attempts) {
      SIRIUS_LOG_WARN("rest_reactor::fetch_footer_suffix: retrying {}/{} after {} (attempt {}/{})",
                      obj.bucket,
                      obj.key,
                      last_error,
                      attempt + 1,
                      _config.max_retry_attempts);
      std::this_thread::sleep_for(compute_backoff(attempt, sink.retry_after, _config));
    }
  }
  throw std::runtime_error("rest_reactor::fetch_footer_suffix: exhausted retries (" + last_error +
                           ") for " + obj.bucket + "/" + obj.key);
}

// ---------------------------------------------------------------------------
// capabilities / factory
// ---------------------------------------------------------------------------

bool rest_reactor::supports(std::string_view path)
{
  try {
    auto const parsed = sirius::io::parse(path);
    return parsed.scheme == "s3";
  } catch (...) {
    return false;
  }
}

std::unique_ptr<rest_reactor::io_object_type> rest_reactor::create_io_object(std::string /*path*/)
{
  // The object size requires a HEAD round-trip and the authorizer, both of
  // which live on the reactor instance — see rest_ioctx::create_io_object.
  throw std::logic_error(
    "rest_reactor::create_io_object: use rest_ioctx::create_io_object (needs HEAD + authorizer)");
}

std::vector<cudf::io::text::byte_range_info> rest_reactor::align_and_coalesce(
  std::span<const cudf::io::text::byte_range_info> ranges, std::optional<size_t> alignment)
{
  // No physical block alignment for REST: honor a caller alignment >= 1 as a
  // lower bound, otherwise treat alignment as 1 (byte) — i.e. pure coalescing.
  size_t const align = std::max<size_t>(alignment.value_or(1), 1);

  std::vector<cudf::io::text::byte_range_info> aligned;
  aligned.reserve(ranges.size());
  for (auto const& r : ranges) {
    if (r.size() <= 0) { continue; }
    auto const offset  = static_cast<size_t>(r.offset());
    auto const end     = offset + static_cast<size_t>(r.size());
    size_t const start = (offset / align) * align;
    size_t const stop  = ((end + align - 1) / align) * align;
    aligned.emplace_back(static_cast<int64_t>(start), static_cast<int64_t>(stop - start));
  }
  if (aligned.empty()) { return aligned; }

  std::sort(aligned.begin(), aligned.end(), [](auto const& a, auto const& b) {
    return a.offset() < b.offset();
  });

  std::vector<cudf::io::text::byte_range_info> coalesced;
  coalesced.reserve(aligned.size());
  coalesced.push_back(aligned.front());
  for (size_t i = 1; i < aligned.size(); ++i) {
    auto& last            = coalesced.back();
    auto const last_start = static_cast<size_t>(last.offset());
    auto const last_end   = last_start + static_cast<size_t>(last.size());
    auto const cur_start  = static_cast<size_t>(aligned[i].offset());
    auto const cur_end    = cur_start + static_cast<size_t>(aligned[i].size());
    if (cur_start <= last_end) {  // overlap or adjacency
      size_t const new_end = std::max(last_end, cur_end);
      last                 = {last.offset(), static_cast<int64_t>(new_end - last_start)};
    } else {
      coalesced.push_back(aligned[i]);
    }
  }
  return coalesced;
}

// ---------------------------------------------------------------------------
// worker loop (epoll + curl_multi_socket_action engine)
// ---------------------------------------------------------------------------

namespace {

struct io_slot {
  curl_easy_ptr easy;
  slot_pool::token token;
  std::unique_ptr<rest_io_op_request> req;
  std::string url;
  curl_slist_ptr headers;
  buf_sink sink;
  header_capture hc;

  void reset() noexcept
  {
    req.reset();
    url.clear();
    headers.reset();
    sink = buf_sink{};
    hc.reset();
    token = {};
  }
};

struct worker_state {
  CURLM* multi{nullptr};
  int epoll_fd{-1};
  int curl_timer_fd{-1};
};

int rest_socket_cb(CURL* /*easy*/, curl_socket_t socket, int what, void* userp, void* socketp)
{
  auto* state = static_cast<worker_state*>(userp);
  if (what == CURL_POLL_REMOVE) {
    ::epoll_ctl(state->epoll_fd, EPOLL_CTL_DEL, socket, nullptr);
    return 0;
  }

  std::uint32_t events = 0;
  if (what == CURL_POLL_IN || what == CURL_POLL_INOUT) events |= EPOLLIN;
  if (what == CURL_POLL_OUT || what == CURL_POLL_INOUT) events |= EPOLLOUT;
  epoll_event event{};
  event.events  = events;
  event.data.fd = socket;
  auto const op = socketp == nullptr ? EPOLL_CTL_ADD : EPOLL_CTL_MOD;
  if (socketp == nullptr) curl_multi_assign(state->multi, socket, state);
  ::epoll_ctl(state->epoll_fd, op, socket, &event);
  return 0;
}

int rest_timer_cb(CURLM* /*multi*/, long timeout_ms, void* userp)
{
  auto* state = static_cast<worker_state*>(userp);
  itimerspec timer{};
  if (timeout_ms == 0) {
    timer.it_value.tv_nsec = 1;
  } else if (timeout_ms > 0) {
    timer.it_value.tv_sec  = timeout_ms / 1000;
    timer.it_value.tv_nsec = (timeout_ms % 1000) * 1'000'000L;
  }
  ::timerfd_settime(state->curl_timer_fd, 0, &timer, nullptr);
  return 0;
}

void drain_fd(int fd) noexcept
{
  std::uint64_t value = 0;
  while (::read(fd, &value, sizeof(value)) > 0) {}
}

}  // namespace

void rest_reactor::worker_loop(std::stop_token const& stop_token)
{
  std::stop_callback const stop_callback(stop_token, [this] { interrupt(); });
  auto const canceled = std::make_error_code(std::errc::operation_canceled);
  std::exception_ptr worker_error;

  try {
    curl_multi_ptr multi{curl_multi_init()};
    if (!multi) throw std::runtime_error("rest_reactor: curl_multi_init failed");

    file_descriptor epoll_fd        = make_epoll_fd();
    file_descriptor curl_timer_fd   = make_timer_fd();
    file_descriptor retry_timer_fd  = make_timer_fd();
    file_descriptor upkeep_timer_fd = make_timer_fd();
    worker_state state{multi.get(), epoll_fd.get(), curl_timer_fd.get()};

    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_SOCKETFUNCTION, &rest_socket_cb));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_SOCKETDATA, &state));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_TIMERFUNCTION, &rest_timer_cb));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_TIMERDATA, &state));
    SIRIUS_CURLM_CHECK(
      curl_multi_setopt(multi.get(), CURLMOPT_PIPELINING, static_cast<long>(CURLPIPE_NOTHING)));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(
      multi.get(), CURLMOPT_MAX_HOST_CONNECTIONS, static_cast<long>(_config.max_connections)));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(
      multi.get(), CURLMOPT_MAXCONNECTS, static_cast<long>(_config.max_connections)));

    auto epoll_add = [&](int fd, std::uint32_t events) {
      epoll_event event{};
      event.events  = events;
      event.data.fd = fd;
      if (::epoll_ctl(epoll_fd.get(), EPOLL_CTL_ADD, fd, &event) != 0) {
        throw std::runtime_error(std::string("rest_reactor: epoll_ctl ADD failed: ") +
                                 std::strerror(errno));
      }
    };
    epoll_add(_wakeup_fd.get(), EPOLLIN);
    epoll_add(curl_timer_fd.get(), EPOLLIN);
    epoll_add(retry_timer_fd.get(), EPOLLIN);
    epoll_add(upkeep_timer_fd.get(), EPOLLIN);

    auto const upkeep_ms = static_cast<long>(_config.upkeep_interval.count());
    if (upkeep_ms > 0) {
      itimerspec timer{};
      timer.it_value.tv_sec = timer.it_interval.tv_sec = upkeep_ms / 1000;
      timer.it_value.tv_nsec = timer.it_interval.tv_nsec = (upkeep_ms % 1000) * 1'000'000L;
      ::timerfd_settime(upkeep_timer_fd.get(), 0, &timer, nullptr);
    }

    curl_share worker_share{/*share_connections=*/true};
    std::vector<io_slot> slots(_config.max_connections);
    slot_pool pool{_config.max_connections};
    for (std::size_t i = 0; i < slots.size(); ++i) {
      curl_easy_ptr handle{curl_easy_init()};
      if (!handle) throw std::runtime_error("rest_reactor: curl_easy_init failed");
      configure_easy_handle(handle.get(),
                            worker_share.get(),
                            upkeep_ms,
                            static_cast<long>(_config.conn_max_age.count()));
      apply_request_opts(handle.get(), _config);
      SIRIUS_CURL_CHECK(curl_easy_setopt(
        handle.get(), CURLOPT_PRIVATE, reinterpret_cast<void*>(static_cast<std::intptr_t>(i))));
      slots[i].easy = std::move(handle);
    }

    std::vector<curl_easy_ptr> warm_handles;
    std::vector<curl_slist_ptr> warm_headers;
    warm_handles.reserve(_config.max_connections);
    warm_headers.reserve(_config.max_connections);
    auto prime_connections = [&](std::string const& bucket) {
      constexpr std::string_view warm_query = "list-type=2&max-keys=0";
      for (std::size_t i = 0; i < _config.max_connections; ++i) {
        try {
          auto auth = _ctx->authorizer()->authorize_list(bucket, warm_query, presign_ttl(_config));
          curl_easy_ptr handle{curl_easy_init()};
          if (!handle) break;
          configure_easy_handle(handle.get(),
                                worker_share.get(),
                                upkeep_ms,
                                static_cast<long>(_config.conn_max_age.count()));
          apply_request_opts(handle.get(), _config);
          auto headers = build_header_list(auth.headers, nullptr);
          SIRIUS_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_URL, auth.url.c_str()));
          SIRIUS_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_HTTPHEADER, headers.get()));
          SIRIUS_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_WRITEFUNCTION, &write_discard));
          SIRIUS_CURL_CHECK(curl_easy_setopt(
            handle.get(), CURLOPT_PRIVATE, reinterpret_cast<void*>(std::intptr_t{-1})));
          SIRIUS_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_FRESH_CONNECT, 1L));
          if (curl_multi_add_handle(multi.get(), handle.get()) != CURLM_OK) break;
          warm_headers.push_back(std::move(headers));
          warm_handles.push_back(std::move(handle));
        } catch (std::exception const& error) {
          SIRIUS_LOG_DEBUG("rest_reactor: warm-up handle {} not issued: {}", i, error.what());
          break;
        }
      }
    };
    auto maybe_prime = [&] {
      if (!_warm_requested.exchange(false, std::memory_order_acq_rel) || !warm_handles.empty()) {
        return;
      }
      std::string bucket;
      {
        std::lock_guard lock(_warm_mtx);
        bucket = _warm_bucket;
      }
      if (!bucket.empty()) prime_connections(bucket);
    };

    struct retry_entry {
      std::chrono::steady_clock::time_point due;
      std::unique_ptr<rest_io_op_request> req;
    };
    auto const retry_compare = [](retry_entry const& lhs, retry_entry const& rhs) {
      return lhs.due > rhs.due;
    };
    std::vector<retry_entry> retry_heap;
    retry_heap.reserve(_config.max_connections);
    std::deque<std::unique_ptr<rest_io_op_request>> ready;
    std::deque<std::unique_ptr<rest_io_op_request>> pending;
    std::unique_ptr<grouped_io_request> active_group;

    struct parked_copy {
      slot_pool::token token;
      cucascade::cuda::cuda_event* event{nullptr};
      std::unique_ptr<rest_io_op_request> req;
    };
    std::unordered_map<int, std::vector<cucascade::cuda::cuda_event>> copy_events;
    std::vector<parked_copy> copying;
    copying.reserve(_config.max_connections);

    auto event_for = [&](int device_id, std::size_t slot_index) {
      if (device_id < 0) {
        throw std::runtime_error("rest_reactor: device copy has no CUDA device id");
      }
      auto& events = copy_events[device_id];
      if (events.size() != _config.max_connections) {
        if (!events.empty()) {
          throw std::runtime_error("rest_reactor: incomplete CUDA event pool");
        }
        rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};
        std::vector<cucascade::cuda::cuda_event> initialized;
        initialized.reserve(_config.max_connections);
        std::generate_n(std::back_inserter(initialized), _config.max_connections, [] {
          return cucascade::cuda::cuda_event{cudaEventDisableTiming};
        });
        events = std::move(initialized);
      }
      return &events.at(slot_index);
    };

    auto poll_copy_completions = [&] {
      using query_result = cucascade::cuda::event::query_result;
      for (auto it = copying.begin(); it != copying.end();) {
        auto const result = it->event->query();
        if (result == query_result::in_progress) {
          ++it;
          continue;
        }
        if (result == query_result::success) {
          it->req->op->finish_success();
        } else {
          it->req->op->finish_error(
            std::make_exception_ptr(std::runtime_error("rest_reactor: device H2D copy failed")),
            true);
        }
        it = copying.erase(it);
      }
    };

    auto arm_retry_timer = [&] {
      itimerspec timer{};
      if (!retry_heap.empty()) {
        auto const now = std::chrono::steady_clock::now();
        auto nanos =
          retry_heap.front().due > now
            ? std::chrono::duration_cast<std::chrono::nanoseconds>(retry_heap.front().due - now)
                .count()
            : std::int64_t{1};
        nanos                  = std::max<std::int64_t>(nanos, 1);
        timer.it_value.tv_sec  = nanos / 1'000'000'000;
        timer.it_value.tv_nsec = nanos % 1'000'000'000;
      }
      ::timerfd_settime(retry_timer_fd.get(), 0, &timer, nullptr);
    };

    auto schedule_retry = [&](std::unique_ptr<rest_io_op_request> req,
                              std::string const& retry_after,
                              bool is_auth,
                              std::string const& reason) {
      try {
        auto& attempt    = is_auth ? req->auth_attempt : req->attempt;
        auto const limit = is_auth ? _config.max_auth_retry_attempts : _config.max_retry_attempts;
        if (attempt + 1 >= limit) {
          req->op->finish_error(std::make_exception_ptr(std::runtime_error(
            "rest_reactor: exhausted retries for " + req->object.bucket + "/" + req->object.key)));
          return;
        }
        auto const delay = compute_backoff(req->attempt, retry_after, _config);
        SIRIUS_LOG_WARN("rest_reactor: retrying {}/{} after {} (attempt {}/{})",
                        req->object.bucket,
                        req->object.key,
                        reason,
                        attempt + 1,
                        limit);
        ++attempt;
        retry_heap.push_back(retry_entry{std::chrono::steady_clock::now() + delay, std::move(req)});
        std::push_heap(retry_heap.begin(), retry_heap.end(), retry_compare);
        arm_retry_timer();
      } catch (...) {
        if (req != nullptr) {
          req->op->finish_error(std::current_exception());
          return;
        }
        throw;
      }
    };

    auto allocate_staging = [&](rest_io_op_request& request) {
      if (!request.needs_staging || request.op->staging_owner != nullptr) return;
      auto* resource = _ctx->host_memory_resource();
      if (resource == nullptr) {
        throw std::runtime_error("rest_reactor: staged device read requires host memory resource");
      }
      auto allocation = resource->allocate_multiple_blocks(request.op->io_rng.size);
      if (allocation == nullptr || allocation->size_bytes() < request.op->io_rng.size) {
        throw std::runtime_error("rest_reactor: failed to allocate complete staging range");
      }
      request.op->iovecs.clear();
      auto remaining = request.op->io_rng.size;
      for (auto* block : allocation->get_blocks()) {
        if (remaining == 0) break;
        auto const bytes = std::min(allocation->block_size(), remaining);
        request.op->iovecs.push_back(iovec{block, bytes});
        remaining -= bytes;
      }
      if (remaining != 0) {
        throw std::runtime_error("rest_reactor: staging blocks do not cover physical operation");
      }
      using allocation_type =
        cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;
      request.op->staging_owner = std::shared_ptr<allocation_type>(std::move(allocation));
    };

    auto expand_active = [&](std::size_t free_connections) {
      if (active_group == nullptr || active_group->empty()) return;
      auto slice            = active_group->take_front();
      auto const slice_size = slice.size();
      try {
        if (slice.is_staged() && !slice.has_device_request()) {
          throw std::invalid_argument("rest_reactor: a staged read must have a device destination");
        }
        auto const* file = dynamic_cast<rest_io_object const*>(active_group->obj.get());
        if (file == nullptr) {
          throw std::invalid_argument("rest_reactor: logical request belongs to another backend");
        }

        // Footer-probe opens already fetched this suffix.  The public io_context
        // now routes every host read through the grouped async path, so retain
        // the old stash fast path here in the worker rather than issuing a
        // second GET for bytes we own.  Only a contiguous host-only slice can
        // be completed this way; fragmented cache fills and device requests
        // still need their normal physical-operation lifecycle.
        if (slice.is_host_request() && slice.is_contiguous()) {
          auto const& stash = file->stash();
          auto const lo     = file->stash_window_lo();
          auto const hi     = stash == nullptr ? lo : lo + stash->size();
          if (stash != nullptr && slice.rng.offset >= lo && slice.rng.offset <= hi &&
              slice.rng.size <= hi - slice.rng.offset) {
            auto* dst = std::get<std::uint8_t*>(slice.h_buffer.buffer);
            std::memcpy(dst, stash->data() + (slice.rng.offset - lo), slice.rng.size);
            _queued_bytes.fetch_sub(slice_size, std::memory_order_relaxed);
            if (slice.on_complete != nullptr) { (*slice.on_complete)({}, true); }
            active_group->coordinator->on_complete();
            if (active_group->empty()) { active_group.reset(); }
            return;
          }
        }

        auto const block_size = _ctx->host_memory_resource() == nullptr
                                  ? std::size_t{0}
                                  : _ctx->host_memory_resource()->get_block_size();
        auto const target =
          dynamic_segment_target(_queued_bytes.load(std::memory_order_relaxed), free_connections);
        auto ranges = physical_ranges(slice, target, block_size);
        for (auto& io_rng : ranges) {
          if (io_rng.offset >= file->size()) {
            io_rng.size = 0;
          } else {
            io_rng.size = std::min(io_rng.size, file->size() - io_rng.offset);
          }
        }
        std::erase_if(ranges, [](range const& io_rng) { return io_rng.empty(); });
        if (ranges.empty()) throw std::runtime_error("rest_reactor: empty physical plan");

        std::vector<std::unique_ptr<rest_io_op_request>> expanded;
        expanded.reserve(ranges.size());
        std::size_t logical_bytes = 0;
        for (auto const io_rng : ranges) {
          auto op               = std::make_unique<io_op_request>();
          op->obj               = active_group->obj;
          op->io_rng            = io_rng;
          op->iovecs            = operation_iovecs(slice, io_rng, block_size);
          op->coordinator       = active_group->coordinator;
          op->on_complete       = slice.on_complete;
          op->completion_chunks = operation_chunks(slice, io_rng, block_size);
          if (slice.has_device_request()) {
            op->device_copy = std::make_unique<device_cpy_request>(
              device_cpy_request{slice.rng, slice.d_buffer, slice.d_buffer.device_id});
          }

          auto request           = std::make_unique<rest_io_op_request>();
          request->object        = file->get_object_ref();
          request->needs_staging = slice.is_staged();
          request->logical_bytes = intersect(slice.rng, io_rng).size;
          logical_bytes += request->logical_bytes;
          request->op = std::move(op);
          expanded.push_back(std::move(request));
        }
        if (logical_bytes != slice_size) {
          throw std::runtime_error("rest_reactor: physical plan does not cover logical slice");
        }

        auto const pending_before = pending.size();
        try {
          for (auto& request : expanded) {
            pending.push_back(std::move(request));
          }
        } catch (...) {
          while (pending.size() != pending_before) {
            pending.pop_back();
          }
          throw;
        }
        active_group->coordinator->add_tasks(expanded.size() - 1);
      } catch (...) {
        _queued_bytes.fetch_sub(slice_size, std::memory_order_relaxed);
        if (slice.on_complete != nullptr) {
          (*slice.on_complete)(slice.h_buffer.fragments(), false);
        }
        active_group->coordinator->report_error(std::current_exception());
      }
      if (active_group->empty()) active_group.reset();
    };

    auto next_fresh = [&](std::size_t free_connections) {
      for (;;) {
        while (!pending.empty() && !pending.front()->op->coordinator->should_continue()) {
          auto request = std::move(pending.front());
          pending.pop_front();
          _queued_bytes.fetch_sub(request->logical_bytes, std::memory_order_relaxed);
          request->op->finish_error(canceled);
        }
        if (!pending.empty()) {
          auto request = std::move(pending.front());
          pending.pop_front();
          _queued_bytes.fetch_sub(request->logical_bytes, std::memory_order_relaxed);
          return request;
        }

        if (active_group != nullptr && !active_group->coordinator->should_continue()) {
          auto const bytes = active_group->remaining_bytes();
          _queued_bytes.fetch_sub(bytes, std::memory_order_relaxed);
          active_group->cancel_remaining(canceled);
          active_group.reset();
        }
        if (active_group == nullptr) {
          if (!_requests.try_dequeue(active_group)) {
            return std::unique_ptr<rest_io_op_request>{};
          }
          if (active_group == nullptr) continue;
        }
        expand_active(free_connections);
      }
    };

    auto setup_easy = [&](io_slot& slot) {
      auto auth =
        _ctx->authorizer()->authorize(slot.req->object, request_method::GET, presign_ttl(_config));
      slot.url           = std::move(auth.url);
      slot.sink.buffers  = slot.req->op->iovecs;
      slot.sink.capacity = slot.req->op->io_rng.size;
      slot.sink.reset();
      slot.hc.reset();
      auto const range = range_header(slot.req->op->io_rng.offset, slot.req->op->io_rng.size);
      slot.headers     = build_header_list(auth.headers, &range);
      auto* handle     = slot.easy.get();
      SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HTTPGET, 1L));
      SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_URL, slot.url.c_str()));
      SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HTTPHEADER, slot.headers.get()));
      SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, &write_to_sink));
      SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_WRITEDATA, &slot.sink));
      SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HEADERFUNCTION, &capture_header));
      SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HEADERDATA, &slot.hc));
    };

    int running  = 0;
    int inflight = 0;
    auto submit  = [&] {
      for (;;) {
        auto token = pool.try_acquire_token();
        if (!token) break;

        std::unique_ptr<rest_io_op_request> request;
        if (!ready.empty()) {
          request = std::move(ready.front());
          ready.pop_front();
        } else {
          auto const occupied = static_cast<std::size_t>(inflight) + copying.size();
          auto const free_connections =
            occupied < _config.max_connections ? _config.max_connections - occupied : 1;
          request = next_fresh(free_connections);
        }
        if (request == nullptr) break;
        if (!request->op->coordinator->should_continue()) {
          request->op->finish_error(canceled);
          continue;
        }

        auto const index = static_cast<std::size_t>(token.slot_index());
        auto& slot       = slots[index];
        slot.token       = std::move(token);
        slot.req         = std::move(request);
        try {
          allocate_staging(*slot.req);
          setup_easy(slot);
          auto const status = curl_multi_add_handle(multi.get(), slot.easy.get());
          if (status != CURLM_OK) {
            throw std::runtime_error(std::string("rest_reactor: curl_multi_add_handle failed: ") +
                                     curl_multi_strerror(status));
          }
          ++inflight;
        } catch (...) {
          slot.req->op->finish_error(std::current_exception());
          slot.reset();
        }
      }
    };

    auto finish = [&](std::size_t index, CURLcode curl_status, long http_status) {
      auto& slot        = slots[index];
      auto& request     = *slot.req;
      auto& op          = *request.op;
      auto const io_rng = op.io_rng;
      bool const full_object =
        io_rng.offset == 0 && op.obj != nullptr && io_rng.size == op.obj->size();
      bool const range_status = http_status == 206 || (http_status == 200 && full_object);

      if (curl_status == CURLE_OK && http_status == 206) {
        auto const start = content_range_start(slot.hc.content_range);
        auto const total = content_range_total(slot.hc.content_range);
        if (!start || *start != io_rng.offset || !total ||
            (op.obj != nullptr && *total != op.obj->size())) {
          op.finish_error(std::make_exception_ptr(std::runtime_error(
            "rest_reactor: 206 Content-Range mismatch (got '" + slot.hc.content_range +
            "', requested offset " + std::to_string(io_rng.offset) + ") for " +
            request.object.bucket + "/" + request.object.key)));
          return false;
        }
      }

      bool const complete_body =
        slot.sink.written == io_rng.size && slot.sink.total_received == io_rng.size;
      if (curl_status == CURLE_OK && range_status && complete_body) {
        if (!request.is_device()) {
          op.finish_success();
          return false;
        }

        try {
          auto* event            = event_for(op.device_copy->device_id, index);
          auto const copy_status = request.copy_h2d_async(event->get());
          if (copy_status != cudaSuccess) {
            op.finish_error(copy_status, true);
            return false;
          }
          copying.push_back(parked_copy{std::move(slot.token), event, std::move(slot.req)});
          slot.reset();
          return true;
        } catch (...) {
          if (slot.req != nullptr) { slot.req->op->finish_error(std::current_exception(), true); }
          return false;
        }
      }

      if (curl_status == CURLE_OK && http_status == 200 && !full_object) {
        op.finish_error(std::make_exception_ptr(
          std::runtime_error("rest_reactor: server ignored Range for " + request.object.bucket +
                             "/" + request.object.key)));
        return false;
      }

      bool const short_read = curl_status == CURLE_OK && range_status && !complete_body;
      bool const retriable  = short_read ||
                             (curl_status != CURLE_OK && is_retriable_curl(curl_status)) ||
                             (curl_status == CURLE_OK && is_retriable_status(http_status));
      bool const auth_retriable = curl_status == CURLE_OK && http_status == 403;
      if (retriable || auth_retriable) {
        auto const reason =
          curl_status != CURLE_OK
            ? std::string(curl_easy_strerror(curl_status))
            : (short_read ? std::string("short read") : "HTTP " + std::to_string(http_status));
        schedule_retry(
          std::move(slot.req), slot.hc.retry_after, auth_retriable && !retriable, reason);
        return false;
      }

      auto const message =
        curl_status != CURLE_OK
          ? std::string(curl_easy_strerror(curl_status))
          : (range_status ? std::string("short read") : "HTTP " + std::to_string(http_status));
      op.finish_error(std::make_exception_ptr(std::runtime_error(
        "rest_reactor: " + message + " for " + request.object.bucket + "/" + request.object.key)));
      return false;
    };

    auto process_completions = [&] {
      int queued = 0;
      while (auto* message = curl_multi_info_read(multi.get(), &queued)) {
        if (message->msg != CURLMSG_DONE) continue;
        auto* handle       = message->easy_handle;
        char* private_data = nullptr;
        curl_easy_getinfo(handle, CURLINFO_PRIVATE, &private_data);
        if (reinterpret_cast<std::intptr_t>(private_data) < 0) {
          curl_multi_remove_handle(multi.get(), handle);
          std::erase_if(warm_handles, [handle](curl_easy_ptr const& candidate) {
            return candidate.get() == handle;
          });
          if (warm_handles.empty()) warm_headers.clear();
          continue;
        }

        long http_status = 0;
        curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &http_status);
        auto const index = static_cast<std::size_t>(reinterpret_cast<std::intptr_t>(private_data));
        curl_multi_remove_handle(multi.get(), handle);
        --inflight;
        if (!finish(index, message->data.result, http_status)) slots[index].reset();
      }
    };

    bool local_drained = false;
    auto drain_local   = [&](std::exception_ptr const& failure, bool complete_copies) noexcept {
      if (local_drained) return;
      local_drained = true;
      grouped_coordinator::error_type const terminal_error =
        failure != nullptr ? grouped_coordinator::error_type{failure}
                             : grouped_coordinator::error_type{canceled};

      for (auto& copy : copying) {
        if (copy.req == nullptr) continue;
        try {
          copy.event->synchronize();
          if (complete_copies && failure == nullptr) {
            copy.req->op->finish_success();
          } else {
            copy.req->op->finish_error(terminal_error, true);
          }
        } catch (...) {
          copy.req->op->finish_error(std::current_exception(), true);
        }
      }
      copying.clear();

      for (auto& handle : warm_handles) {
        curl_multi_remove_handle(multi.get(), handle.get());
      }
      warm_handles.clear();
      warm_headers.clear();

      for (auto& slot : slots) {
        if (slot.req == nullptr) continue;
        curl_multi_remove_handle(multi.get(), slot.easy.get());
        slot.req->op->finish_error(terminal_error);
        slot.reset();
      }
      for (auto& retry : retry_heap) {
        if (retry.req != nullptr) retry.req->op->finish_error(terminal_error);
      }
      retry_heap.clear();
      for (auto& request : ready) {
        if (request != nullptr) request->op->finish_error(terminal_error);
      }
      ready.clear();
      for (auto& request : pending) {
        if (request == nullptr) continue;
        _queued_bytes.fetch_sub(request->logical_bytes, std::memory_order_relaxed);
        request->op->finish_error(terminal_error);
      }
      pending.clear();
      if (active_group != nullptr) {
        _queued_bytes.fetch_sub(active_group->remaining_bytes(), std::memory_order_relaxed);
        active_group->cancel_remaining(terminal_error);
        active_group.reset();
      }
    };

    try {
      std::vector<epoll_event> events(_config.max_connections);
      maybe_prime();
      submit();
      while (!stop_token.stop_requested()) {
        auto const timeout_ms = copying.empty() ? -1 : 1;
        auto const count =
          ::epoll_wait(epoll_fd.get(), events.data(), static_cast<int>(events.size()), timeout_ms);
        if (count < 0) {
          if (errno == EINTR) continue;
          SIRIUS_LOG_ERROR("rest_reactor: epoll_wait failed: {}", std::strerror(errno));
          break;
        }
        for (int i = 0; i < count; ++i) {
          auto const fd = events[static_cast<std::size_t>(i)].data.fd;
          if (fd == _wakeup_fd.get()) {
            drain_fd(_wakeup_fd.get());
          } else if (fd == curl_timer_fd.get()) {
            drain_fd(curl_timer_fd.get());
            curl_multi_socket_action(multi.get(), CURL_SOCKET_TIMEOUT, 0, &running);
          } else if (fd == retry_timer_fd.get()) {
            drain_fd(retry_timer_fd.get());
            auto const now = std::chrono::steady_clock::now();
            while (!retry_heap.empty() && retry_heap.front().due <= now) {
              std::pop_heap(retry_heap.begin(), retry_heap.end(), retry_compare);
              ready.push_back(std::move(retry_heap.back().req));
              retry_heap.pop_back();
            }
            arm_retry_timer();
          } else if (fd == upkeep_timer_fd.get()) {
            drain_fd(upkeep_timer_fd.get());
            if (inflight == 0 && !slots.empty()) curl_easy_upkeep(slots.front().easy.get());
          } else {
            int action      = 0;
            auto const mask = events[static_cast<std::size_t>(i)].events;
            if (mask & EPOLLIN) action |= CURL_CSELECT_IN;
            if (mask & EPOLLOUT) action |= CURL_CSELECT_OUT;
            if (mask & (EPOLLERR | EPOLLHUP)) action |= CURL_CSELECT_ERR;
            curl_multi_socket_action(multi.get(), fd, action, &running);
          }
        }
        process_completions();
        poll_copy_completions();
        maybe_prime();
        submit();
      }
      drain_local(nullptr, true);
    } catch (...) {
      auto failure = std::current_exception();
      drain_local(failure, false);
      throw;
    }
  } catch (std::exception const& error) {
    worker_error = std::current_exception();
    SIRIUS_LOG_ERROR("rest_reactor worker_loop: {}", error.what());
  } catch (...) {
    worker_error = std::current_exception();
    SIRIUS_LOG_ERROR("rest_reactor worker_loop: unknown error");
  }

  {
    std::lock_guard lock(_enqueue_mutex);
    _accepting = false;
    _running   = false;
    _stopped   = true;
  }
  grouped_coordinator::error_type const terminal_error =
    worker_error != nullptr ? grouped_coordinator::error_type{worker_error}
                            : grouped_coordinator::error_type{canceled};
  std::unique_ptr<grouped_io_request> group;
  while (_requests.try_dequeue(group)) {
    if (group != nullptr) {
      _queued_bytes.fetch_sub(group->remaining_bytes(), std::memory_order_relaxed);
      group->cancel_remaining(terminal_error);
    }
    group.reset();
  }
}

}  // namespace sirius::io::rest
