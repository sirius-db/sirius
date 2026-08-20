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

#include <ctrack.hpp>
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

// Lock-free max: raise @p a to @p v when v is larger.  Backs the *_ns_max perf
// counters (relaxed — perf metrics tolerate reordering).
void atomic_max_relaxed(std::atomic<std::uint64_t>& a, std::uint64_t v) noexcept
{
  std::uint64_t cur = a.load(std::memory_order_relaxed);
  while (v > cur && !a.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {}
}

// ---- libcurl callbacks -----------------------------------------------------

/// Write callback: copy curl's bytes into the sink's destination buffer at the
/// running cursor, and ALWAYS report the full incoming size so curl never
/// aborts the transfer with CURLE_WRITE_ERROR.  Overflow past capacity is
/// counted (total_received) but not stored, so a server that ignored the Range
/// header can be detected after the fact.
size_t write_to_sink(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  // Deliberately not ctracked: curl invokes this once per socket read, so at
  // scan fan-out it runs tens of millions of times per query and any probe here
  // costs more than it measures.  The enclosing curl_multi_socket_action is
  // ctracked instead.
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

/// Sentinel for a @c planned_buffer that belongs to no caller entry (a hole).
constexpr size_t kNoTag = std::numeric_limits<size_t>::max();

/// One destination buffer of a planned read: the absolute file offset its first
/// byte carries, how many bytes it takes, where they land, and which caller-side
/// entry it came from.
///
/// @c host is null for two unrelated reasons, told apart by @c hole.  A hole is
/// a bridged gap — bytes the reactor fetches only to keep one GET contiguous and
/// then drops on arrival — and only ever appears between two real buffers of a
/// multi-buffer chunk.  A non-hole null is a bounce-staged read the reactor
/// backs with one of its pinned slots at submit time, which is why such a buffer
/// is never fused with anything: the slot is per chunk.
struct planned_buffer {
  size_t file_off{0};
  size_t len{0};
  uint8_t* host{nullptr};
  size_t tag{kNoTag};
  bool hole{false};
};

/// Cut one coalesced span into ranged-GET chunks and append them to @p out.
///
/// A span of S bytes becomes n = max(1, S / @p max_chunk_size) chunks — integer
/// division, so @p max_chunk_size is the floor a chunk may not go below rather
/// than a ceiling — with S balanced evenly across them (the first S % n chunks
/// take one extra byte).  31 MiB against a 16 MiB floor is one 31 MiB GET;
/// 33 MiB is 17 + 16.  A buffer straddling a cut is divided between the two
/// chunks, each piece keeping the tag of the buffer it came from.
void cut_span(std::span<const planned_buffer> span,
              size_t max_chunk_size,
              std::vector<std::vector<planned_buffer>>& out)
{
  size_t total = 0;
  for (auto const& b : span) {
    total += b.len;
  }
  if (total == 0) { return; }

  size_t const n    = std::max<size_t>(total / std::max<size_t>(max_chunk_size, 1), 1);
  size_t const base = total / n;
  size_t const rem  = total % n;

  size_t bi       = 0;  // buffer being consumed
  size_t consumed = 0;  // bytes already taken from span[bi]
  for (size_t c = 0; c < n; ++c) {
    size_t want = base + (c < rem ? 1 : 0);
    std::vector<planned_buffer> chunk;
    while (want > 0 && bi < span.size()) {
      auto const& b     = span[bi];
      size_t const take = std::min(want, b.len - consumed);
      chunk.push_back(planned_buffer{b.file_off + consumed,
                                     take,
                                     b.host != nullptr ? b.host + consumed : nullptr,
                                     b.tag,
                                     b.hole});
      consumed += take;
      want -= take;
      if (consumed == b.len) {
        ++bi;
        consumed = 0;
      }
    }
    out.push_back(std::move(chunk));
  }
}

/// Plan the ranged GETs for a set of destination buffers: coalesce, then cut.
///
/// Coalesce — a run of buffers is fused into one contiguous span while the gap
/// to the next is at most @p max_gap, each bridged gap becoming a hole buffer.
/// Fusing costs the gap's bytes and saves a round trip, which is the trade an
/// object store rewards.  A bounce-staged buffer (null @c host, not a hole)
/// never fuses in either direction — submit() backs it with a single pinned slot
/// and its H2D copy resolves against that slot.
///
/// Cut — see @c cut_span: each span is divided into chunks of at least
/// @p max_chunk_size bytes, balanced.
///
/// Buffers are expected in file order and disjoint.  Neither is required for
/// correctness — a buffer that starts before the running span's end (an overlap,
/// or simply an out-of-order input) is left unfused and fetched by a GET of its
/// own, so its bytes still land — but an overlap between two ascending buffers
/// means the same bytes are paid for twice, which is a caller bug worth catching
/// in debug.
std::vector<std::vector<planned_buffer>> plan_chunks(std::span<const planned_buffer> bufs,
                                                     size_t max_gap,
                                                     size_t max_chunk_size)
{
#ifndef NDEBUG
  for (size_t k = 1; k < bufs.size(); ++k) {
    // Only ascending pairs are checked: a descending pair is an unsorted input,
    // which costs fusion but nothing else.
    assert((bufs[k].file_off < bufs[k - 1].file_off ||
            bufs[k].file_off >= bufs[k - 1].file_off + bufs[k - 1].len) &&
           "plan_chunks: destination buffers must not overlap");
  }
#endif
  std::vector<std::vector<planned_buffer>> out;
  out.reserve(bufs.size());
  std::vector<planned_buffer> span;
  for (size_t i = 0; i < bufs.size();) {
    if (bufs[i].len == 0) {
      ++i;
      continue;
    }
    span.clear();
    span.push_back(bufs[i]);
    size_t span_end = bufs[i].file_off + bufs[i].len;
    size_t j        = i + 1;
    if (bufs[i].host != nullptr) {
      while (j < bufs.size() && bufs[j].len > 0 && bufs[j].host != nullptr) {
        if (bufs[j].file_off < span_end || bufs[j].file_off - span_end > max_gap) { break; }
        if (size_t const gap = bufs[j].file_off - span_end; gap > 0) {
          span.push_back(planned_buffer{span_end, gap, nullptr, kNoTag, /*hole=*/true});
        }
        span.push_back(bufs[j]);
        span_end = bufs[j].file_off + bufs[j].len;
        ++j;
      }
    }
    cut_span(std::span<const planned_buffer>(span.data(), span.size()), max_chunk_size, out);
    i = j;
  }
  return out;
}

/// Fold a planned chunk into the segment the reactor submits: one contiguous
/// file range whose response body is scattered across the chunk's buffers in
/// file order (a hole's null buffer tells @c write_to_sink to drop those bytes).
io_object_segment to_segment(std::span<const planned_buffer> chunk)
{
  io_object_segment seg{chunk.front().file_off, chunk.front().len, chunk.front().host};
  for (auto const& b : chunk.subspan(1)) {
    seg.append(iovec{static_cast<void*>(b.host), b.len});
  }
  return seg;
}

/// One flattened entry of a vectored host-to-device plan: the read span
/// [offset, offset + size), where it lands (a null @c host_buffer means the
/// reactor stages it through one of its pinned bounce slots), and the absolute
/// file window [copy_lo, copy_hi) of that span which is H2D-copied to
/// @c device_dst (which addresses copy_lo).  Entries are referenced by index
/// from the @c planned_buffer tags the planner carries through coalescing and
/// cutting, so a chunk's buffers resolve back to their entry positionally-free.
struct planned_device_segment {
  size_t offset;
  size_t size;
  uint8_t* host_buffer;
  uint8_t* device_dst;
  size_t copy_lo;
  size_t copy_hi;
};

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

  if (_ctx->host_memory_resource() != nullptr) {
    _bounce_slot_size = _ctx->host_memory_resource()->get_block_size();
  }

  // The wakeup fd is cheap and the worker (started in start()) registers it with
  // epoll, so create it up front.  No pinned bounce allocation and no worker
  // thread until start() — keeps a parked (unused) reactor cheap.
  _wakeup_fd = make_event_fd();
}

void rest_reactor::start()
{
  if (_worker.joinable()) { return; }  // already started

  if (_ctx->host_memory_resource() != nullptr) {
    // One pinned bounce buffer per slot (1:1 with the easy-handle pool), since a
    // slot stages at most one device read at a time.
    _bounce_storage = _ctx->host_memory_resource()->allocate_multiple_blocks(
      _config.max_connections * _bounce_slot_size);
  }

  _worker =
    std::jthread([this](const std::stop_token& st) { worker_loop(st); }, _stop_source.get_token());
  if (!_tname.empty()) {
    std::string const full_name = _tname + "_worker";
    pthread_setname_np(_worker.native_handle(), full_name.c_str());
  }
}

rest_reactor::~rest_reactor() { shutdown(); }

void rest_reactor::interrupt()
{
  // Break the worker out of epoll_wait.
  uint64_t one = 1;
  ssize_t rc   = ::write(_wakeup_fd.get(), &one, sizeof(one));
  (void)rc;  // EAGAIN on a saturated counter is fine — a wakeup is already pending.
}

void rest_reactor::shutdown()
{
  if (_worker.joinable()) {
    _stop_source.request_stop();
    interrupt();
    _worker.join();
  }
}

void rest_reactor::enqueue(request_type_ptr req)
{
  auto chunks = req->get_all_chunks();
  enqueue_chunks(chunks);
}

void rest_reactor::enqueue_chunks(std::span<std::unique_ptr<rest_chunked_rx_request>> batch)
{
  if (batch.empty()) { return; }
  std::size_t bytes = 0;
  if (_config.perf_instrumentation) {
    auto const now = std::chrono::steady_clock::now();
    for (auto& c : batch) {
      if (c) { c->t_enqueue = now; }
    }
  }
  for (auto const& c : batch) {
    if (c) { bytes += c->chunk.size; }
  }
  // Publish the depth BEFORE the chunks become visible to the worker: the worker
  // subtracts on dequeue, so incrementing after would let a fast worker drive the
  // counter negative.  It is only ever read as a dispatch hint, so relaxed
  // ordering is enough.
  _queued_bytes.fetch_add(bytes, std::memory_order_relaxed);
  bool const ok = _requests.enqueue_bulk(std::make_move_iterator(batch.data()), batch.size());
  if (!ok) {
    _queued_bytes.fetch_sub(bytes, std::memory_order_relaxed);
    throw std::runtime_error("rest_reactor::enqueue_chunks: enqueue_bulk failed");
  }
  interrupt();
}

// ---------------------------------------------------------------------------
// request preparation (host paths)
// ---------------------------------------------------------------------------

rest_reactor::request_type_ptr rest_reactor::prep_host_rx_request(const reactor_config_type& cfg,
                                                                  const io_object_type& file,
                                                                  const io_object_segment& segment)
{
  return prep_host_rx_request(cfg, file, segment, host_read_attribution::async_chunk);
}

rest_reactor::request_type_ptr rest_reactor::prep_host_rx_request(const reactor_config_type& cfg,
                                                                  const io_object_type& file,
                                                                  const io_object_segment& segment,
                                                                  host_read_attribution attribution)
{
  if (segment.size == 0) { return rest_rx_request::create({}); }

  // A host read must carry the caller's destination buffer.  A null buffer means
  // "reactor-staged" (internal bounce slot), which only makes sense for device
  // reads: a host read staged through the bounce would report success while the
  // bytes sit unreachable in a reactor-private buffer.
  assert(segment.is_buffer_allocated() &&
         "rest_reactor::prep_host_rx_request: host read requires a non-null destination buffer");

  // The same floor the planned paths use (see cut_span), applied directly since
  // one contiguous segment needs no coalescing: as many requests as
  // max_chunk_size fits into the read, then the bytes balanced evenly across
  // them.  Every request therefore carries at least max_chunk_size bytes, and a
  // read smaller than that stays a single GET.
  size_t const n_chunks =
    std::max<size_t>(segment.size / std::max<size_t>(cfg.max_chunk_size, 1), 1);

  auto manager       = std::make_shared<request_manager>(segment.size, n_chunks);
  auto const obj     = file.get_object_ref();
  size_t const fsize = file.size();
  uint8_t* const dst = segment.data();

  size_t const base = segment.size / n_chunks;
  size_t const rem  = segment.size % n_chunks;

  std::vector<std::unique_ptr<rest_chunked_rx_request>> chunks;
  chunks.reserve(n_chunks);
  size_t pos = 0;  // byte offset within the segment
  for (size_t c = 0; c < n_chunks; ++c) {
    size_t const piece          = base + (c < rem ? 1 : 0);
    auto req                    = std::make_unique<rest_chunked_rx_request>();
    req->object                 = obj;
    req->chunk                  = io_object_segment{segment.offset + pos, piece, dst + pos};
    req->file_size              = fsize;
    req->manager                = manager;
    req->perf_blocking_host_get = (attribution == host_read_attribution::blocking);
    chunks.push_back(std::move(req));
    pos += piece;
  }
  return rest_rx_request::create(std::move(chunks));
}

rest_reactor::request_type_ptr rest_reactor::prep_host_rxv_request(
  const reactor_config_type& cfg, const io_object_type& file, std::span<io_object_segment> segments)
{
  if (segments.empty()) { return rest_rx_request::create({}); }

  size_t const fsize = file.size();

  // Clamp each segment to the file end (dropping empties), preserving file
  // order, and total the requested bytes (what the future reports — the bridged
  // gap bytes below are read but never delivered, so they are not counted).
  std::vector<planned_buffer> bufs;
  bufs.reserve(segments.size());
  size_t bytes_requested = 0;
  for (auto const& s : segments) {
    // See prep_host_rx_request: host reads must carry caller buffers; a
    // null-buffer segment here would be silently staged through an internal
    // bounce slot and its bytes lost to the caller.
    assert(s.is_buffer_allocated() &&
           "rest_reactor::prep_host_rxv_request: host read requires non-null "
           "destination buffers");
    size_t const c = s.offset < fsize ? std::min(s.size, fsize - s.offset) : 0;
    if (c == 0) { continue; }
    bufs.push_back(planned_buffer{s.offset, c, s.data(), bufs.size()});
    bytes_requested += c;
  }
  if (bufs.empty()) { return rest_rx_request::create({}); }

  // Coalesce near neighbors into scatter GETs and cut the result at the chunk
  // floor; neither step changes the byte total delivered to the caller.
  auto groups = plan_chunks(std::span<const planned_buffer>(bufs.data(), bufs.size()),
                            cfg.merge_max_gap,
                            cfg.max_chunk_size);

  auto manager   = std::make_shared<request_manager>(bytes_requested, groups.size());
  auto const obj = file.get_object_ref();

  std::vector<std::unique_ptr<rest_chunked_rx_request>> chunks;
  chunks.reserve(groups.size());
  for (auto const& g : groups) {
    auto req       = std::make_unique<rest_chunked_rx_request>();
    req->object    = obj;
    req->chunk     = to_segment(std::span<const planned_buffer>(g.data(), g.size()));
    req->file_size = fsize;
    req->manager   = manager;
    chunks.push_back(std::move(req));
  }
  return rest_rx_request::create(std::move(chunks));
}

rest_reactor::request_type_ptr rest_reactor::prep_device_rx_request(const reactor_config_type& cfg,
                                                                    const io_object_type& file,
                                                                    uint8_t* dst,
                                                                    size_t offset,
                                                                    size_t size,
                                                                    rmm::cuda_stream_view stream,
                                                                    int device_id)
{
  if (size == 0) { return rest_rx_request::create({}); }
  if (cfg.bounce_block_size == 0) {
    throw std::runtime_error(
      "rest_reactor::prep_device_rx_request: device reads require a host_memory_resource on the "
      "reactor_context for bounce staging");
  }

  // REST has no GPU-direct path, so the read is staged through reactor-owned
  // pinned bounce slots and H2D-copied to dst.  No block alignment: split the
  // requested range (clamped to the file end) into bounce-sized windows, one
  // staged GET + one H2D copy each.
  size_t const fsize = file.size();
  size_t const end   = std::min(offset + size, fsize);
  if (offset >= end) { return rest_rx_request::create({}); }
  size_t const wanted = end - offset;
  size_t const bounce = cfg.bounce_block_size;
  size_t const n_win  = (wanted + bounce - 1) / bounce;

  auto manager   = std::make_shared<request_manager>(wanted, n_win);
  auto const obj = file.get_object_ref();

  std::vector<std::unique_ptr<rest_chunked_rx_request>> chunks;
  chunks.reserve(n_win);
  for (size_t w = offset; w < end; w += bounce) {
    size_t const rs = std::min(bounce, end - w);
    auto req        = std::make_unique<rest_chunked_rx_request>();
    req->object     = obj;
    req->chunk      = io_object_segment{w, rs};  // null buffer => reactor stages
    req->file_size  = fsize;
    auto cpy        = std::make_unique<device_cpy_request>();
    cpy->stream     = stream;
    cpy->device_id  = device_id;
    cpy->copies.push_back(device_cpy_request::copy{/*dst=*/dst + (w - offset),
                                                   /*src=*/nullptr,  // resolved to the bounce slot
                                                   /*src_off=*/0,
                                                   /*size=*/rs});
    req->cpy_req = std::move(cpy);
    req->manager = manager;
    chunks.push_back(std::move(req));
  }
  return rest_rx_request::create(std::move(chunks));
}

rest_reactor::request_type_ptr rest_reactor::prep_host_to_device_rx_request(
  const reactor_config_type& cfg,
  const io_object_type& file,
  std::span<io_object_segment> segments,
  uint8_t* dst,
  size_t offset,
  size_t size,
  rmm::cuda_stream_view stream,
  int device_id)
{
  // Device read staged through caller-supplied pinned host buffers.  Near
  // neighbors are coalesced into one contiguous scatter GET (whose response
  // lands across their buffers, like uring's readv); each buffer then H2D-copies
  // only the part overlapping the device window [offset, req_end) into dst, as a
  // batch of copies issued on one stream.
  if (size == 0 || segments.empty()) { return rest_rx_request::create({}); }

  size_t const fsize   = file.size();
  size_t const req_end = offset + size;
  auto const obj       = file.get_object_ref();

  // Validate overlap, total the device-buffer bytes each segment fills (the
  // value reported to the caller — not the host read size, which over-reads to
  // the file end), and clamp each segment's read to the file end (file order
  // preserved for the coalescer).
  size_t bytes_covered = 0;
  std::vector<planned_buffer> bufs;
  bufs.reserve(segments.size());
  for (auto const& s : segments) {
    size_t const lo = std::max(offset, s.offset);
    size_t const hi = std::min({req_end, s.offset + s.size, fsize});
    if (lo >= hi) {
      throw std::runtime_error(
        "rest_reactor::prep_host_to_device_rx_request: segment does not overlap the requested "
        "device range");
    }
    bytes_covered += hi - lo;
    // hi > lo implies s.offset < fsize, so the clamp is well-defined.
    bufs.push_back(
      planned_buffer{s.offset, std::min(s.size, fsize - s.offset), s.data(), bufs.size()});
  }

  auto groups = plan_chunks(std::span<const planned_buffer>(bufs.data(), bufs.size()),
                            cfg.merge_max_gap,
                            cfg.max_chunk_size);

  auto manager = std::make_shared<request_manager>(bytes_covered, groups.size());

  std::vector<std::unique_ptr<rest_chunked_rx_request>> chunks;
  chunks.reserve(groups.size());
  for (auto const& g : groups) {
    // One copy per buffer in the chunk, each clipped to the device window and
    // carrying an absolute src (the buffers are separate host allocations).
    // Holes are bridged gap bytes with nowhere to go, so they copy nothing.
    auto cpy       = std::make_unique<device_cpy_request>();
    cpy->stream    = stream;
    cpy->device_id = device_id;
    cpy->copies.reserve(g.size());
    for (auto const& b : g) {
      if (b.hole) { continue; }
      size_t const data_lo = std::max(offset, b.file_off);
      size_t const data_hi = std::min(req_end, b.file_off + b.len);
      if (data_lo < data_hi) {
        // A non-hole null buffer is a bounce-staged sub-range: submit() backs the
        // chunk with a pinned bounce slot (set_data) and the H2D copy must read
        // from that slot, so leave src null and carry the intra-buffer offset in
        // src_off — copy_async then resolves src = bounce_buffer + src_off.
        // Encoding a null buffer as `nullptr + off` (a non-null near-null
        // pointer) instead would bypass that bounce fallback and fault the H2D.
        // plan_chunks never fuses such a buffer, so it is alone in its chunk and
        // the bounce holds its whole span from offset 0.
        bool const bounce_staged = (b.host == nullptr);
        cpy->copies.push_back(device_cpy_request::copy{
          /*dst=*/dst + (data_lo - offset),
          /*src=*/bounce_staged ? nullptr : b.host + (data_lo - b.file_off),
          /*src_off=*/bounce_staged ? (data_lo - b.file_off) : size_t{0},
          /*size=*/data_hi - data_lo});
      }
    }

    auto req       = std::make_unique<rest_chunked_rx_request>();
    req->object    = obj;
    req->chunk     = to_segment(std::span<const planned_buffer>(g.data(), g.size()));
    req->file_size = fsize;
    req->cpy_req   = std::move(cpy);
    req->manager   = manager;
    chunks.push_back(std::move(req));
  }
  return rest_rx_request::create(std::move(chunks));
}

rest_reactor::request_type_ptr rest_reactor::prep_device_rxv_request(
  const reactor_config_type& cfg,
  const io_object_type& file,
  std::span<const io_device_range> ranges,
  rmm::cuda_stream_view stream,
  int device_id)
{
  if (ranges.empty()) { return rest_rx_request::create({}); }
  if (cfg.bounce_block_size == 0) {
    throw std::runtime_error(
      "rest_reactor::prep_device_rxv_request: device reads require a host_memory_resource on "
      "the reactor_context for bounce staging");
  }

  size_t const fsize  = file.size();
  size_t const bounce = cfg.bounce_block_size;
  auto const obj      = file.get_object_ref();

  size_t bytes_requested = 0;
  size_t n_win           = 0;
  for (auto const& r : ranges) {
    if (r.device_dst == nullptr || r.offset >= fsize) { continue; }
    size_t const wanted = std::min(r.size, fsize - r.offset);
    if (wanted == 0) { continue; }
    bytes_requested += wanted;
    n_win += (wanted + bounce - 1) / bounce;
  }
  if (n_win == 0) { return rest_rx_request::create({}); }

  auto manager = std::make_shared<request_manager>(bytes_requested, n_win);

  std::vector<std::unique_ptr<rest_chunked_rx_request>> chunks;
  chunks.reserve(n_win);
  for (auto const& r : ranges) {
    if (r.device_dst == nullptr || r.offset >= fsize) { continue; }
    size_t const end = r.offset + std::min(r.size, fsize - r.offset);
    for (size_t w = r.offset; w < end; w += bounce) {
      size_t const rs = std::min(bounce, end - w);
      auto req        = std::make_unique<rest_chunked_rx_request>();
      req->object     = obj;
      req->chunk      = io_object_segment{w, rs};
      req->file_size  = fsize;
      auto cpy        = std::make_unique<device_cpy_request>();
      cpy->stream     = stream;
      cpy->device_id  = device_id;
      cpy->copies.push_back(device_cpy_request::copy{/*dst=*/r.device_dst + (w - r.offset),
                                                     /*src=*/nullptr,
                                                     /*src_off=*/0,
                                                     /*size=*/rs});
      req->cpy_req = std::move(cpy);
      req->manager = manager;
      chunks.push_back(std::move(req));
    }
  }
  return rest_rx_request::create(std::move(chunks));
}

rest_reactor::request_type_ptr rest_reactor::prep_host_to_device_rxv_request(
  const reactor_config_type& cfg,
  const io_object_type& file,
  std::span<const io_host_device_range> ranges,
  rmm::cuda_stream_view stream,
  int device_id)
{
  if (ranges.empty()) { return rest_rx_request::create({}); }

  size_t const fsize  = file.size();
  size_t const bounce = cfg.bounce_block_size;
  auto const obj      = file.get_object_ref();

  std::vector<planned_device_segment> plan;
  plan.reserve(ranges.size());
  for (auto const& r : ranges) {
    if (r.size == 0) { continue; }
    if (r.device_dst == nullptr) {
      throw std::runtime_error(
        "rest_reactor::prep_host_to_device_rxv_request: range has no device destination");
    }
    if (!r.is_copy_window_valid()) {
      throw std::runtime_error(
        "rest_reactor::prep_host_to_device_rxv_request: copy window lies outside the read "
        "span");
    }
    size_t const copy_lo = r.copy_offset;
    size_t const copy_hi = std::min(r.copy_offset + r.copy_size, fsize);
    if (copy_lo >= copy_hi) {
      throw std::runtime_error(
        "rest_reactor::prep_host_to_device_rxv_request: range does not overlap the requested "
        "device range");
    }
    size_t const read_end = r.offset + std::min(r.size, fsize - r.offset);
    if (r.host_buffer != nullptr) {
      plan.push_back(planned_device_segment{
        r.offset, read_end - r.offset, r.host_buffer, r.device_dst, copy_lo, copy_hi});
      continue;
    }
    if (bounce == 0) {
      throw std::runtime_error(
        "rest_reactor::prep_host_to_device_rxv_request: device reads require a "
        "host_memory_resource on the reactor_context for bounce staging");
    }
    for (size_t w = r.offset; w < read_end; w += bounce) {
      size_t const rs       = std::min(bounce, read_end - w);
      size_t const piece_lo = std::max(copy_lo, w);
      size_t const piece_hi = std::min(copy_hi, w + rs);
      if (piece_lo >= piece_hi) { continue; }
      plan.push_back(planned_device_segment{
        w, rs, nullptr, r.device_dst + (piece_lo - copy_lo), piece_lo, piece_hi});
    }
  }
  if (plan.empty()) { return rest_rx_request::create({}); }

  // Each planned read becomes one tagged buffer; the tag survives coalescing and
  // cutting, so a buffer always knows the plan entry (and therefore the device
  // destination and copy window) it belongs to — no positional lockstep between
  // the plan and the chunks the planner hands back.
  size_t bytes_covered = 0;
  std::vector<planned_buffer> bufs;
  bufs.reserve(plan.size());
  for (auto const& p : plan) {
    bytes_covered += p.copy_hi - p.copy_lo;
    bufs.push_back(planned_buffer{p.offset, p.size, p.host_buffer, bufs.size()});
  }

  auto groups = plan_chunks(std::span<const planned_buffer>(bufs.data(), bufs.size()),
                            cfg.merge_max_gap,
                            cfg.max_chunk_size);

  auto manager = std::make_shared<request_manager>(bytes_covered, groups.size());

  std::vector<std::unique_ptr<rest_chunked_rx_request>> chunks;
  chunks.reserve(groups.size());
  for (auto const& g : groups) {
    auto cpy       = std::make_unique<device_cpy_request>();
    cpy->stream    = stream;
    cpy->device_id = device_id;
    cpy->copies.reserve(g.size());
    for (auto const& b : g) {
      if (b.hole) { continue; }
      assert(b.tag < plan.size() && "planned buffer carries a tag outside the plan");
      auto const& p        = plan[b.tag];
      size_t const data_lo = std::max(p.copy_lo, b.file_off);
      size_t const data_hi = std::min(p.copy_hi, b.file_off + b.len);
      if (data_lo < data_hi) {
        cpy->copies.push_back(device_cpy_request::copy{
          /*dst=*/p.device_dst + (data_lo - p.copy_lo),
          /*src=*/b.host != nullptr ? b.host + (data_lo - b.file_off) : nullptr,
          /*src_off=*/b.host != nullptr ? size_t{0} : (data_lo - b.file_off),
          /*size=*/data_hi - data_lo});
      }
    }

    auto req       = std::make_unique<rest_chunked_rx_request>();
    req->object    = obj;
    req->chunk     = to_segment(std::span<const planned_buffer>(g.data(), g.size()));
    req->file_size = fsize;
    req->cpy_req   = std::move(cpy);
    req->manager   = manager;
    chunks.push_back(std::move(req));
  }
  return rest_rx_request::create(std::move(chunks));
}

// ---------------------------------------------------------------------------
// synchronous paths
// ---------------------------------------------------------------------------

size_t rest_reactor::host_read(const io_object_type& file, size_t offset, size_t size, uint8_t* dst)
{
  if (size == 0) { return 0; }
  size = std::min(size, file.size() > offset ? file.size() - offset : size_t{0});
  if (size == 0) { return 0; }

  // Serve reads fully inside the suffix-range footer stash locally (the parquet
  // trailer/footer reads after a probe); a straddling read falls through to a GET.
  if (auto const& stash = file.stash(); stash) {
    size_t const lo = file.stash_window_lo();
    size_t const hi = lo + stash->size();
    if (offset >= lo && offset + size <= hi) {
      std::memcpy(dst, stash->data() + (offset - lo), size);
      return size;
    }
  }

  // Drive the blocking read through the worker's async pipeline (pooled
  // connections, parallel ranged GETs, the shared retry/backoff policy) and
  // synchronize on its future — rather than a one-shot easy handle that pays a
  // full TCP+TLS handshake per call and duplicates the retry logic.  Build the
  // request, grab its future BEFORE enqueue (which moves the chunks out), then
  // block: get() rethrows the first reported error or returns the byte count.
  auto req = prep_host_rx_request(
    _config, file, io_object_segment{offset, size, dst}, host_read_attribution::blocking);
  auto fut = req->get_future();
  enqueue(std::move(req));
  return std::move(fut).get();
}

rest_perf_snapshot rest_reactor::perf_snapshot() const noexcept
{
  rest_perf_snapshot s;
  s.chunk_get_ns_total       = _perf.chunk_get_ns_total.load(std::memory_order_relaxed);
  s.chunk_get_count          = _perf.chunk_get_count.load(std::memory_order_relaxed);
  s.chunk_get_ns_max         = _perf.chunk_get_ns_max.load(std::memory_order_relaxed);
  s.queue_wait_ns_total      = _perf.queue_wait_ns_total.load(std::memory_order_relaxed);
  s.queue_wait_count         = _perf.queue_wait_count.load(std::memory_order_relaxed);
  s.ttfb_ns                  = _perf.ttfb_ns.load(std::memory_order_relaxed);
  s.h2d_observed_ns_total    = _perf.h2d_observed_ns_total.load(std::memory_order_relaxed);
  s.h2d_observed_count       = _perf.h2d_observed_count.load(std::memory_order_relaxed);
  s.h2d_observed_ns_max      = _perf.h2d_observed_ns_max.load(std::memory_order_relaxed);
  s.retries_total            = _perf.retries_total.load(std::memory_order_relaxed);
  s.terminal_failures_total  = _perf.terminal_failures_total.load(std::memory_order_relaxed);
  s.device_stream_sync_total = _perf.device_stream_sync_total.load(std::memory_order_relaxed);
  s.payload_bytes_read_total = _perf.payload_bytes_read_total.load(std::memory_order_relaxed);
  s.blocking_host_get_count  = _perf.blocking_host_get_count.load(std::memory_order_relaxed);
  s.blocking_host_get_wall_ns_total =
    _perf.blocking_host_get_wall_ns_total.load(std::memory_order_relaxed);
  s.blocking_host_get_wall_ns_max =
    _perf.blocking_host_get_wall_ns_max.load(std::memory_order_relaxed);
  s.submit_slot_starved_total = _perf.submit_slot_starved_total.load(std::memory_order_relaxed);
  s.submit_work_starved_total = _perf.submit_work_starved_total.load(std::memory_order_relaxed);
  s.submit_added_total        = _perf.submit_added_total.load(std::memory_order_relaxed);
  s.inflight_sum              = _perf.inflight_sum.load(std::memory_order_relaxed);
  s.inflight_samples          = _perf.inflight_samples.load(std::memory_order_relaxed);
  s.inflight_max              = _perf.inflight_max.load(std::memory_order_relaxed);
  s.loop_idle_ns_total        = _perf.loop_idle_ns_total.load(std::memory_order_relaxed);
  s.loop_wall_ns_total        = _perf.loop_wall_ns_total.load(std::memory_order_relaxed);
  s.conn_opened_total         = _perf.conn_opened_total.load(std::memory_order_relaxed);
  s.retry_slowdown_total      = _perf.retry_slowdown_total.load(std::memory_order_relaxed);
  s.retry_server_err_total    = _perf.retry_server_err_total.load(std::memory_order_relaxed);
  s.retry_transport_total     = _perf.retry_transport_total.load(std::memory_order_relaxed);
  s.retry_short_read_total    = _perf.retry_short_read_total.load(std::memory_order_relaxed);
  s.retry_auth_total          = _perf.retry_auth_total.load(std::memory_order_relaxed);
  s.retry_delay_ns_total      = _perf.retry_delay_ns_total.load(std::memory_order_relaxed);
  s.curl_dns_ns_total         = _perf.curl_dns_ns_total.load(std::memory_order_relaxed);
  s.curl_connect_ns_total     = _perf.curl_connect_ns_total.load(std::memory_order_relaxed);
  s.curl_tls_ns_total         = _perf.curl_tls_ns_total.load(std::memory_order_relaxed);
  s.curl_ttfb_ns_total        = _perf.curl_ttfb_ns_total.load(std::memory_order_relaxed);
  s.curl_total_ns_total       = _perf.curl_total_ns_total.load(std::memory_order_relaxed);
  s.curl_timed_count          = _perf.curl_timed_count.load(std::memory_order_relaxed);
  return s;
}

void rest_reactor::reset_perf() noexcept
{
  auto z = [](std::atomic<std::uint64_t>& a) { a.store(0, std::memory_order_relaxed); };
  z(_perf.chunk_get_ns_total);
  z(_perf.chunk_get_count);
  z(_perf.chunk_get_ns_max);
  z(_perf.queue_wait_ns_total);
  z(_perf.queue_wait_count);
  z(_perf.ttfb_ns);
  z(_perf.h2d_observed_ns_total);
  z(_perf.h2d_observed_count);
  z(_perf.h2d_observed_ns_max);
  z(_perf.retries_total);
  z(_perf.terminal_failures_total);
  z(_perf.device_stream_sync_total);
  z(_perf.payload_bytes_read_total);
  z(_perf.blocking_host_get_count);
  z(_perf.blocking_host_get_wall_ns_total);
  z(_perf.blocking_host_get_wall_ns_max);
  z(_perf.submit_slot_starved_total);
  z(_perf.submit_work_starved_total);
  z(_perf.submit_added_total);
  z(_perf.inflight_sum);
  z(_perf.inflight_samples);
  z(_perf.inflight_max);
  z(_perf.loop_idle_ns_total);
  z(_perf.loop_wall_ns_total);
  z(_perf.conn_opened_total);
  z(_perf.retry_slowdown_total);
  z(_perf.retry_server_err_total);
  z(_perf.retry_transport_total);
  z(_perf.retry_short_read_total);
  z(_perf.retry_auth_total);
  z(_perf.retry_delay_ns_total);
  z(_perf.curl_dns_ns_total);
  z(_perf.curl_connect_ns_total);
  z(_perf.curl_tls_ns_total);
  z(_perf.curl_ttfb_ns_total);
  z(_perf.curl_total_ns_total);
  z(_perf.curl_timed_count);
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
  CTRACK_NAME("rest::head_object(blocking)");
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
        _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
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
      _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
      throw std::runtime_error("rest_reactor::head_object: " + last_error + " for " + obj.bucket +
                               "/" + obj.key);
    }
    if (attempt + 1 < _config.max_retry_attempts) {
      _perf.retries_total.fetch_add(1, std::memory_order_relaxed);
      SIRIUS_LOG_WARN("rest_reactor::head_object: retrying {}/{} after {} (attempt {}/{})",
                      obj.bucket,
                      obj.key,
                      last_error,
                      attempt + 1,
                      _config.max_retry_attempts);
      std::this_thread::sleep_for(compute_backoff(attempt, hc.retry_after, _config));
    }
  }
  _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
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
  CTRACK_NAME("rest::list_page(blocking)");
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

    // Control-plane response: the XML body is deliberately NOT credited to
    // chunk_get_count / payload_bytes_read_total — those budget object reads.
    if (rc == CURLE_OK && status == 200) { return body; }

    last_error =
      rc != CURLE_OK ? std::string(curl_easy_strerror(rc)) : ("HTTP " + std::to_string(status));
    bool const retriable =
      (rc != CURLE_OK && is_retriable_curl(rc)) || (rc == CURLE_OK && is_retriable_status(status));
    if (!retriable) {
      _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
      throw std::runtime_error("rest_reactor::list_page: " + last_error + " for " + bucket_s + "/" +
                               prefix_s);
    }
    if (attempt + 1 < _config.max_retry_attempts) {
      _perf.retries_total.fetch_add(1, std::memory_order_relaxed);
      SIRIUS_LOG_WARN("rest_reactor::list_page: retrying {}/{} after {} (attempt {}/{})",
                      bucket_s,
                      prefix_s,
                      last_error,
                      attempt + 1,
                      _config.max_retry_attempts);
      std::this_thread::sleep_for(compute_backoff(attempt, hc.retry_after, _config));
    }
  }
  _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
  throw std::runtime_error("rest_reactor::list_page: exhausted retries (" + last_error + ") for " +
                           bucket_s + "/" + prefix_s);
}

footer_probe rest_reactor::fetch_footer_suffix(std::string_view bucket,
                                               std::string_view key,
                                               std::size_t n)
{
  // Bind-time, blocking, and off the reactor's pooled connections: each call
  // opens a fresh TCP+TLS connection and every file's probe runs on one reactor.
  // Tracked because a wide scan pays this serially before any data moves.
  CTRACK_NAME("rest::fetch_footer_suffix(blocking)");
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

    auto const t0     = std::chrono::steady_clock::now();
    CURLcode const rc = curl_easy_perform(h.get());
    long status       = 0;
    curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

    // payload_bytes_read_total is always-on and per-attempt (see the async
    // worker's finish()), so credit every attempt's wire bytes outside the
    // perf_instrumentation gate.
    _perf.payload_bytes_read_total.fetch_add(sink.total_received, std::memory_order_relaxed);

    // suffix_write_cb aborts any non-206 body, so a CURLE_WRITE_ERROR here is our
    // own doing and the HTTP status is still valid; only a different curl error
    // (no HTTP status) is a genuine transport failure.
    if (rc != CURLE_OK && rc != CURLE_WRITE_ERROR) {
      last_error = std::string(curl_easy_strerror(rc));
      if (!is_retriable_curl(rc)) {
        _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
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
        // Account the footer suffix GET the same way the async chunk path does
        // (it replaces the tail+body GETs that used to run through the pipeline),
        // so bind-time footer reads stay visible in the perf snapshot.
        if (_config.perf_instrumentation) {
          auto const get_ns =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - t0)
                                         .count());
          _perf.chunk_get_ns_total.fetch_add(get_ns, std::memory_order_relaxed);
          _perf.chunk_get_count.fetch_add(1, std::memory_order_relaxed);
          atomic_max_relaxed(_perf.chunk_get_ns_max, get_ns);
          std::uint64_t expected = 0;
          _perf.ttfb_ns.compare_exchange_strong(expected, get_ns, std::memory_order_relaxed);
        }
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
      _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
      throw std::runtime_error("rest_reactor::fetch_footer_suffix: HTTP " + std::to_string(status) +
                               " for " + obj.bucket + "/" + obj.key);
    }
    if (attempt + 1 < _config.max_retry_attempts) {
      _perf.retries_total.fetch_add(1, std::memory_order_relaxed);
      SIRIUS_LOG_WARN("rest_reactor::fetch_footer_suffix: retrying {}/{} after {} (attempt {}/{})",
                      obj.bucket,
                      obj.key,
                      last_error,
                      attempt + 1,
                      _config.max_retry_attempts);
      std::this_thread::sleep_for(compute_backoff(attempt, sink.retry_after, _config));
    }
  }
  _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
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

/// One reusable I/O slot — the unit of concurrency, mirroring uring_reactor's
/// io_slot.  Owns its persistent easy handle and its dedicated pinned bounce
/// buffer (1:1 with the slot; null when no host memory resource was given), and
/// carries the per-request state whose address curl references for the duration
/// of a transfer (the URL string, header list, write/header targets).  The
/// slot_pool gates which slots are in use; a slot holds its acquisition token
/// while busy, so a slot is also its own bounce-buffer reservation.
struct io_slot {
  // Persistent slot resources (set once at pool creation, never reset):
  curl_easy_ptr easy;
  uint8_t* bounce{nullptr};

  // Held while the slot is in use; releasing it returns the slot to the pool.
  // For a bounce-staged device read it is moved into `copying` so the slot is
  // not reused until the H2D copy off its bounce buffer completes.
  slot_pool::token token;

  // Per-request state (cleared by reset() between requests):
  std::unique_ptr<rest_chunked_rx_request> req;
  std::string url;  // backs CURLOPT_URL
  curl_slist_ptr headers;
  buf_sink sink;
  header_capture hc;

  /// Clear the per-request state and release the slot back to its pool (a
  /// no-op when the token was already moved into `copying`).  The handle and
  /// bounce buffer persist.
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

/// Worker-loop state reachable from curl's C socket/timer callbacks.
struct worker_state {
  CURLM* multi{nullptr};
  int epoll_fd{-1};
  int curl_timer_fd{-1};
};

/// CURLMOPT_SOCKETFUNCTION: mirror curl's interest in @p s into epoll.
int rest_socket_cb(CURL* /*easy*/, curl_socket_t s, int what, void* userp, void* socketp)
{
  auto* ws = static_cast<worker_state*>(userp);
  if (what == CURL_POLL_REMOVE) {
    // Best-effort delete; the socket may already be gone (ENOENT/EBADF).
    ::epoll_ctl(ws->epoll_fd, EPOLL_CTL_DEL, s, nullptr);
    return 0;
  }
  uint32_t events = 0;
  if (what == CURL_POLL_IN || what == CURL_POLL_INOUT) { events |= EPOLLIN; }
  if (what == CURL_POLL_OUT || what == CURL_POLL_INOUT) { events |= EPOLLOUT; }
  epoll_event ev{};
  ev.events  = events;
  ev.data.fd = s;
  // socketp is curl's per-socket user pointer: null on first sighting (ADD),
  // non-null thereafter (MOD).  We only use it as an "already added" marker.
  int const op = (socketp == nullptr) ? EPOLL_CTL_ADD : EPOLL_CTL_MOD;
  if (socketp == nullptr) { curl_multi_assign(ws->multi, s, ws); }
  ::epoll_ctl(ws->epoll_fd, op, s, &ev);
  return 0;
}

/// CURLMOPT_TIMERFUNCTION: arm/disarm the curl timerfd.
int rest_timer_cb(CURLM* /*multi*/, long timeout_ms, void* userp)
{
  auto* ws = static_cast<worker_state*>(userp);
  itimerspec its{};  // all-zero => disarm
  if (timeout_ms == 0) {
    its.it_value.tv_nsec = 1;  // fire essentially immediately
  } else if (timeout_ms > 0) {
    its.it_value.tv_sec  = timeout_ms / 1000;
    its.it_value.tv_nsec = (timeout_ms % 1000) * 1'000'000L;
  }
  ::timerfd_settime(ws->curl_timer_fd, 0, &its, nullptr);
  return 0;
}

/// Drain (and discard) all pending reads from a non-blocking fd.
void drain_fd(int fd) noexcept
{
  uint64_t v = 0;
  while (::read(fd, &v, sizeof(v)) > 0) {}
}

}  // namespace

void rest_reactor::worker_loop(const std::stop_token& stop_token)
{
  // Wake the loop out of epoll_wait when shutdown is requested.
  std::stop_callback const stop_cb(stop_token, [this] { interrupt(); });

  try {
    curl_multi_ptr multi{curl_multi_init()};
    if (!multi) { throw std::runtime_error("rest_reactor: curl_multi_init failed"); }

    file_descriptor epoll_fd        = make_epoll_fd();
    file_descriptor curl_timer_fd   = make_timer_fd();
    file_descriptor retry_timer_fd  = make_timer_fd();
    file_descriptor upkeep_timer_fd = make_timer_fd();
    worker_state ws{multi.get(), epoll_fd.get(), curl_timer_fd.get()};

    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_SOCKETFUNCTION, &rest_socket_cb));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_SOCKETDATA, &ws));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_TIMERFUNCTION, &rest_timer_cb));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_TIMERDATA, &ws));
    // Parallel TCP streams rather than HTTP/2 multiplexing: with multiplexing
    // off, curl opens a separate connection per concurrent transfer (up to
    // MAX_HOST_CONNECTIONS == max_connections) instead of funneling every GET
    // over one h2 connection bounded by the server's stream limit.  Against S3,
    // independent connections give better aggregate throughput on large ranged
    // reads, and the slot pool's "N connections" model then actually holds.
    SIRIUS_CURLM_CHECK(
      curl_multi_setopt(multi.get(), CURLMOPT_PIPELINING, static_cast<long>(CURLPIPE_NOTHING)));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(
      multi.get(), CURLMOPT_MAX_HOST_CONNECTIONS, static_cast<long>(_config.max_connections)));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(
      multi.get(), CURLMOPT_MAXCONNECTS, static_cast<long>(_config.max_connections)));

    auto epoll_add = [&](int fd, uint32_t events) {
      epoll_event ev{};
      ev.events  = events;
      ev.data.fd = fd;
      if (::epoll_ctl(epoll_fd.get(), EPOLL_CTL_ADD, fd, &ev) != 0) {
        throw std::runtime_error(std::string("rest_reactor: epoll_ctl ADD failed: ") +
                                 std::strerror(errno));
      }
    };
    epoll_add(_wakeup_fd.get(), EPOLLIN);
    epoll_add(curl_timer_fd.get(), EPOLLIN);
    epoll_add(retry_timer_fd.get(), EPOLLIN);
    epoll_add(upkeep_timer_fd.get(), EPOLLIN);

    // Idle-connection keepalive: fire the upkeep timer periodically so idle
    // pooled connections get an HTTP/2 PING (gated per-connection by
    // CURLOPT_UPKEEP_INTERVAL_MS).  Disabled when upkeep_interval is zero.
    long const upkeep_ms = static_cast<long>(_config.upkeep_interval.count());
    if (upkeep_ms > 0) {
      itimerspec its{};
      its.it_value.tv_sec = its.it_interval.tv_sec = upkeep_ms / 1000;
      its.it_value.tv_nsec = its.it_interval.tv_nsec = (upkeep_ms % 1000) * 1'000'000L;
      ::timerfd_settime(upkeep_timer_fd.get(), 0, &its, nullptr);
    }

    // Slot pool: max_connections reusable io_slots, each owning its easy handle
    // (configured once with the static performance + TLS/timeout options) and,
    // when device staging is enabled, its dedicated pinned bounce buffer.  The
    // connection cache is shared via a worker-local curl_share so idle
    // connections survive across handles and are reachable by curl_easy_upkeep
    // — safe because only this worker thread touches it.  A slot_pool gates
    // which slots are in use (a free slot also owns a free bounce buffer); each
    // handle carries its slot index in CURLOPT_PRIVATE so a completion maps back
    // to its slot in O(1) with no per-request allocation or hashing.
    curl_share worker_share{/*share_connections=*/true};
    std::vector<io_slot> slots(_config.max_connections);
    slot_pool pool{_config.max_connections};

    std::vector<uint8_t*> bounce_bufs;  // one per slot when device staging is on
    if (_bounce_storage) {
      auto blocks = _bounce_storage->get_blocks();
      bounce_bufs.reserve(blocks.size());
      for (auto* b : blocks) {
        bounce_bufs.push_back(reinterpret_cast<uint8_t*>(b));
      }
    }

    for (std::size_t i = 0; i < _config.max_connections; ++i) {
      curl_easy_ptr h{curl_easy_init()};
      if (!h) { throw std::runtime_error("rest_reactor: curl_easy_init failed"); }
      configure_easy_handle(
        h.get(), worker_share.get(), upkeep_ms, static_cast<long>(_config.conn_max_age.count()));
      apply_request_opts(h.get(), _config);
      // Stable per-handle identity (the slot index); never changes across the
      // requests this handle serves.
      SIRIUS_CURL_CHECK(curl_easy_setopt(
        h.get(), CURLOPT_PRIVATE, reinterpret_cast<void*>(static_cast<intptr_t>(i))));
      slots[i].easy   = std::move(h);
      slots[i].bounce = i < bounce_bufs.size() ? bounce_bufs[i] : nullptr;
    }

    // Warm-up requests in flight.  Deliberately NOT slot handles: a slot handle
    // is configured for ranged chunk GETs and reused across requests, so
    // borrowing one would leave warm-up request options on a handle that later
    // serves a chunk.  These are throwaway handles on the same `worker_share`,
    // which is what matters -- the connection they open lands in that shared
    // cache and the slot handles pick it up from there.
    std::vector<curl_easy_ptr> warm_handles;
    std::vector<curl_slist_ptr> warm_headers;  // must outlive their transfers

    // Open `max_connections` connections against `bucket`, one bucket-scoped
    // ListObjectsV2 each.  The response is irrelevant -- a 403 from a credential
    // that cannot list completed the same handshake a 200 would have -- so
    // nothing here inspects status, retries, or reports failure.
    auto prime_connections = [&](std::string const& bucket) {
      // max-keys=0 is the cheapest well-formed bucket request there is: a signed
      // ListObjectsV2 that names no object and returns a near-empty body.
      constexpr std::string_view k_warm_query = "list-type=2&max-keys=0";
      for (std::size_t i = 0; i < _config.max_connections; ++i) {
        try {
          auto const authd =
            _ctx->authorizer()->authorize_list(bucket, k_warm_query, presign_ttl(_config));
          curl_easy_ptr h{curl_easy_init()};
          if (!h) { break; }
          configure_easy_handle(h.get(),
                                worker_share.get(),
                                upkeep_ms,
                                static_cast<long>(_config.conn_max_age.count()));
          apply_request_opts(h.get(), _config);
          curl_slist_ptr hdrs = build_header_list(authd.headers, nullptr);
          SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
          SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
          SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &write_discard));
          // Distinguishes a warm-up completion from a slot completion, whose
          // CURLOPT_PRIVATE is its slot index in [0, max_connections).
          SIRIUS_CURL_CHECK(
            curl_easy_setopt(h.get(), CURLOPT_PRIVATE, reinterpret_cast<void*>(intptr_t{-1})));
          // A fresh connection per handle is the entire point: without this the
          // second handle would reuse the first's connection and the pool would
          // end up one deep.
          SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_FRESH_CONNECT, 1L));
          if (curl_multi_add_handle(multi.get(), h.get()) != CURLM_OK) { break; }
          warm_headers.push_back(std::move(hdrs));
          warm_handles.push_back(std::move(h));
        } catch (std::exception const& e) {
          SIRIUS_LOG_DEBUG("rest_reactor: warm-up handle {} not issued: {}", i, e.what());
          break;
        }
      }
      SIRIUS_LOG_INFO(
        "rest_reactor: warming {} connections against bucket '{}'", warm_handles.size(), bucket);
    };

    auto maybe_prime = [&]() {
      if (!_warm_requested.exchange(false, std::memory_order_acq_rel)) { return; }
      // A round still in flight means the pool is already being filled; a second
      // round on top of it would only open connections the first is opening.
      if (!warm_handles.empty()) { return; }
      std::string bucket;
      {
        std::lock_guard lk{_warm_mtx};
        bucket = _warm_bucket;
      }
      if (bucket.empty()) { return; }
      prime_connections(bucket);
    };

    int running  = 0;
    int inflight = 0;  // GETs currently added to the multi (excludes parked H2D)

    // -- retry scheduling --------------------------------------------------
    // A min-heap of chunks keyed by their due time.  When retry_timer_fd fires,
    // every chunk whose due time has passed moves into `ready`, which submit()
    // drains ahead of the inbound queue.  The timer is always armed to the
    // earliest due time in the heap.
    struct retry_entry {
      std::chrono::steady_clock::time_point due;
      std::unique_ptr<rest_chunked_rx_request> req;
    };
    auto const retry_cmp = [](const retry_entry& a, const retry_entry& b) { return a.due > b.due; };
    std::vector<retry_entry> retry_heap;
    std::deque<std::unique_ptr<rest_chunked_rx_request>> ready;

    // -- device staging ----------------------------------------------------
    // A reactor-staged device read (null-buffer chunk) lands in its slot's
    // bounce buffer, then an async H2D copy moves the bytes to the device.  The
    // slot's pool token is held in `copying` (alongside the copy's event) until
    // that copy completes — so the slot, and its bounce buffer, are not reused
    // meanwhile.  Crucially, the chunk is NOT reported complete until the event
    // is observed done: the bytes are not actually on the device until the copy
    // off the bounce buffer finishes, so the manager (and the byte count to
    // credit) ride along here and chunk_complete / report_error is deferred to
    // poll_copy_completions.  Events are pooled per device, indexed by slot.
    struct parked_copy {
      slot_pool::token token;
      cucascade::cuda::cuda_event* event{nullptr};
      std::shared_ptr<request_manager> manager;
      std::size_t bytes{0};
    };
    std::unordered_map<int, std::vector<cucascade::cuda::cuda_event>> copy_events;
    std::vector<parked_copy> copying;
    if (_bounce_storage) {
      int const n_dev = rmm::get_num_cuda_devices();
      for (int d = 0; d < n_dev; ++d) {
        rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{d}};
        auto& evs = copy_events[d];
        evs.reserve(_config.max_connections);
        std::generate_n(std::back_inserter(evs), _config.max_connections, [] {
          return cucascade::cuda::cuda_event{cudaEventDisableTiming};
        });
      }
    }

    auto poll_copy_completions = [&]() {
      using query_status = cucascade::cuda::event::query_result;
      // Credit (or fail) each bounce-staged chunk only now that its H2D copy is
      // actually done — the device bytes were not valid until this point.  On
      // success report the chunk complete; on a copy error fail the request
      // rather than silently dropping it.  Either way the entry is erased, which
      // drops its token and returns the slot (and bounce buffer) to the pool.
      for (auto it = copying.begin(); it != copying.end();) {
        query_status const st = it->event->query();
        if (st == query_status::in_progress) {
          ++it;
          continue;
        }
        if (st == query_status::success) {
          it->manager->chunk_complete(it->bytes);
        } else {
          _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
          it->manager->report_error(
            std::make_exception_ptr(std::runtime_error("rest_reactor: device H2D copy failed")));
        }
        it = copying.erase(it);
      }
    };

    auto arm_retry_timer = [&]() {
      itimerspec its{};  // all-zero => disarm
      if (!retry_heap.empty()) {
        auto const now = std::chrono::steady_clock::now();
        auto const due = retry_heap.front().due;
        auto ns        = due > now
                           ? std::chrono::duration_cast<std::chrono::nanoseconds>(due - now).count()
                           : std::int64_t{1};
        if (ns <= 0) { ns = 1; }
        its.it_value.tv_sec  = ns / 1'000'000'000;
        its.it_value.tv_nsec = ns % 1'000'000'000;
      }
      ::timerfd_settime(retry_timer_fd.get(), 0, &its, nullptr);
    };

    // Re-enqueue a chunk for a later attempt after a backoff.  @p is_auth picks
    // the bounded HTTP-403 (re-presign) budget instead of the transient-error
    // budget, so a stale presigned URL gets a few fresh-signature retries while
    // a genuine AccessDenied still fails fast.
    auto schedule_retry = [&](std::unique_ptr<rest_chunked_rx_request> req,
                              std::string const& retry_after,
                              bool is_auth,
                              std::string const& reason) {
      std::size_t& counter = is_auth ? req->auth_attempt : req->attempt;
      std::size_t const max_attempts =
        is_auth ? _config.max_auth_retry_attempts : _config.max_retry_attempts;
      if (counter + 1 >= max_attempts) {
        _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
        req->manager->report_error(std::make_exception_ptr(std::runtime_error(
          "rest_reactor: exhausted retries for " + req->object.bucket + "/" + req->object.key)));
        return;
      }
      // Backoff tracks the transient-attempt count; an auth retry re-presigns
      // and reuses the current step without inflating it.
      auto const delay = compute_backoff(req->attempt, retry_after, _config);
      _perf.retries_total.fetch_add(1, std::memory_order_relaxed);
      _perf.retry_delay_ns_total.fetch_add(
        static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(delay).count()),
        std::memory_order_relaxed);
      SIRIUS_LOG_WARN("rest_reactor: retrying {}/{} after {} (attempt {}/{})",
                      req->object.bucket,
                      req->object.key,
                      reason,
                      counter + 1,
                      max_attempts);
      counter += 1;
      retry_heap.push_back(retry_entry{std::chrono::steady_clock::now() + delay, std::move(req)});
      std::push_heap(retry_heap.begin(), retry_heap.end(), retry_cmp);
      arm_retry_timer();
    };

    auto setup_easy = [&](io_slot& s) {
      CTRACK_NAME("rest::setup_easy");
      CURL* const h = s.easy.get();
      // SigV4 presign: 2x SHA256 + 5x HMAC-SHA256 per attempt, on this worker
      // thread.  Tracked separately because it is pure CPU inside the event
      // loop — if it shows up, every reactor is signing instead of polling.
      authorized_request authd;
      {
        CTRACK_NAME("rest::presign");
        authd =
          _ctx->authorizer()->authorize(s.req->object, request_method::GET, presign_ttl(_config));
      }
      s.url           = std::move(authd.url);
      s.sink.buffers  = std::span<iovec>(s.req->chunk.buffers);
      s.sink.capacity = s.req->chunk.size;
      s.sink.reset();
      s.hc.reset();
      std::string const range = range_header(s.req->chunk.offset, s.req->chunk.size);
      s.headers               = build_header_list(authd.headers, &range);
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_URL, s.url.c_str()));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HTTPHEADER, s.headers.get()));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &write_to_sink));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_WRITEDATA, &s.sink));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, &capture_header));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HEADERDATA, &s.hc));
    };

    auto submit = [&]() {
      CTRACK_NAME("rest::submit");
      // Sample the multi handle's occupancy once per pass, before we top it up:
      // the mean of this series is the connection concurrency the link actually
      // sees, which is what "are we driving S3 hard enough?" reduces to.
      {
        auto const now = static_cast<std::uint64_t>(inflight);
        _perf.inflight_sum.fetch_add(now, std::memory_order_relaxed);
        _perf.inflight_samples.fetch_add(1, std::memory_order_relaxed);
        auto prev = _perf.inflight_max.load(std::memory_order_relaxed);
        while (now > prev &&
               !_perf.inflight_max.compare_exchange_weak(prev, now, std::memory_order_relaxed)) {}
      }
      // Acquire a slot (and thus a bounce buffer) up front; an invalid token
      // means all slots are busy.  A token taken for a skipped/empty dequeue is
      // released by its RAII destructor at continue/break.
      while (true) {
        slot_pool::token tok = pool.try_acquire_token();
        if (!tok) {
          // Every connection is busy: the reactor is saturated and the queue
          // (if non-empty) is waiting on S3, not on us.
          _perf.submit_slot_starved_total.fetch_add(1, std::memory_order_relaxed);
          break;
        }

        // Submission priority: due retries (ready) ahead of fresh inbound work
        // so a backed-off request is not starved by new ones.
        std::unique_ptr<rest_chunked_rx_request> dr;
        if (!ready.empty()) {
          dr = std::move(ready.front());
          ready.pop_front();
        } else if (_requests.try_dequeue(dr)) {
          // The chunk has left the queue; it is now this reactor's in-flight work
          // and no longer counts toward the depth dispatch balances against.
          if (dr) { _queued_bytes.fetch_sub(dr->chunk.size, std::memory_order_relaxed); }
        } else {
          // Slots to spare but nothing to run: the producer is the bottleneck.
          _perf.submit_work_starved_total.fetch_add(1, std::memory_order_relaxed);
          break;
        }
        if (!dr) { continue; }
        if (dr->manager->has_error()) {
          dr.reset();
          continue;
        }

        int const i = tok.slot_index();
        io_slot& s  = slots[static_cast<size_t>(i)];
        // A bounce-staged device read must re-bind to THIS slot's bounce on
        // every (re)submission.  A retried request still carries the previous
        // attempt's set_data (chunk.data() == that now-freed slot's bounce) and
        // staged_through_bounce, so without the second term needs_bounce would
        // be false on retry: the sink would fill — and the parked H2D would
        // drain from — a foreign slot's bounce while this slot's token is parked.
        bool const needs_bounce =
          dr->is_device() && (!dr->chunk.is_buffer_allocated() || dr->staged_through_bounce);
        if (needs_bounce && s.bounce == nullptr) {
          // Device staging requested but no host memory resource was configured.
          _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
          dr->manager->report_error(std::make_exception_ptr(std::runtime_error(
            "rest_reactor: device staging unavailable (no host memory resource)")));
          dr.reset();
          continue;  // token released, slot stays free
        }

        s.req = std::move(dr);
        if (_config.perf_instrumentation) {
          auto const now  = std::chrono::steady_clock::now();
          s.req->t_submit = now;
          if (s.req->attempt == 0) {
            auto const wait_ns = static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(now - s.req->t_enqueue).count());
            _perf.queue_wait_ns_total.fetch_add(wait_ns, std::memory_order_relaxed);
            _perf.queue_wait_count.fetch_add(1, std::memory_order_relaxed);
          }
        }
        if (needs_bounce) {
          s.req->chunk.set_data(s.bounce);
          // Record bounce-staging before finish() reads it: set_data has just
          // made is_buffer_allocated() true, so finish() must rely on this flag
          // (not the chunk) to take the event-synchronized recycle path and hold
          // the slot's bounce until the H2D copy off it completes.
          s.req->staged_through_bounce = true;
        }
        s.token = std::move(tok);  // slot holds its token while in use
        setup_easy(s);
        curl_multi_add_handle(multi.get(), s.easy.get());
        ++inflight;
        _perf.submit_added_total.fetch_add(1, std::memory_order_relaxed);
      }
    };

    // Handle one completed transfer.  Returns true iff the slot was parked in
    // `copying` (held until its H2D copy finishes); otherwise the caller
    // recycles the slot immediately.
    auto finish = [&](int i, CURLcode rc, long status) -> bool {
      io_slot& s = slots[static_cast<size_t>(i)];
      // Always-on: credit the HTTP response body bytes this attempt received
      // (write_to_sink already summed them in sink.total_received) BEFORE any
      // success / retry / terminal branching, so a 503 -> retry -> 206 counts
      // both bodies and short / failed reads are reflected in the byte budget.
      _perf.payload_bytes_read_total.fetch_add(s.sink.total_received, std::memory_order_relaxed);
      auto& req           = *s.req;
      bool const ok_range = (status == 206) || (status == 200 && req.chunk.offset == 0);
      // A 206 must report, via Content-Range, that it delivered the exact range
      // we asked for.  The write callback scatters the body in arrival order
      // with no idea what file offset it covers, so a 206 whose body is a
      // *different* range than requested (a misbehaving proxy/CDN/cache, or a
      // multipart/byteranges response curl does not parse) would otherwise be
      // accepted as correct data — silent corruption.  Validate the start
      // offset before trusting any 206 body; a mismatch (or an unparsable /
      // missing Content-Range) is a terminal error, not a transient one.
      if (rc == CURLE_OK && status == 206) {
        auto const start = content_range_start(s.hc.content_range);
        if (!start || *start != req.chunk.offset) {
          _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
          req.manager->report_error(std::make_exception_ptr(std::runtime_error(
            "rest_reactor: 206 Content-Range mismatch (got '" + s.hc.content_range +
            "', requested offset " + std::to_string(req.chunk.offset) + ") for " +
            req.object.bucket + "/" + req.object.key)));
          return false;
        }
      }
      if (rc == CURLE_OK && ok_range && s.sink.written >= req.chunk.size) {
        if (_config.perf_instrumentation) {
          auto const get_ns =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - req.t_submit)
                                         .count());
          // Every completed ranged GET bumps chunk_get_*, blocking host_reads
          // included. A blocking single host_read additionally bumps
          // blocking_host_get_* — the two are additive, not disjoint.
          _perf.chunk_get_ns_total.fetch_add(get_ns, std::memory_order_relaxed);
          _perf.chunk_get_count.fetch_add(1, std::memory_order_relaxed);
          atomic_max_relaxed(_perf.chunk_get_ns_max, get_ns);
          std::uint64_t expected = 0;
          _perf.ttfb_ns.compare_exchange_strong(expected, get_ns, std::memory_order_relaxed);
          if (req.perf_blocking_host_get) {
            _perf.blocking_host_get_count.fetch_add(1, std::memory_order_relaxed);
            _perf.blocking_host_get_wall_ns_total.fetch_add(get_ns, std::memory_order_relaxed);
            atomic_max_relaxed(_perf.blocking_host_get_wall_ns_max, get_ns);
          }
        }
        if (req.is_device()) {
          // Issue the async H2D copy.  Bounce-staged reads need a CUDA event so
          // the slot (its bounce buffer) is only reused once the copy off it
          // completes; caller-buffer reads detach (the caller's stream orders
          // the copy).
          bool const needs_event          = req.needs_event_for_synchronization();
          cucascade::cuda::cuda_event* ev = nullptr;
          cudaEvent_t cev                 = nullptr;
          if (needs_event) {
            ev  = &copy_events[req.cpy_req->device_id][static_cast<size_t>(i)];
            cev = ev->get();
          }
          std::chrono::steady_clock::time_point h2d_start;
          if (_config.perf_instrumentation) { h2d_start = std::chrono::steady_clock::now(); }
          cudaError_t const err = req.copy_h2d_async(cev);
          if (err != cudaSuccess) {
            _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
            req.manager->report_error(err);
            return false;
          }
          if (_config.perf_instrumentation) {
            auto const h2d_ns =
              static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - h2d_start)
                                           .count());
            _perf.h2d_observed_ns_total.fetch_add(h2d_ns, std::memory_order_relaxed);
            _perf.h2d_observed_count.fetch_add(1, std::memory_order_relaxed);
            atomic_max_relaxed(_perf.h2d_observed_ns_max, h2d_ns);
          }
          if (needs_event) {
            // Bounce buffer is still feeding the copy: hand the slot's token to
            // `copying` (with the event, manager and byte count) so the slot —
            // and its bounce buffer — is only reused once the copy completes,
            // and the chunk is credited only then (the device bytes are not yet
            // valid).  Clear the rest of the per-request state now so a shutdown
            // cancel won't touch a done request.
            copying.push_back(parked_copy{std::move(s.token), ev, req.manager, s.sink.written});
            s.reset();
            return true;
          }
          // Caller-buffer device read: the caller's stream orders the copy, so
          // it is safe to credit the chunk now and recycle the slot.
          req.manager->chunk_complete(s.sink.written);
          return false;
        }
        req.manager->chunk_complete(s.sink.written);
        return false;
      }
      if (rc == CURLE_OK && status == 200 && req.chunk.offset != 0) {
        // Server ignored Range and returned the whole object: the bytes start
        // at offset 0, not req.offset — non-retriable, would loop forever.
        _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
        req.manager->report_error(std::make_exception_ptr(
          std::runtime_error("rest_reactor: server ignored Range (HTTP 200) for " +
                             req.object.bucket + "/" + req.object.key)));
        return false;
      }
      // Error or truncated transfer.  A short read or a transient HTTP / curl
      // failure is retried (re-authorized and re-submitted after a backoff);
      // anything else is terminal.  Retry moves the chunk out of the slot into
      // the heap, so the slot/connection is freed for other work during backoff.
      bool const short_read = rc == CURLE_OK && ok_range && s.sink.written < req.chunk.size;
      bool const retriable  = short_read || (rc != CURLE_OK && is_retriable_curl(rc)) ||
                             (rc == CURLE_OK && is_retriable_status(status));
      // A 403 is most often a presigned URL that expired while queued; re-issue
      // with a fresh signature a bounded number of times before giving up.
      bool const auth_retriable = rc == CURLE_OK && status == 403;
      if (retriable || auth_retriable) {
        // Attribute the cause before scheduling: "throttled by S3" and "the
        // link dropped a connection" call for opposite reactions, and the
        // aggregate retries_total cannot tell them apart.
        if (short_read) {
          _perf.retry_short_read_total.fetch_add(1, std::memory_order_relaxed);
        } else if (rc != CURLE_OK) {
          _perf.retry_transport_total.fetch_add(1, std::memory_order_relaxed);
        } else if (status == 429 || status == 503) {
          _perf.retry_slowdown_total.fetch_add(1, std::memory_order_relaxed);
        } else if (auth_retriable && !retriable) {
          _perf.retry_auth_total.fetch_add(1, std::memory_order_relaxed);
        } else {
          _perf.retry_server_err_total.fetch_add(1, std::memory_order_relaxed);
        }
        std::string const reason =
          rc != CURLE_OK ? std::string(curl_easy_strerror(rc))
                         : (short_read ? "short read" : "HTTP " + std::to_string(status));
        schedule_retry(std::move(s.req),
                       s.hc.retry_after,
                       /*is_auth=*/auth_retriable && !retriable,
                       reason);
        return false;
      }
      std::string const msg = rc != CURLE_OK
                                ? std::string(curl_easy_strerror(rc))
                                : (ok_range ? "short read" : "HTTP " + std::to_string(status));
      _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
      req.manager->report_error(std::make_exception_ptr(std::runtime_error(
        "rest_reactor: " + msg + " for " + req.object.bucket + "/" + req.object.key)));
      return false;
    };

    auto process_completions = [&]() {
      CURLMsg* msg = nullptr;
      int in_queue = 0;
      while ((msg = curl_multi_info_read(multi.get(), &in_queue)) != nullptr) {
        if (msg->msg != CURLMSG_DONE) { continue; }
        CTRACK_NAME("rest::completion");
        CURL* const h = msg->easy_handle;
        // A warm-up request.  Its connection is already back in the worker's shared
        // cache, which was the whole point, and the response is irrelevant.  Bail
        // before the accounting below: charging its handshake to
        // conn_opened_total would corrupt the one counter that says whether the
        // READ path is reconnecting.
        {
          char* warm_priv = nullptr;
          curl_easy_getinfo(h, CURLINFO_PRIVATE, &warm_priv);
          if (reinterpret_cast<intptr_t>(warm_priv) < 0) {
            curl_multi_remove_handle(multi.get(), h);
            std::erase_if(warm_handles, [h](curl_easy_ptr const& p) { return p.get() == h; });
            // The header lists back the transfers, so they can only go once the
            // last of them is done.
            if (warm_handles.empty()) { warm_headers.clear(); }
            continue;
          }
        }
        CURLcode const rc = msg->data.result;
        long status       = 0;
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
        // Connection churn: >0 means this transfer could not reuse a pooled
        // connection and paid a fresh TCP (+TLS) handshake.  Always on — it is
        // one getinfo and it is the difference between "S3 is slow" and "we
        // keep reconnecting".
        {
          long num_connects = 0;
          if (curl_easy_getinfo(h, CURLINFO_NUM_CONNECTS, &num_connects) == CURLE_OK &&
              num_connects > 0) {
            _perf.conn_opened_total.fetch_add(static_cast<std::uint64_t>(num_connects),
                                              std::memory_order_relaxed);
          }
        }
        if (_config.perf_instrumentation) {
          // libcurl's phase timings are cumulative from transfer start, so the
          // per-phase cost is each successive difference.
          curl_off_t dns = 0, conn = 0, tls = 0, start = 0, total = 0;
          curl_easy_getinfo(h, CURLINFO_NAMELOOKUP_TIME_T, &dns);
          curl_easy_getinfo(h, CURLINFO_CONNECT_TIME_T, &conn);
          curl_easy_getinfo(h, CURLINFO_APPCONNECT_TIME_T, &tls);
          curl_easy_getinfo(h, CURLINFO_STARTTRANSFER_TIME_T, &start);
          curl_easy_getinfo(h, CURLINFO_TOTAL_TIME_T, &total);
          auto const us_to_ns = [](curl_off_t v) {
            return v > 0 ? static_cast<std::uint64_t>(v) * 1000U : std::uint64_t{0};
          };
          _perf.curl_dns_ns_total.fetch_add(us_to_ns(dns), std::memory_order_relaxed);
          _perf.curl_connect_ns_total.fetch_add(us_to_ns(conn > dns ? conn - dns : 0),
                                                std::memory_order_relaxed);
          _perf.curl_tls_ns_total.fetch_add(us_to_ns(tls > conn ? tls - conn : 0),
                                            std::memory_order_relaxed);
          _perf.curl_ttfb_ns_total.fetch_add(us_to_ns(start), std::memory_order_relaxed);
          _perf.curl_total_ns_total.fetch_add(us_to_ns(total), std::memory_order_relaxed);
          _perf.curl_timed_count.fetch_add(1, std::memory_order_relaxed);
        }
        // Recover the slot index stashed in CURLOPT_PRIVATE at pool creation.
        char* priv = nullptr;
        curl_easy_getinfo(h, CURLINFO_PRIVATE, &priv);
        int const i = static_cast<int>(reinterpret_cast<intptr_t>(priv));
        curl_multi_remove_handle(multi.get(), h);
        --inflight;
        // finish() returns true when it parked the slot in `copying` (device
        // H2D in flight) — poll_copy_completions recycles it once the event
        // clears.  Otherwise reset() releases the slot's token back to the pool.
        if (!finish(i, rc, status)) { slots[static_cast<size_t>(i)].reset(); }
      }
    };

    std::vector<epoll_event> events{};
    events.resize(_config.max_connections);
    maybe_prime();  // a warm-up asked for before the worker was up
    submit();       // kickstart anything already queued
    // Duty-cycle accounting: charge each loop iteration's wall time to "idle" or
    // "busy" by whether anything was on the wire when the iteration began.  One
    // clock read per iteration, and epoll_wait is the only blocking point, so
    // this attributes the whole loop without sampling error worth caring about.
    auto loop_mark     = std::chrono::steady_clock::now();
    bool span_was_idle = inflight == 0;
    while (!stop_token.stop_requested()) {
      // Block indefinitely when idle; while H2D copies are outstanding, poll on
      // a short timeout so completed copies release their bounce slots promptly.
      int const timeout_ms = copying.empty() ? -1 : 1;
      int n                = 0;
      {
        // Time parked here is the reactor having nothing to do.  Compare
        // against rest::submit + rest::write_to_sink: a reactor that is mostly
        // in epoll_wait while queries stall is starved of work, not of link.
        CTRACK_NAME("rest::epoll_wait(idle)");
        n = ::epoll_wait(epoll_fd.get(), events.data(), _config.max_connections, timeout_ms);
      }
      if (n < 0) {
        if (errno == EINTR) { continue; }
        throw std::runtime_error(std::string("rest_reactor: epoll_wait failed: ") +
                                 std::strerror(errno));
      }
      for (int i = 0; i < n; ++i) {
        int const fd = events[i].data.fd;
        if (fd == _wakeup_fd.get()) {
          drain_fd(_wakeup_fd.get());
        } else if (fd == curl_timer_fd.get()) {
          drain_fd(curl_timer_fd.get());
          CTRACK_NAME("rest::curl_action(timeout)");
          curl_multi_socket_action(multi.get(), CURL_SOCKET_TIMEOUT, 0, &running);
        } else if (fd == retry_timer_fd.get()) {
          drain_fd(retry_timer_fd.get());
          auto const now = std::chrono::steady_clock::now();
          while (!retry_heap.empty() && retry_heap.front().due <= now) {
            std::pop_heap(retry_heap.begin(), retry_heap.end(), retry_cmp);
            ready.push_back(std::move(retry_heap.back().req));
            retry_heap.pop_back();
          }
          arm_retry_timer();
        } else if (fd == upkeep_timer_fd.get()) {
          drain_fd(upkeep_timer_fd.get());
          // Keep idle pooled connections warm.  Only when fully idle — an
          // in-flight transfer already keeps its connection active, and this
          // keeps upkeep off the hot path.  One call walks the worker's shared
          // connection cache, so any pooled handle covers all of them.
          if (inflight == 0 && !slots.empty()) { curl_easy_upkeep(slots.front().easy.get()); }
        } else {
          int ev_bitmask = 0;
          if (events[i].events & EPOLLIN) { ev_bitmask |= CURL_CSELECT_IN; }
          if (events[i].events & EPOLLOUT) { ev_bitmask |= CURL_CSELECT_OUT; }
          if (events[i].events & (EPOLLERR | EPOLLHUP)) { ev_bitmask |= CURL_CSELECT_ERR; }
          // Everything expensive about a transfer happens inside here: TLS
          // record decryption and the write_to_sink memcpy both run on this
          // thread, under this call.  It is per socket-event, not per byte, so
          // the probe is cheap — unlike one inside write_to_sink itself.
          CTRACK_NAME("rest::curl_action(socket)");
          curl_multi_socket_action(multi.get(), fd, ev_bitmask, &running);
        }
      }
      process_completions();
      poll_copy_completions();
      maybe_prime();
      submit();
      {
        // Close the span that began after the previous submit() — that is the
        // window whose emptiness `span_was_idle` describes — then open the next.
        auto const now     = std::chrono::steady_clock::now();
        auto const elapsed = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(now - loop_mark).count());
        loop_mark = now;
        _perf.loop_wall_ns_total.fetch_add(elapsed, std::memory_order_relaxed);
        if (span_was_idle) {
          _perf.loop_idle_ns_total.fetch_add(elapsed, std::memory_order_relaxed);
        }
        span_was_idle = inflight == 0;
      }
    }

    // Drain on shutdown.  First wait for in-flight H2D copies to finish so the
    // bounce storage is not freed (at reactor destruction) while a copy still
    // reads from it.  chunk_complete was deferred for these, so credit each one
    // now that its copy has landed (or fail it if the copy errored) — otherwise
    // the request_manager would never reach total_chunks.
    for (auto& pc : copying) {
      try {
        _perf.device_stream_sync_total.fetch_add(1, std::memory_order_relaxed);
        pc.event->synchronize();
        pc.manager->chunk_complete(pc.bytes);
      } catch (const std::exception& e) {
        SIRIUS_LOG_ERROR("rest_reactor: copy-event synchronize on shutdown failed: {}", e.what());
        pc.manager->report_error(std::make_exception_ptr(std::runtime_error(
          std::string("rest_reactor: device H2D copy failed on shutdown: ") + e.what())));
      }
    }
    copying.clear();

    // Warm-up handles carry no request and nobody is waiting on them, so they
    // just need detaching before their storage goes.
    for (auto& h : warm_handles) {
      curl_multi_remove_handle(multi.get(), h.get());
    }
    warm_handles.clear();
    warm_headers.clear();

    // Detach in-flight handles and cancel every outstanding request (in-flight,
    // retry-scheduled, ready, queued) so no future is left unfulfilled.
    for (auto& s : slots) {
      if (s.req) {
        curl_multi_remove_handle(multi.get(), s.easy.get());
        s.req->manager->report_error(std::make_error_code(std::errc::operation_canceled));
        s.reset();
      }
    }
    for (auto& e : retry_heap) {
      if (e.req) {
        e.req->manager->report_error(std::make_error_code(std::errc::operation_canceled));
      }
    }
    retry_heap.clear();
    for (auto& r : ready) {
      if (r) { r->manager->report_error(std::make_error_code(std::errc::operation_canceled)); }
    }
    ready.clear();
  } catch (const std::exception& e) {
    SIRIUS_LOG_ERROR("rest_reactor worker_loop: {}", e.what());
  }

  std::unique_ptr<rest_chunked_rx_request> dr;
  while (_requests.try_dequeue(dr)) {
    if (dr) {
      _queued_bytes.fetch_sub(dr->chunk.size, std::memory_order_relaxed);
      dr->manager->report_error(std::make_error_code(std::errc::operation_canceled));
    }
    dr.reset();
  }
}

}  // namespace sirius::io::rest
